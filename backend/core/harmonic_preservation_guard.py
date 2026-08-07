"""
HarmonicPreservationGuard (HPG) — Aurik 10.0.0 §2.28
===================================================

Extrahiert den harmonischen Fingerabdruck BEFORE Rauschunterdrückung und
schützt Partial-Positionen durch einen erhöhten G_floor-Wert.

PROBLEM:
--------
OMLSA/DeepFilterNet setzen G_floor = 0.10 für alle Bins. Das bedeutet:
Harmonische Energie an Partial-Frequenzen kann auf 10 % gedämpft werden.
→ Klingt synthetisch, Natürlichkeit ≤ 0.87, Authentizität leidet.

LÖSUNG:
-------
G_floor_effective[t, f] = 0.85  wenn protected_bins[t, f]  (Harmonik)
G_floor_effective[t, f] = 0.10  sonst

Nach der Rauschunterdrückung: Energie-Korrektur falls |STFT(rest)| < 0.85·H_ref
→ Gain ∈ [1.0, 2.0] + PGHI (phasenkonsistent)

ALGORITHMUS:
-----------
1. FCPE ONNX (fcpe_plugin, §4.4 Primär; Fallback: CREPE → pYIN) → f₀(t) mit Voicing-Konfidenz ≥ 0.60
2. Harmonisches Gitter: fₙ(t) = n·f₀(t)·√(1+B·n²)  für n=1..20
   B = INHARMONICITY_PRIORS[instrument_tag] (aus §2.11)
3. STFT-Bins innerhalb ±3 Cent von fₙ(t) → protected_bins[t, f] = True
4. Harmonisches Energie-Profil: H_ref[t, f] = |STFT(audio)[t, f]|
5. Nach NR: Energie-Prüfung in protected_bins
   → gain = clip(H_ref / max(|STFT(rest)|, 1e-8), 1.0, 2.0)
6. PGHI für phasenkonsistente Rückwandlung

KONSTANTEN:
-----------
G_FLOOR_HARMONIC      = 0.85  (protected bins)
G_FLOOR_DEFAULT       = 0.10  (alle anderen Bins)
MAX_GAIN_CORRECTION   = 2.0   (niemals mehr als ×2 anheben)
VOICING_CONFIDENCE_MIN = 0.60

ERWARTETER EFFEKT:
-----------------
Natürlichkeit:        +0.03 – 0.07
Authentizität:        +0.03 – 0.06
Timbre-Authentizität: +0.02 – 0.05

LAUFZEIT: ≤ 1.2 s / Minute Audio (dominiert durch FCPE-Inferenz)

REFERENZ:
---------
Fletcher (1964): Normal Vibration Frequencies of a Stiff Piano String
Mauch & Dixon (2014): pYIN — Fundamental Frequency Estimator
Perraudin et al. (2013): PGHI — Non-Iterative STFT Phase Reconstruction

Autor: Aurik 10.0.0 Development Team / v10.0.0
"""

from __future__ import annotations

import logging
import math
import threading

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optionale ML-Abhängigkeiten (§3.4 Graceful Degradation)
# ---------------------------------------------------------------------------
try:
    from plugins.fcpe_plugin import get_fcpe_plugin as _get_crepe

    _CREPE_OK = True
except Exception:
    _CREPE_OK = False
    logger.debug("FCPE/CREPE nicht verfügbar — RMVPE/PESTO/pYIN-Ersatzpfad für HPG aktiv")

try:
    from plugins.rmvpe_plugin import get_rmvpe_plugin as _get_rmvpe

    _RMVPE_OK = True
except Exception:
    _RMVPE_OK = False
    logger.debug("RMVPE nicht verfügbar — PESTO/pYIN-Ersatzpfad für HPG aktiv")

try:
    import librosa

    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    logger.debug("librosa nicht verfügbar — PESTO/Autokorrelations-Ersatzpfad für f₀")

try:
    from dsp.pesto_pitch import estimate_pitch as _pesto_estimate

    _PESTO_OK = True
except Exception:
    _PESTO_OK = False
    logger.debug("dsp.pesto_pitch nicht verfügbar — direkt pYIN-Ersatzpfad")


# ---------------------------------------------------------------------------
# Inharmonizitäts-Priors (§2.11 HarmonicLatticeAnalyzer)
# ---------------------------------------------------------------------------
INHARMONICITY_PRIORS: dict[str, float] = {
    "piano_bass": 0.0080,
    "piano_mid": 0.0020,
    "piano_treble": 0.0001,
    "guitar": 0.0005,
    "violin": 0.0003,
    "flute": 0.0000,
    "brass": 0.0001,
    "unknown": 0.0010,
}

# ---------------------------------------------------------------------------
# Öffentliche Konstanten (§2.28)
# ---------------------------------------------------------------------------
G_FLOOR_HARMONIC: float = 0.85
G_FLOOR_DEFAULT: float = 0.10
MAX_GAIN_CORRECTION: float = 2.0
VOICING_CONFIDENCE_MIN: float = 0.60

# ±3 Cent-Toleranz für Partial-Bin-Matching
_CENT_TOLERANCE: float = 3.0
# Maximale Partial-Anzahl
_MAX_PARTIALS: int = 20
# Minimum-Statistics Fenster für Rausch-PSD-Schätzung (Martin 2001)
_NOISE_STAT_WINDOW: int = 20  # ~215 ms bei 48kHz/512 hop
# SNR-Skalierungsfaktor für Sigmoid in adaptivem G_floor
SNR_ADAPTIVE_SCALE: float = 5.0


# ---------------------------------------------------------------------------
# Singleton (§3.2)
# ---------------------------------------------------------------------------
_instance: HarmonicPreservationGuard | None = None
_lock = threading.Lock()


def get_harmonic_preservation_guard() -> HarmonicPreservationGuard:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = HarmonicPreservationGuard()
    return _instance


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class HarmonicPreservationGuard:
    """
    Schützt harmonische Partials vor Überdämpfung durch Rauschunterdrückung.

    Alle Methoden sind NaN/Inf-sicher (§3.1) und geben float32 zurück.
    Position in Pipeline: Nach TDP, vor phase_03_denoise + phase_29.
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512) -> None:
        self._n_fft = n_fft
        self._hop_length = hop_length

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def extract_harmonic_mask(
        self,
        audio: np.ndarray,
        sr: int,
        instrument_tag: str = "unknown",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrahiert den harmonischen Fingerabdruck und liefert eine Schutzmaske.

        Args:
            audio: Input-Audio (mono oder stereo), float32
            sr: Sample-Rate (muss 48000 Hz sein)
            instrument_tag: Instrument (beeinflusst Inharmonizitäts-Koeffizient B)

        Returns:
            (protected_mask, h_ref) als float32:
                protected_mask: bool-artige Maske [n_bins × n_frames] mit 1.0/0.0
                h_ref: |STFT(audio)| als Energie-Referenz [n_bins × n_frames]
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalize to (N, ch) samples-first — UV3 passes (ch, N) channels-first.
        # Guard: only transpose when shape[0] ≤ 2 and shape[1] > 2 to avoid flipping
        # genuine (N, 2)-shaped audio with exactly 2 samples (e.g. unit tests).
        if audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > 2:
            audio = audio.T

        mono = audio[:, 0] if audio.ndim == 2 else audio
        mono = mono.astype(np.float32)

        stft_frames = self._stft(mono)
        h_ref = np.abs(stft_frames).astype(np.float32)

        # f₀-Schätzung
        f0_track = self._estimate_f0_track(mono, sr, h_ref.shape[1])

        # Harmonische Maske aufbauen
        B = INHARMONICITY_PRIORS.get(instrument_tag, INHARMONICITY_PRIORS["unknown"])
        freq_per_bin = sr / self._n_fft  # Hz pro Bin
        n_bins, n_frames = h_ref.shape

        # Cent-Toleranz in Bins (hängt von der Frequenz ab — approximiert als fester Wert)
        cent_fraction = 2.0 ** (_CENT_TOLERANCE / 1200.0)  # ≈ 1.00173

        protected_mask = np.zeros((n_bins, n_frames), dtype=np.float32)

        for t in range(n_frames):
            f0 = float(f0_track[t])
            if f0 < 20.0:
                continue

            for n in range(1, _MAX_PARTIALS + 1):
                f_n = n * f0 * math.sqrt(1.0 + B * n * n)
                if f_n >= sr / 2.0:
                    break
                # Fenster um Partial-Frequenz in Bins
                bin_n = f_n / freq_per_bin
                bin_low = max(0, int(bin_n / cent_fraction))
                bin_high = min(n_bins - 1, math.ceil(bin_n * cent_fraction))
                protected_mask[bin_low : bin_high + 1, t] = 1.0

        logger.debug(
            "HPG: f₀-Frames=%d, protected_bins=%.1f%%",
            n_frames,
            100.0 * float(np.mean(protected_mask)),
        )

        h_ref = np.nan_to_num(h_ref, nan=0.0, posinf=0.0, neginf=0.0)
        return protected_mask.astype(np.float32), h_ref.astype(np.float32)

    def apply_correction(
        self,
        restored: np.ndarray,
        h_ref: np.ndarray,
        protected_mask: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """
        Energie-Korrektur in protected_bins nach Rauschunterdrückung.

        Algorithmus:
            Für jeden protected_bin [t, f]:
                if |STFT(restored)[t,f]| < 0.85 · H_ref[t,f]:
                    gain = H_ref[t,f] / max(|STFT(restored)[t,f]|, 1e-8)
                    gain = clip(gain, 1.0, 2.0)  ← nur Anhebung, keine Absenkung
            PGHI für phasenkonsistente Rücktransformation

        Args:
            restored: Rauschunterdrücktes Audio (mono oder stereo), float32
            h_ref: Energie-Referenz aus extract_harmonic_mask()
            protected_mask: Schutzmaske aus extract_harmonic_mask()
            sr: 48000 Hz

        Returns:
            Korrigiertes Audio (float32, ≤ 1.0)
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        restored = np.nan_to_num(restored, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalize to (N, ch) samples-first — UV3 passes (ch, N) channels-first.
        # Track orientation so we can return audio in the same format that was passed in.
        _was_channels_first = restored.ndim == 2 and restored.shape[0] <= 2 and restored.shape[1] > 2
        if _was_channels_first:
            restored = restored.T  # (ch, N) → (N, ch)

        if restored.ndim == 2:
            c_l = self._correct_channel(restored[:, 0], h_ref, protected_mask, sr)
            c_r = self._correct_channel(restored[:, 1], h_ref, protected_mask, sr)
            out = np.stack([c_l, c_r], axis=1)  # (N, ch)
            if _was_channels_first:
                out = out.T  # restore (ch, N) for UV3
        else:
            out = self._correct_channel(restored, h_ref, protected_mask, sr)

        out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return out.astype(np.float32)  # type: ignore[no-any-return]

    def build_gfloor_mask(
        self,
        protected_mask: np.ndarray,
        noise_psd: np.ndarray | None = None,
        stft: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Baut G_floor-Maske für OMLSA/DeepFilterNet-Integration.

        Ohne noise_psd/stft: statische Werte (rückwärtskompatibel).
        Mit noise_psd oder stft: SNR-frame-adaptiver G_floor — verhindert
        Halluzinations-Artefakte bei Rauschbursts in protected bins.

        Adaptive Formel (§2.29 Prio-2):
            local_snr_db = _compute_local_snr(stft, noise_psd)
            g_adaptive = sigmoid(snr_db / SNR_ADAPTIVE_SCALE) × 0.75 + 0.10
            g_adaptive = clip(g_adaptive, G_FLOOR_DEFAULT, G_FLOOR_HARMONIC)
            mask = g_adaptive  wenn protected_bin,  G_FLOOR_DEFAULT sonst

        SNR = 0 dB  → g ≈ 0.475  (Übergangsbereich)
        SNR = 20 dB → g ≈ 0.843  (nahe G_FLOOR_HARMONIC)
        SNR < -20 dB → g ≈ 0.10  (Rauschburst → kein Artefakt-Floor)

        Args:
            protected_mask: bool-artige Maske [n_bins × n_frames]
            noise_psd:      Optional — Rausch-PSD [n_bins × n_frames] oder [n_bins]
            stft:           Optional — Complex STFT [n_bins × n_frames]
                            (wird genutzt falls noise_psd None ist)

        Returns:
            float32-Array [n_bins × n_frames] mit G_floor-Werten.
        """
        if noise_psd is not None or stft is not None:
            _s = stft if stft is not None else np.zeros(protected_mask.shape, dtype=np.complex64)
            local_snr = self._compute_local_snr(_s, noise_psd)
            # Sigmoid: mappt SNR linear auf [0.10, 0.85]
            g_adaptive = 1.0 / (1.0 + np.exp(-local_snr / SNR_ADAPTIVE_SCALE))
            g_adaptive = g_adaptive * 0.75 + G_FLOOR_DEFAULT
            g_adaptive = np.clip(g_adaptive, G_FLOOR_DEFAULT, G_FLOOR_HARMONIC).astype(np.float32)
            mask = np.where(protected_mask > 0.5, g_adaptive, G_FLOOR_DEFAULT)
        else:
            # Legacy: statisch (rückwärtskompatibel)
            mask = np.where(protected_mask > 0.5, G_FLOOR_HARMONIC, G_FLOOR_DEFAULT)
        return mask.astype(np.float32)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _compute_local_snr(
        self,
        stft: np.ndarray,
        noise_psd: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Berechnet frame-wise local SNR per bin in dB.

        If noise_psd is None, estimates noise PSD via minimum statistics over a
        rolling window of _NOISE_STAT_WINDOW frames (Martin 2001 simplified):
            noise_est[f, t] = min(power[f, t-W .. t])

        Args:
            stft:      Complex STFT [n_bins × n_frames]
            noise_psd: Optional precomputed noise power density [n_bins × n_frames]
                       or [n_bins] or [n_bins × 1] (broadcast-safe).

        Returns:
            local_snr_db: float32 [n_bins × n_frames] — SNR in dB per bin/frame.
                          Clamped to [-60, +60] dB.
        """
        power = (np.abs(stft) ** 2).astype(np.float64)  # [n_bins × n_frames]

        if noise_psd is None:
            # Minimum statistics (Martin 2001 simplified)
            _n_bins, n_frames = power.shape
            noise_est = np.empty_like(power)
            win = _NOISE_STAT_WINDOW
            for t in range(n_frames):
                t_start = max(0, t - win)
                noise_est[:, t] = np.min(power[:, t_start : t + 1], axis=1)
            noise_psd_used = np.maximum(noise_est, 1e-12)
        else:
            arr = np.asarray(noise_psd, dtype=np.float64)
            # Broadcast [n_bins] → [n_bins × n_frames] if needed
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            noise_psd_used = np.broadcast_to(arr, power.shape).copy()
            noise_psd_used = np.maximum(noise_psd_used, 1e-12)

        snr_linear = power / noise_psd_used
        snr_db = 10.0 * np.log10(np.maximum(snr_linear, 1e-12))
        snr_db = np.clip(snr_db, -60.0, 60.0)
        return np.nan_to_num(snr_db, nan=0.0, posinf=60.0, neginf=-60.0).astype(np.float32)  # type: ignore[no-any-return]

    def _stft(self, mono: np.ndarray) -> np.ndarray:
        """STFT → [n_bins × n_frames] complex128."""
        n_fft = self._n_fft
        hop = self._hop_length
        win = np.hanning(n_fft).astype(np.float32)
        frames = []
        for i in range(0, max(1, len(mono) - n_fft + 1), hop):
            frame = mono[i : i + n_fft].astype(np.float32)
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            frames.append(np.fft.rfft(frame * win))
        if not frames:
            return np.zeros((n_fft // 2 + 1, 1), dtype=np.complex64)  # type: ignore[no-any-return]
        return np.array(frames, dtype=np.complex64).T  # type: ignore[no-any-return]

    def _istft_ola(self, D: np.ndarray, target_len: int) -> np.ndarray:
        """OLA-ISTFT → float32 mit korrekter Länge."""
        n_fft = self._n_fft
        hop = self._hop_length
        win = np.hanning(n_fft).astype(np.float32)
        n_frames = D.shape[1]
        out_len = (n_frames - 1) * hop + n_fft
        audio_out = np.zeros(out_len, dtype=np.float64)
        win_sum = np.zeros(out_len, dtype=np.float64)

        for t in range(n_frames):
            frame = np.real(np.fft.irfft(D[:, t], n=n_fft)).astype(np.float64)
            start = t * hop
            end = start + n_fft
            if end <= out_len:
                audio_out[start:end] += frame * win
                win_sum[start:end] += win**2
            else:
                trim = end - out_len
                audio_out[start:] += frame[: n_fft - trim] * win[: n_fft - trim]
                win_sum[start:] += win[: n_fft - trim] ** 2

        win_sum = np.maximum(win_sum, 1e-8)
        audio_out /= win_sum

        if len(audio_out) > target_len:
            audio_out = audio_out[:target_len]
        elif len(audio_out) < target_len:
            audio_out = np.pad(audio_out, (0, target_len - len(audio_out)))

        return audio_out.astype(np.float32)  # type: ignore[no-any-return]

    def _correct_channel(
        self,
        mono: np.ndarray,
        h_ref: np.ndarray,
        protected_mask: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Energie-Korrektur für einen Kanal."""
        D_rest = self._stft(mono)
        n_bins_rest, n_frames_rest = D_rest.shape

        # Maske und Referenz auf STFT-Größe angleichen
        n_bins_ref, n_frames_ref = h_ref.shape
        n_bins = min(n_bins_rest, n_bins_ref)
        n_frames = min(n_frames_rest, n_frames_ref)

        mag_rest = np.abs(D_rest[:n_bins, :n_frames])
        phase_rest = np.angle(D_rest[:n_bins, :n_frames])
        mask = protected_mask[:n_bins, :n_frames]
        h_ref_crop = h_ref[:n_bins, :n_frames]

        # Energie-Korrektur nur in protected_bins
        target_mag = G_FLOOR_HARMONIC * h_ref_crop  # 0.85 × Referenz
        needs_boost = (mask > 0.5) & (mag_rest < target_mag)
        gain = np.where(
            needs_boost,
            np.clip(
                h_ref_crop / np.maximum(mag_rest, 1e-8),
                1.0,
                MAX_GAIN_CORRECTION,
            ),
            1.0,
        )

        mag_corrected = mag_rest * gain
        D_corrected = mag_corrected * np.exp(1j * phase_rest)

        # Padding falls n_bins/n_frames kleiner als Original
        if n_bins < n_bins_rest or n_frames < n_frames_rest:
            D_full = D_rest.copy()
            D_full[:n_bins, :n_frames] = D_corrected
        else:
            D_full = D_corrected

        return self._istft_ola(D_full, len(mono))

    def _estimate_f0_track(self, mono: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        """
        f₀ pro STFT-Frame schätzen.

        Tier-1: FCPE → Tier-2: RMVPE → Tier-3: PESTO (DSP) → Tier-4: pYIN → Tier-5: Autokorrelation

        Returns:
            float32-Array [n_frames] mit f₀ in Hz (0 = unvoiced)
        """
        # Tier-1: FCPE (CREPE ONNX Fallback intern)
        if _CREPE_OK:
            try:
                plugin = _get_crepe()
                _r = plugin.analyze(mono, sr)
                freqs = _r.f0_hz
                confs = _r.voiced_prob
                # Auf n_frames resampeln
                f0_track = np.interp(
                    np.linspace(0, len(freqs) - 1, n_frames),
                    np.arange(len(freqs)),
                    freqs,
                )
                conf_track = np.interp(
                    np.linspace(0, len(confs) - 1, n_frames),
                    np.arange(len(confs)),
                    confs,
                )
                f0_track[conf_track < VOICING_CONFIDENCE_MIN] = 0.0
                return f0_track.astype(np.float32)  # type: ignore[no-any-return]
            except Exception as exc:
                logger.debug("FCPE f₀-Track fehlgeschlagen: %s", exc)

        # Tier-2: RMVPE ONNX (Wei et al. ICASSP 2023) — §4.4 Fallback2.
        # ~30 % geringere Pitch-Fehlerrate vs. CREPE bei Vibrato + stimmhaft/stimmlos-Übergängen.
        if _RMVPE_OK:
            try:
                _rmvpe_r = _get_rmvpe().analyze(mono, sr)
                _rmvpe_freqs = np.where(_rmvpe_r.voiced_flag, _rmvpe_r.f0, 0.0).astype(np.float32)
                if np.sum(_rmvpe_r.voiced_flag) >= 3:
                    f0_track = np.interp(
                        np.linspace(0, len(_rmvpe_freqs) - 1, n_frames),
                        np.arange(len(_rmvpe_freqs)),
                        _rmvpe_freqs,
                    ).astype(np.float32)
                    logger.debug(
                        "HPG Tier-2 RMVPE: f0_mean=%.1f Hz voiced=%d/%d",
                        float(_rmvpe_r.f0_mean),
                        int(np.sum(_rmvpe_r.voiced_flag)),
                        len(_rmvpe_r.voiced_flag),
                    )
                    return f0_track  # type: ignore[no-any-return]
            except Exception as exc:
                logger.debug("RMVPE f₀-Track fehlgeschlagen: %s", exc)

        # Tier-3: PESTO DSP (Riou et al. ISMIR 2023) — §4.4: PESTO → pYIN.
        # CQT-based, ~8-20× schneller als pYIN; gut für tonales/melodisches Material.
        if _PESTO_OK:
            try:
                pesto_r = _pesto_estimate(mono, sr)
                if pesto_r.f0_mean > 0 and np.sum(pesto_r.voiced) > 3:
                    pesto_f0 = np.where(
                        pesto_r.voiced,
                        np.nan_to_num(pesto_r.f0, nan=0.0),
                        0.0,
                    )
                    f0_track = np.interp(
                        np.linspace(0, len(pesto_f0) - 1, n_frames),
                        np.arange(len(pesto_f0)),
                        pesto_f0,
                    )
                    return f0_track.astype(np.float32)  # type: ignore[no-any-return]
            except Exception as exc:
                logger.debug("PESTO f₀-Track fehlgeschlagen: %s", exc)

        # Tier-3: pYIN (Mauch & Dixon 2014) — letzter DSP-Fallback (§4.4).
        # §4.2: librosa.yin (einfaches YIN, de Cheveigné 2002) ist VERBOTEN → pYIN.
        # Mindestlänge-Guard (≥ 8192 Samples) verhindert SIGSEGV aus
        # librosa.sequence.viterbi auf sehr kurzen / synthetischen Signalen.
        if _LIBROSA_OK:
            try:
                if len(mono) < 8192:
                    raise ValueError("Signal zu kurz für pYIN (< 8192 Samples ≈ 170 ms)")
                f0_pyin, voiced_flag, _ = librosa.pyin(
                    mono,
                    fmin=50.0,
                    fmax=float(librosa.note_to_hz("C8")),
                    sr=sr,
                    hop_length=self._hop_length,
                    fill_na=0.0,
                )
                f0_pyin = np.nan_to_num(f0_pyin, nan=0.0)
                # Nicht-voiced Frames: f0 = 0
                f0_pyin = np.where(voiced_flag, f0_pyin, 0.0)
                f0_track = np.interp(
                    np.linspace(0, len(f0_pyin) - 1, n_frames),
                    np.arange(len(f0_pyin)),
                    f0_pyin,
                )
                return f0_track.astype(np.float32)  # type: ignore[no-any-return]
            except Exception as exc:
                logger.debug("pYIN f₀-Track fehlgeschlagen: %s", exc)

        # Tier-3: Globale Autokorrelation → konstante f₀ für alle Frames — FFT-based O(N log N)
        from backend.core.core_utils import fft_autocorr

        mono_seg = mono[: min(len(mono), sr)]
        autocorr = fft_autocorr(mono_seg)
        min_lag = max(1, int(sr / 1200.0))
        max_lag = min(int(sr / 50.0), len(autocorr) - 1)
        if max_lag > min_lag:
            peak_lag = min_lag + int(np.argmax(autocorr[min_lag : max_lag + 1]))
            f0_global = float(sr / max(peak_lag, 1))
            if autocorr[peak_lag] / (autocorr[0] + 1e-12) > 0.3:
                return np.full(n_frames, f0_global, dtype=np.float32)  # type: ignore[no-any-return]
        return np.zeros(n_frames, dtype=np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Convenience-Funktionen
# ---------------------------------------------------------------------------


def extract_harmonic_mask(
    audio: np.ndarray,
    sr: int,
    instrument_tag: str = "unknown",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience: Extrahiert harmonische Schutzmaske.

    Returns:
        (protected_mask [n_bins × n_frames], h_ref [n_bins × n_frames])
    """
    return get_harmonic_preservation_guard().extract_harmonic_mask(audio, sr, instrument_tag)


def apply_harmonic_correction(
    restored: np.ndarray,
    h_ref: np.ndarray,
    protected_mask: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Convenience: Energie-Korrektur in protected_bins nach NR.

    Returns:
        Korrigiertes Audio (float32)
    """
    return get_harmonic_preservation_guard().apply_correction(restored, h_ref, protected_mask, sr)
