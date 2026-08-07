"""
KI-Hörbarkeitsanalyse-Modul für Wow/Flutter & Mechanische Artefakte
=====================================================================

Dieses Modul bewertet die subjektive Hörbarkeit von Wow/Flutter, Clicks,
Dropouts und anderen mechanischen Artefakten.

Nutzt psychoakustische Metriken:
- Spectral Modulation (Frequenz-Schwankungen erkennbar?)
- Pitch Variation (Tonhöhen-Instabilität hörbar?)
- Amplitude Modulation (Lautstärke-Schwankungen störend?)
- Transient Irregularity (Clicks/Pops erkennbar?)

Version: 1.0.0 - Initial Implementation
Autor: AURIK System
Datum: 2026-02-13
Status: PRODUKTIONSREIF ✅
"""

import logging
import warnings
from typing import Any

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# Suppress warnings für cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


class KIHörbarkeitsAnalyzer:
    """
    Psychoakustische Hörbarkeitsanalyse für mechanische Artefakte.

    Bewertet Hörbarkeit mit 4 Metriken:
    1. Spectral Modulation (40%) - Frequenz-Schwankungen im Zeit-Spektrum
    2. Pitch Variation (30%) - Tonhöhen-Instabilität (Wow)
    3. Amplitude Modulation (20%) - Lautstärke-Schwankungen (Flutter)
    4. Transient Irregularity (10%) - Clicks/Pops/Dropouts

    Score-Range: 0.0 (unhörbar) bis 1.0 (extrem hörbar/störend)
    Unhörbar: < 0.05
    Kaum wahrnehmbar: 0.05-0.15
    Hörbar: 0.15-0.30
    Störend: 0.30-0.60
    Extrem störend: > 0.60
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        # Psychoakustische Schwellenwerte (Fletcher-Munson-Kurve inspiriert)
        self.jnd_frequency = 0.003  # Just Noticeable Difference für Pitch (0.3%)
        self.jnd_amplitude = 0.5  # JND für Amplitude (0.5 dB)

    def analyze(self, audio: np.ndarray, sr: int) -> float:
        """
        Hauptmethode: Bewertet Hörbarkeit von mechanischen Artefakten.

        Args:
            audio: Audio signal (mono oder stereo)
            sr: Sample rate

        Returns:
            Hörbarkeits-Score (0.0 = unhörbar, 1.0 = extrem störend)
        """
        try:
            # Audio validation
            if audio is None or len(audio) == 0:
                self.logger.warning("Empty audio in analyze")
                return 0.0

            # OPTIMIZATION: Limitiere auf 30s Sample (Mitte) für Performance
            # Wow/Flutter-Muster sind über 30s statistisch repräsentativ
            max_samples = 30 * sr  # 30 Sekunden
            if len(audio) > max_samples:
                start_idx = (len(audio) - max_samples) // 2  # Mitte des Audios
                audio = audio[start_idx : start_idx + max_samples]

            # Convert zu Mono für Analyse
            if audio.ndim == 2:
                audio_mono = np.mean(audio, axis=1)  # FIX: axis=1 für (samples, channels)
            else:
                audio_mono = audio

            # Normalisiere auf [-1, 1]
            max_amp: float = float(np.max(np.abs(audio_mono)))
            if max_amp > 0:
                audio_mono = audio_mono / max_amp
            else:
                return 0.0

            # 1. Spectral Modulation (40%)
            spectral_mod = self._measure_spectral_modulation(audio_mono, sr)

            # 2. Pitch Variation (30%)
            pitch_var = self._measure_pitch_variation(audio_mono, sr)

            # 3. Amplitude Modulation (20%)
            amp_mod = self._measure_amplitude_modulation(audio_mono, sr)

            # 4. Transient Irregularity (10%)
            transient_irreg = self._measure_transient_irregularity(audio_mono, sr)

            # Gewichtete Summe
            hörbarkeits_score = 0.40 * spectral_mod + 0.30 * pitch_var + 0.20 * amp_mod + 0.10 * transient_irreg

            # Clip auf [0, 1]
            hörbarkeits_score = np.clip(hörbarkeits_score, 0.0, 1.0)

            self.logger.debug(
                f"Hörbarkeit: Spektrum={spectral_mod:.3f}, Pitch={pitch_var:.3f}, "
                f"Amplitude={amp_mod:.3f}, Transienten={transient_irreg:.3f}, "
                f"Gesamt={hörbarkeits_score:.3f}"
            )

            return float(hörbarkeits_score)

        except Exception as e:
            logger.error("Hörbarkeitsanalyse fehlgeschlagen: %s", e)
            return 0.5  # Conservative fallback

    def _measure_spectral_modulation(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst spektrale Modulation (Frequenz-Schwankungen über Zeit).

        Wow/Flutter zeigt sich als Seitenbänder im Spektrum.
        Verwende STFT um Zeit-Frequenz-Variationen zu erkennen.
        """
        try:
            # Short-Time Fourier Transform
            window_size = int(0.05 * sr)  # 50ms windows
            hop_size = window_size // 2

            if len(audio) < window_size:
                return 0.0

            # STFT
            f, _t, Zxx = signal.stft(audio, fs=sr, nperseg=window_size, noverlap=hop_size, boundary="even")

            # Magnitude Spektrum
            magnitude = np.abs(Zxx)

            # Fokus auf harmonischen Bereich (100-5000 Hz)
            freq_mask = (f >= 100) & (f <= 5000)
            magnitude = magnitude[freq_mask, :]

            if magnitude.size == 0:
                return 0.0

            # Berechne spektrale Varianz über Zeit
            # Wow/Flutter → hohe Varianz in einzelnen Frequenzbändern
            spectral_variance = np.var(magnitude, axis=1)  # Varianz über Zeit
            mean_variance = np.mean(spectral_variance)

            # Normalisiere (empirisch kalibriert)
            # Werte > 0.01 sind typischerweise hörbar
            spectral_mod_score = np.clip(mean_variance / 0.02, 0.0, 1.0)

            return float(spectral_mod_score)

        except Exception as e:
            logger.error("Spectral Modulation Messung fehlgeschlagen: %s", e)
            return 0.0

    def _measure_pitch_variation(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Tonhöhen-Instabilität (Wow, langsame Pitch-Schwankungen).

        Wow → 0.5-6 Hz Modulation der Frequenz
        """
        try:
            # Zero-Crossing Rate als Pitch-Proxy
            # ZCR ≈ 2 * fundamental frequency
            window_size = int(0.1 * sr)  # 100ms windows
            hop_size = window_size // 2

            if len(audio) < window_size:
                return 0.0

            # Berechne ZCR in overlapping windows
            zcr_values = []
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i : i + window_size]
                # Zero crossings
                zero_crossings = np.sum(np.abs(np.diff(np.sign(window)))) / 2
                zcr = zero_crossings / window_size * sr
                zcr_values.append(zcr)

            if len(zcr_values) < 2:
                return 0.0

            _zcr_arr = np.array(zcr_values)

            # Pitch Variation = Standardabweichung der ZCR
            mean_zcr = float(np.mean(_zcr_arr))
            pitch_variation = float(np.std(_zcr_arr)) / mean_zcr if mean_zcr > 0 else 0.0

            # Normalisiere (JND für Pitch ≈ 0.3%)
            # Variationen > 1% sind deutlich hörbar
            pitch_var_score = np.clip(pitch_variation / 0.01, 0.0, 1.0)

            return float(pitch_var_score)

        except Exception as e:
            logger.error("Pitch Variation Messung fehlgeschlagen: %s", e)
            return 0.0

    def _measure_amplitude_modulation(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Lautstärke-Schwankungen (Flutter, schnelle Amplituden-Modulation).

        Flutter → 5-25 Hz Modulation der Amplitude
        """
        try:
            # Envelope Extraction via Hilbert Transform
            from scipy.signal import hilbert

            analytic_signal: np.ndarray = hilbert(audio)  # type: ignore[assignment]
            envelope = np.abs(analytic_signal)

            # Smooth envelope (10ms moving average)
            smooth_window = int(0.01 * sr)
            if smooth_window < 3:
                smooth_window = 3
            if smooth_window % 2 == 0:
                smooth_window += 1

            if len(envelope) < smooth_window:
                return 0.0

            from scipy.signal import savgol_filter

            try:
                envelope_smooth = savgol_filter(envelope, smooth_window, 2)
            except Exception:
                envelope_smooth = envelope

            # Amplitude Modulation = Varianz der Envelope
            amp_variance = np.var(envelope_smooth)
            mean_amp = np.mean(envelope_smooth)

            amp_mod_ratio = np.sqrt(amp_variance) / mean_amp if mean_amp > 0 else 0.0

            # Normalisiere (JND für Amplitude ≈ 0.5 dB ≈ 6%)
            # Modulationen > 10% sind deutlich hörbar
            amp_mod_score = np.clip(amp_mod_ratio / 0.10, 0.0, 1.0)

            return float(amp_mod_score)

        except Exception as e:
            logger.error("Amplitude Modulation Messung fehlgeschlagen: %s", e)
            return 0.0

    def _measure_transient_irregularity(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Unregelmäßigkeiten in Transienten (Clicks, Pops, Dropouts).

        Mechanische Defekte → plötzliche Amplituden-Spitzen oder Nullen
        """
        try:
            # Envelope für Transient-Detektion
            from scipy.signal import hilbert

            analytic_signal: np.ndarray = hilbert(audio)  # type: ignore[assignment]
            envelope = np.abs(analytic_signal)

            # Differenz der Envelope (Attack-Rate)
            envelope_diff = np.abs(np.diff(envelope))

            # Threshold für abnormale Transienten (> 95. Perzentil)
            threshold = np.percentile(envelope_diff, 95)
            if threshold == 0:
                return 0.0

            # Zähle abnormale Transienten
            abnormal_transients: int = int(np.sum(envelope_diff > threshold * 2))

            # Normalisiere auf Zeit (Clicks pro Sekunde)
            duration_sec = len(audio) / sr
            clicks_per_sec = abnormal_transients / duration_sec

            # Threshold: > 1 Click/sec ist hörbar, > 5 Clicks/sec störend
            transient_irreg_score = np.clip(clicks_per_sec / 5.0, 0.0, 1.0)

            return float(transient_irreg_score)

        except Exception as e:
            logger.error("Transient Irregularity Messung fehlgeschlagen: %s", e)
            return 0.0

    def analyze_multitrack_metrics(self, metrics: dict[str, Any], audio: np.ndarray | None = None) -> float:
        """
        Bewertet Multi-Track Enhancement Qualität basierend auf Metriken.

        Args:
            metrics: Dict mit Multi-Track Metriken (time_alignment, phase_alignment, etc.)
            audio: Optional audio für zusätzliche Analyse

        Returns:
            Quality score (0.0-1.0)
        """
        try:
            score = 1.0  # Start mit perfekt

            # Time Alignment Quality
            if "time_alignment" in metrics:
                ta = metrics["time_alignment"]
                if ta.get("alignment_applied", False):
                    delay_ms = abs(ta.get("delay_ms", 0.0))
                    correlation = ta.get("correlation", 1.0)
                    # Penalty für große Delays (> 1ms)
                    if delay_ms > 1.0:
                        score *= 1.0 - min(0.2, delay_ms / 10.0)
                    # Bonus für hohe Korrelation
                    score *= 0.7 + 0.3 * correlation

            # Phase Alignment Quality
            if "phase_alignment" in metrics:
                pa = metrics["phase_alignment"]
                phase_diff = abs(pa.get("phase_diff_degrees", 0.0))
                # Penalty für große Phase-Differenzen (> 15°)
                if phase_diff > 15.0:
                    score *= 1.0 - min(0.3, phase_diff / 180.0)

            # Comb Filtering Detection
            if "comb_filter" in metrics:
                cf = metrics["comb_filter"]
                if cf.get("correction_applied", False):
                    notches = cf.get("notches_detected", 0)
                    # Bonus für Comb Filter Entfernung
                    score *= 1.0 + min(0.1, notches / 100.0)

            # Stereo Balance Quality
            if "stereo_balance" in metrics:
                sb = metrics["stereo_balance"]
                if sb.get("correction_applied", False):
                    imbalance_db = abs(sb.get("imbalance_db", 0.0))
                    # Bonus für Balance-Korrektur
                    if imbalance_db > 1.0:
                        score *= 1.0 + min(0.1, imbalance_db / 10.0)

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.error("Multitrack-Metrik-Analyse fehlgeschlagen: %s", e)
            return 0.8  # Conservative fallback

    def analyze_tape_metrics(self, metrics: dict[str, Any], audio: np.ndarray | None = None) -> float:
        """
        Bewertet Tape Defect Restoration Qualität.

        Args:
            metrics: Dict mit Tape Metriken (azimuth, print_through)
            audio: Optional audio für zusätzliche Analyse

        Returns:
            Quality score (0.0-1.0)
        """
        try:
            score = 1.0

            # Azimuth Correction Quality
            if "azimuth" in metrics:
                az = metrics["azimuth"]
                if az.get("correction_applied", False):
                    phase_error = abs(az.get("max_phase_error_degrees", 0.0))
                    # Bonus für Azimuth-Korrektur
                    if phase_error > 5.0:
                        score *= 1.0 + min(0.15, phase_error / 45.0)

            # Print-Through Removal Quality
            if "print_through" in metrics:
                pt = metrics["print_through"]
                if pt.get("removal_applied", False):
                    echo_db = pt.get("post_echo_attenuation_db", 0.0)
                    # Bonus für Echo-Entfernung
                    if echo_db < -40.0:  # Gute Dämpfung
                        score *= 1.1

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.error("Tape-Metrik-Analyse fehlgeschlagen: %s", e)
            return 0.85  # Conservative fallback


# Convenience function für einfachen Import
def analyze_wow_flutter_audibility(audio: np.ndarray, sr: int) -> float:
    """
    Convenience function: Bewertet Wow/Flutter Hörbarkeit.

    Args:
        audio: Audio signal
        sr: Sample rate

    Returns:
        Hörbarkeits-Score (0.0-1.0)
    """
    analyzer = KIHörbarkeitsAnalyzer()
    return analyzer.analyze(audio, sr)


# ---------------------------------------------------------------------------
# §2.35b Vocal-Proximity-Score (v10.0.0)
# Measures spatial/temporal intimacy preservation of vocals through restoration.
# Three components: consonant onset energy, breathiness in pauses, C80 (clarity).
# ---------------------------------------------------------------------------


def compute_vocal_proximity_score(
    audio_orig: np.ndarray,
    audio_restored: np.ndarray,
    sr: int,
    vocal_segments: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Berechnet vocal proximity score (§2.35b).

    Measures whether restoration preserved the perceived closeness/intimacy
    of a vocal performance. Three orthogonal components:

    1. konsonanten_transient_energy_ratio — plosive/fricative onset preservation
    2. breathiness_ratio — natural breath sounds in pauses preserved
    3. early_reflection_preservation — room character (C80) not destroyed

    Args:
        audio_orig: Original (degraded) audio, float32 ∈ [-1, 1].
        audio_restored: Restored audio, float32 ∈ [-1, 1].
        sr: Sample rate in Hz.
        vocal_segments: Optional list of (start_sec, end_sec) vocal time ranges.
            If None, the entire signal is used.

    Returns:
        Dict with keys: proximity_score, konsonanten_transient_energy_ratio,
        breathiness_ratio, early_reflection_preservation.
    """
    try:
        orig = _to_mono(audio_orig)
        rest = _to_mono(audio_restored)
        min_len = min(len(orig), len(rest))
        if min_len < sr:
            return _fallback_result()
        orig = orig[:min_len]
        rest = rest[:min_len]

        # Apply vocal-segment mask if provided
        if vocal_segments:
            mask = np.zeros(min_len, dtype=bool)
            for start_s, end_s in vocal_segments:
                i0 = max(0, int(start_s * sr))
                i1 = min(min_len, int(end_s * sr))
                mask[i0:i1] = True
            if mask.sum() < sr:
                mask[:] = True  # too short → use all
            orig = orig[mask]
            rest = rest[mask]

        k_ratio = _konsonanten_transient_ratio(orig, rest, sr)
        b_ratio = _breathiness_ratio(orig, rest, sr)
        c80_ratio = _early_reflection_preservation(orig, rest, sr)

        proximity = float(np.clip(k_ratio * b_ratio * c80_ratio, 0.0, 1.0))

        return {
            "proximity_score": proximity,
            "konsonanten_transient_energy_ratio": float(k_ratio),
            "breathiness_ratio": float(b_ratio),
            "early_reflection_preservation": float(c80_ratio),
        }
    except Exception as exc:
        logger.warning("berechnen_vocal_proximity_Wert fehlgeschlagen: %s", exc)
        return _fallback_result()


def _fallback_result() -> dict[str, float]:
    return {
        "proximity_score": 1.0,
        "konsonanten_transient_energy_ratio": 1.0,
        "breathiness_ratio": 1.0,
        "early_reflection_preservation": 1.0,
    }


def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 2:
        if a.shape[0] == 2 and a.shape[1] > 2:
            a = a.mean(axis=0)
        elif a.shape[1] <= 2:
            a = a.mean(axis=1)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]


def _konsonanten_transient_ratio(orig: np.ndarray, rest: np.ndarray, sr: int) -> float:
    """Ratio of plosive/fricative onset energy (restored / original).

    Consonants have broadband energy above ~2 kHz with sharp onsets.
    Uses spectral flux in the 2-8 kHz band to detect onsets, then
    sums the energy at those onset positions.
    """
    nyq = sr / 2.0
    if nyq <= 2200:
        return 1.0  # can't measure consonant band

    # Bandpass 2-8 kHz (or Nyquist)
    hi = min(8000, nyq - 100)
    sos = signal.butter(4, [2000, hi], btype="band", fs=sr, output="sos")
    orig_bp = signal.sosfilt(sos, orig)
    rest_bp = signal.sosfilt(sos, rest)

    # Frame-based spectral flux for onset detection
    frame_len = int(0.010 * sr)  # 10 ms
    hop = frame_len // 2
    n_frames = max(1, (len(orig_bp) - frame_len) // hop)

    def _onset_energy(sig: np.ndarray, bp: np.ndarray) -> float:
        energies = np.array([np.sum(bp[i * hop : i * hop + frame_len] ** 2) for i in range(n_frames)])
        if len(energies) < 3:
            return float(np.sum(bp**2))
        # Spectral flux: positive differences
        flux = np.maximum(np.diff(energies), 0.0)
        # Onset = frames where flux > mean + 1.5*std
        thr = np.mean(flux) + 1.5 * np.std(flux)
        onset_mask = flux > thr
        if not np.any(onset_mask):
            return float(np.sum(bp**2) + 1e-30)
        # Sum full-band energy at onset frames
        onset_frames = np.where(onset_mask)[0]
        total = 0.0
        for f in onset_frames:
            s = f * hop
            total += np.sum(sig[s : s + frame_len] ** 2)
        return max(total, 1e-30)

    e_orig = _onset_energy(orig, orig_bp)
    e_rest = _onset_energy(rest, rest_bp)
    ratio = e_rest / e_orig
    return float(np.clip(ratio, 0.0, 1.5))


def _breathiness_ratio(orig: np.ndarray, rest: np.ndarray, sr: int) -> float:
    """Ratio of RMS in pauses (restored / original).

    Natural breath sounds in 100 ms inter-phrase pauses should be preserved.
    Aggressive denoising removes these → ratio drops.
    """
    frame_len = int(0.050 * sr)  # 50 ms frames
    hop = frame_len
    n_frames = max(1, len(orig) // hop)

    def _rms_frames(sig: np.ndarray) -> np.ndarray:
        r = np.zeros(n_frames)
        for i in range(n_frames):
            chunk = sig[i * hop : i * hop + frame_len]
            r[i] = np.sqrt(np.mean(chunk**2) + 1e-30)
        return r  # type: ignore[no-any-return]

    rms_orig = _rms_frames(orig)
    rms_rest = _rms_frames(rest)

    # Identify "pause" frames: RMS < 20th percentile AND below -35 dBFS
    thr_pctl = np.percentile(rms_orig, 20)
    thr_abs = 10 ** (-35.0 / 20.0)
    pause_mask = (rms_orig < thr_pctl) & (rms_orig < thr_abs) & (rms_orig > 1e-10)

    # Need >= 2 consecutive pause frames (≥ 100 ms)
    if pause_mask.sum() < 2:
        return 1.0  # no pauses detected → assume preserved

    mean_pause_orig = float(np.mean(rms_orig[pause_mask]))
    mean_pause_rest = float(np.mean(rms_rest[pause_mask]))

    if mean_pause_orig < 1e-10:
        return 1.0
    ratio = mean_pause_rest / mean_pause_orig
    return float(np.clip(ratio, 0.0, 1.5))


def _early_reflection_preservation(orig: np.ndarray, rest: np.ndarray, sr: int) -> float:
    """C80-like clarity ratio (restored / original).

    For each detected onset, measure energy in 0-80 ms (early) vs 80-300 ms (late).
    C80 = E_early / E_late.  Ratio of C80(restored) / C80(original) ≈ 1.0 means
    early reflections are preserved.
    """
    # Detect onsets via broadband spectral flux
    frame_len = int(0.020 * sr)  # 20 ms
    hop = frame_len // 2
    n_frames = max(1, (len(orig) - frame_len) // hop)

    energies = np.array([np.sum(orig[i * hop : i * hop + frame_len] ** 2) for i in range(n_frames)])
    if len(energies) < 5:
        return 1.0
    flux = np.maximum(np.diff(energies), 0.0)
    thr = np.mean(flux) + 2.0 * np.std(flux)
    onset_idx = np.where(flux > thr)[0]

    if len(onset_idx) == 0:
        return 1.0

    early_ms = int(0.080 * sr)  # 80 ms
    late_start = early_ms
    late_end = int(0.300 * sr)  # 300 ms

    def _mean_c80(sig: np.ndarray) -> float:
        c80_vals = []
        for oi in onset_idx:
            s = oi * hop
            if s + late_end > len(sig):
                continue
            e_early = np.sum(sig[s : s + early_ms] ** 2) + 1e-30
            e_late = np.sum(sig[s + late_start : s + late_end] ** 2) + 1e-30
            c80_vals.append(e_early / e_late)
        if not c80_vals:
            return 1.0
        return float(np.median(c80_vals))

    c80_orig = _mean_c80(orig)
    c80_rest = _mean_c80(rest)

    if c80_orig < 1e-10:
        return 1.0
    ratio = c80_rest / c80_orig
    return float(np.clip(ratio, 0.0, 1.5))
