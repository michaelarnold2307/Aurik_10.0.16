from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Öffentliche Konstanten (§2.11)
# ---------------------------------------------------------------------------
MAX_PARTIALS: int = 20  # Maximale Anzahl analysierter Partials (n = 1..20)
MAX_CENT_DEVIATION: float = 5.0  # Maximale erlaubte Abweichung in Cent (§2.11)

INHARMONICITY_PRIORS: dict[str, float] = {
    "piano_bass": 0.0080,
    "piano_mid": 0.0020,
    "piano_treble": 0.0001,
    "guitar": 0.0005,
    "violin": 0.0003,
    "flute": 0.0,
    "brass": 0.0001,
    "unknown": 0.0001,
}


@dataclass
class PartialAnalysis:
    partial_index: int
    target_hz: float
    observed_hz: float
    deviation_cents: float
    protected: bool = True

    # --- Alias-Properties (Spec §2.11 + Tests) ---
    @property
    def partial_n(self) -> int:
        """Alias für partial_index (Spec §2.11: n = 1..20)."""
        return self.partial_index

    @property
    def needs_correction(self) -> bool:
        """True wenn |deviation_cents| > MAX_CENT_DEVIATION (8 Cent, §v10.64).

        War 3 Cent (§2.11) — zu empfindlich. Ein akustischer Flügel hat
        5–15 Cent Inharmonizität auf hohen Partialtönen; das ist KEIN Defekt,
        sondern die timbrale Signatur des Instruments. 3 Cent zwangskorrigiert
        entfernt diese Signatur → klingt "plasticky" (Uncanny Valley).
        8 Cent korrigiert nur echte tuning/tracking-Fehler, bewahrt Charakter.
        """
        return abs(self.deviation_cents) > 8.0

    @property
    def freq_expected_hz(self) -> float:
        return self.target_hz

    @property
    def freq_detected_hz(self) -> float:
        return self.observed_hz

    @property
    def deviation_cent(self) -> float:
        return self.deviation_cents


@dataclass
class HarmonicLatticeResult:
    f0_hz: float
    inharmonicity_b: float
    partial_frequencies_hz: list[float] = field(default_factory=list)
    partial_deviations_cents: list[float] = field(default_factory=list)
    coherence_score: float = 0.0
    confidence: float = 0.0
    # Erweiterte Felder (§2.11, Tests)
    instrument_tag: str = "unknown"
    lattice_score: float = 1.0
    needs_enforcement: bool = False
    partials: list[PartialAnalysis] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Serialisierungsformat (§2.11)."""
        return {
            "f0_hz": self.f0_hz,
            "inharmonicity_b": self.inharmonicity_b,
            "lattice_score": self.lattice_score,
            "instrument_tag": self.instrument_tag,
            "needs_enforcement": self.needs_enforcement,
            "coherence_score": self.coherence_score,
            "confidence": self.confidence,
            "n_partials": len(self.partials),
        }


class HarmonicLatticeAnalyzer:
    INHARMONICITY_PRIORS = INHARMONICITY_PRIORS

    def _null_result(self, instrument_tag: str = "unknown") -> HarmonicLatticeResult:
        """Gibt Null-Ergebnis zurück (kein f₀ erkennbar, kein Enforcement nötig)."""
        return HarmonicLatticeResult(
            f0_hz=0.0,
            inharmonicity_b=float(self.INHARMONICITY_PRIORS.get(instrument_tag, 0.0001)),
            partial_frequencies_hz=[],
            partial_deviations_cents=[],
            coherence_score=1.0,
            confidence=0.0,
            instrument_tag=instrument_tag,
            lattice_score=1.0,
            needs_enforcement=False,
            partials=[],
        )

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        instrument_tag: str = "unknown",
    ) -> HarmonicLatticeResult:
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        mono = self._to_mono(audio)
        if mono.size == 0 or np.std(mono) < 1e-8:
            return self._null_result(instrument_tag)

        f0 = self._estimate_f0(mono, sr)
        b_prior = float(self.INHARMONICITY_PRIORS.get(instrument_tag, 0.0001))

        # Detect actual partial positions from the audio spectrum.
        # Uses _detect_partials() which searches ±5 % bands around ideal positions.
        detected = self._detect_partials(mono, sr, f0)

        # Refine inharmonicity coefficient B from measured data when possible
        # (Fletcher 1964: fₙ = n·f₀·√(1 + B·n²)).
        b = self._estimate_b_from_partials(detected, f0) if len(detected) >= 3 else b_prior

        # Build per-partial analysis objects merging ideal lattice with observed positions.
        detected_by_n = {p.partial_index: p for p in detected}
        freq_list: list[float] = []
        dev_list: list[float] = []
        partial_objs: list[PartialAnalysis] = []
        for n in range(1, 21):
            ideal = float(n * f0 * math.sqrt(max(1e-12, 1.0 + b * (n**2))))
            freq_list.append(ideal)
            if n in detected_by_n:
                obs_hz = detected_by_n[n].observed_hz
                dev_c = detected_by_n[n].deviation_cents
            else:
                obs_hz = ideal  # undetected → assume on-target
                dev_c = 0.0
            dev_list.append(dev_c)
            partial_objs.append(
                PartialAnalysis(
                    partial_index=n,
                    target_hz=ideal,
                    observed_hz=obs_hz,
                    deviation_cents=dev_c,
                    protected=True,
                )
            )

        # Confidence: fraction of expected partials actually detected (max 1.0).
        confidence = float(np.clip(len(detected) / 10.0, 0.0, 1.0))
        score = float(np.clip(1.0 - np.mean(np.abs(dev_list)) / 50.0, 0.0, 1.0))
        needs_enf = any(abs(d) > 3.0 for d in dev_list)
        return HarmonicLatticeResult(
            f0_hz=float(f0),
            inharmonicity_b=b,
            partial_frequencies_hz=freq_list,
            partial_deviations_cents=dev_list,
            coherence_score=score,
            confidence=confidence,
            instrument_tag=instrument_tag,
            lattice_score=score,
            needs_enforcement=needs_enf,
            partials=partial_objs,
        )

    def enforce_coherence(
        self,
        audio: np.ndarray,
        sr: int,
        lattice_result: HarmonicLatticeResult,
    ) -> np.ndarray:
        """Nudge outlier partials toward the Fletcher inharmonicity lattice.

        For each partial whose observed peak deviates from the ideal inharmonic
        frequency by more than 3 cents, applies a narrow zero-phase Bell-EQ pair
        (notch at observed position, boost at target position) in the STFT domain.
        Correction magnitude is confidence-weighted; wet factor ≤ 0.30.

        Input *audio* must be 1-D (mono channel).
        Phase 07 calls this per-channel and column_stacks the result.

        Reference: Fletcher (1964), J. Acoust. Soc. Am. 36 (1): 203–209.
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio_f32 = np.asarray(audio, dtype=np.float32)
        audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0)

        # Fast exits — §0 Primum non nocere: if nothing to correct, passthrough.
        if (
            not lattice_result.needs_enforcement
            or lattice_result.confidence < 0.25
            or lattice_result.f0_hz <= 0.0
            or audio_f32.ndim != 1
            or len(audio_f32) < 512
        ):
            return np.clip(audio_f32, -1.0, 1.0)  # type: ignore[no-any-return]

        n_samples = len(audio_f32)
        n_fft = min(2048, n_samples)
        hop = n_fft // 4
        nyquist_cap = sr / 2.0 * 0.95

        # Build frequency-domain correction gain (uniform across all frames).
        # Narrow Bell-EQ: Q ≈ 25 → bandwidth ≈ f/25 Hz.
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float64)
        correction_gain = np.ones(len(freqs), dtype=np.float64)

        # Confidence-weighted wet factor — bounded to [0.05, 0.30].
        wet = float(np.clip(0.30 * lattice_result.confidence, 0.05, 0.30))

        n_corrections = 0
        for pa in lattice_result.partials:
            if not pa.needs_correction:
                continue
            f_tgt = float(pa.target_hz)
            f_obs = float(pa.observed_hz)
            if f_tgt <= 0.0 or f_obs <= 0.0 or f_tgt > nyquist_cap or abs(f_tgt - f_obs) < 0.5:
                continue
            # Correction depth proportional to deviation magnitude, capped at 0.45.
            depth = float(np.clip(0.50 * abs(pa.deviation_cents) / 50.0, 0.05, 0.45))
            bw = f_tgt / 25.0 + 1.0  # +1 Hz floor to avoid division by zero
            # Narrow notch at observed (incorrect) position.
            correction_gain *= 1.0 - depth * np.exp(-0.5 * ((freqs - f_obs) / bw) ** 2)
            # Narrow boost at target (correct) position (slightly weaker to stay conservative).
            correction_gain *= 1.0 + depth * 0.75 * np.exp(-0.5 * ((freqs - f_tgt) / bw) ** 2)
            n_corrections += 1

        if n_corrections == 0:
            return np.clip(audio_f32, -1.0, 1.0)  # type: ignore[no-any-return]

        # Apply correction via STFT → spectral multiply → ISTFT (zero-phase, boundary='even').
        try:
            from scipy.signal import istft, stft

            _f, _t, Zxx = stft(
                audio_f32.astype(np.float64),
                fs=sr,
                nperseg=n_fft,
                noverlap=n_fft - hop,
                window="hann",
                boundary="even",
            )
            Zxx_corr = Zxx * correction_gain[:, np.newaxis]
            _, audio_corr = istft(
                Zxx_corr,
                fs=sr,
                nperseg=n_fft,
                noverlap=n_fft - hop,
                window="hann",
                boundary="even",
            )
            audio_corr = audio_corr.astype(np.float32)
            # Trim or zero-pad to original length.
            if len(audio_corr) >= n_samples:
                audio_corr = audio_corr[:n_samples]
            else:
                audio_corr = np.pad(audio_corr, (0, n_samples - len(audio_corr)))
        except Exception as e:
            logger.warning("harmonic_lattice_analyzer.py::unbekannter Ersatzpfad: %s", e)
            return np.clip(audio_f32, -1.0, 1.0)  # type: ignore[no-any-return]  # graceful passthrough on any STFT error

        # Dry/wet blend — confidence already factored into `wet`.
        out = (1.0 - wet) * audio_f32 + wet * audio_corr
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), -1.0, 1.0)  # type: ignore[no-any-return]

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            # Accept both layouts: [N, C] and [C, N].
            if arr.shape[1] <= 2 and arr.shape[0] > arr.shape[1]:
                arr = np.mean(arr, axis=1)
            elif arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
                arr = np.mean(arr, axis=0)
            else:
                arr = np.mean(arr, axis=-1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]

    @staticmethod
    def _estimate_f0(audio: np.ndarray, sr: int) -> float:
        if audio.size < 4:
            return 220.0
        n = int(2 ** np.ceil(np.log2(max(256, audio.size))))
        spec = np.fft.rfft(audio[: min(audio.size, 16384)], n=n)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
        mask = (freqs >= 50.0) & (freqs <= 1200.0)
        if not np.any(mask):
            return 220.0
        idx_local = int(np.argmax(mag[mask]))
        f0 = float(freqs[mask][idx_local])
        if not np.isfinite(f0) or f0 <= 0.0:
            return 220.0
        return f0

    def _detect_partials(
        self,
        audio: np.ndarray,
        sr: int,
        f0: float,
        min_energy: float = 1e-6,
    ) -> list[PartialAnalysis]:
        """Detektiert Partials im Spektrum des Audios für gegebenen f₀.

        Args:
            audio:      Eingabe-Audio (mono, float32)
            sr:         Sample-Rate (muss 48000 sein)
            f0:         Grundfrequenz in Hz
            min_energy: Minimale Energie für Partial-Detektion

        Returns:
            Liste von PartialAnalysis-Objekten (max. MAX_PARTIALS)
        """
        if f0 <= 0.0 or audio.size < 4:
            return []
        mono = self._to_mono(audio)
        if mono.size < 4:
            return []
        win = min(mono.size, 8192)
        spec = np.fft.rfft(mono[:win], n=win)
        mag = np.abs(spec).astype(np.float32)
        freqs = np.fft.rfftfreq(win, d=1.0 / float(sr)).astype(np.float32)
        b = float(self.INHARMONICITY_PRIORS.get("unknown", 0.0001))
        result: list[PartialAnalysis] = []
        for n in range(1, MAX_PARTIALS + 1):
            ideal = n * f0 * math.sqrt(max(1e-12, 1.0 + b * n * n))
            # Fenster ±5 % um ideale Frequenz
            low = ideal * 0.95
            high = ideal * 1.05
            band = (freqs >= low) & (freqs <= high)
            if not band.any():
                continue
            peak_mag = float(np.max(mag[band]))
            if peak_mag < min_energy:
                continue
            peak_freq = float(freqs[band][int(np.argmax(mag[band]))])
            dev_cent = 1200.0 * math.log2(peak_freq / ideal) if ideal > 0 and peak_freq > 0 else 0.0
            result.append(
                PartialAnalysis(
                    partial_index=n,
                    target_hz=float(ideal),
                    observed_hz=float(peak_freq),
                    deviation_cents=float(dev_cent),
                    protected=True,
                )
            )
        return result

    @staticmethod
    def _estimate_b_from_partials(
        partials: list[PartialAnalysis],
        f0: float,
    ) -> float:
        """Schätzt Inharmonizitäts-Koeffizient B aus gemessenen Partials.

        Nutzt das Fletcher-Modell: fₙ = n·f₀·√(1 + B·n²)
        → B = ((fₙ / (n·f₀))² − 1) / n²

        Args:
            partials: Liste von PartialAnalysis-Objekten
            f0:       Grundfrequenz in Hz

        Returns:
            Geschätztes B ∈ [0.0, 0.05], NaN-sicher
        """
        if not partials or f0 <= 0.0:
            return 0.0001
        b_vals: list[float] = []
        for p in partials:
            n = p.partial_index
            if n < 2 or p.observed_hz <= 0.0:
                continue
            denom = (n * f0) ** 2
            if denom < 1e-12:
                continue
            ratio_sq = (p.observed_hz / (n * f0)) ** 2
            b_est = (ratio_sq - 1.0) / max(1.0, float(n * n))
            if math.isfinite(b_est) and b_est >= 0.0:
                b_vals.append(b_est)
        if not b_vals:
            return 0.0001
        return float(np.clip(np.median(b_vals), 0.0, 0.05))


_instance: HarmonicLatticeAnalyzer | None = None
_lock = threading.Lock()


def get_harmonic_lattice_analyzer() -> HarmonicLatticeAnalyzer:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = HarmonicLatticeAnalyzer()
    return _instance


def get_harmonic_lattice() -> HarmonicLatticeAnalyzer:
    return get_harmonic_lattice_analyzer()


def analyze_harmonic_lattice(
    audio: np.ndarray,
    sr: int,
    instrument_tag: str = "unknown",
) -> HarmonicLatticeResult:
    return get_harmonic_lattice_analyzer().analyze(audio, sr, instrument_tag)


__all__ = [
    "INHARMONICITY_PRIORS",
    "MAX_CENT_DEVIATION",
    "MAX_PARTIALS",
    "HarmonicLatticeAnalyzer",
    "HarmonicLatticeResult",
    "PartialAnalysis",
    "analyze_harmonic_lattice",
    "get_harmonic_lattice",
    "get_harmonic_lattice_analyzer",
]
