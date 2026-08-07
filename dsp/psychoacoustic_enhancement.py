"""
psychoacoustic_enhancement.py - Psychoacoustic Bass Enhancement for AURIK 6.0

Implements "Missing Fundamental" effect - generates harmonics to create bass perception
on bass-weak systems (smartphones, laptop speakers, earbuds).

Technique: Generate 2nd/3rd harmonics of sub-bass (20-80Hz) in audible range (40-240Hz)
Result: Bass presence without actual low-frequency content

References:
- Waves MaxxBass, Renaissance Bass
- SPL Vitalizer
- Psychoacoustic principles by Fletcher-Munson
"""

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, hilbert, sosfilt

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.psychoacoustic_enhancement")
logger.setLevel(logging.INFO)


class PsychoacousticEnhancer:
    def __init__(
        self,
        bass_freq_range: tuple[float, float] = (20, 80),
        harmonic_gain_db: float = 6.0,
        mix: float = 0.5,
    ):
        """Initialisiert PsychoacousticEnhancer.

        Args:
            bass_freq_range: Frequency range for bass extraction (Hz)
            harmonic_gain_db: Gain for generated harmonics (dB)
            mix: Dry/wet mix (0.0 = dry, 1.0 = wet)
        """
        self.bass_freq_range = bass_freq_range
        self.harmonic_gain_db = harmonic_gain_db
        self.mix = np.clip(mix, 0.0, 1.0)

    def process(
        self, audio: np.ndarray, sr: int, use_deep_learning: bool = False, audit_log: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Apply psychoacoustic bass enhancement with quality gate, audit logging, and optional DL-inference.
        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate
            use_deep_learning: Use optional DL-Inferenz (torch/jit) falls verfügbar
            audit_log: Audit-Logging aktivieren
        Returns:
            Tuple of (enhanced audio, metrics dict)
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        metrics = {}
        fallback_used = False
        try:
            # Quality Gate: Input-Checks
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr < 8000:
                raise ValueError("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
            if np.isnan(audio).any():
                raise ValueError("Audio enthält NaN-Werte")
            if np.max(np.abs(audio)) > 1.5:
                logger.warning("Audio möglicherweise nicht normiert (max > 1.5)")

            if use_deep_learning and _TORCH_AVAILABLE:
                logger.info("Deep-Learning-Inferenz aktiviert für psychoakustische Verbesserung.")
                # TorchScript-Modell (Platzhalter)
                logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                fallback_used = True
                result = self._process_classic(audio, sr)
            else:
                result = self._process_classic(audio, sr)
        except Exception as e:
            logger.error("Fehler bei psychoakustischer Verbesserung: %s", e)
            fallback_used = True
            result = audio.copy()
            metrics["error"] = str(e)

        # Calculate enhancement metrics
        bass_energy_before = self._measure_bass_energy(audio, sr)
        bass_energy_after = self._measure_bass_energy(result, sr)
        enhancement_db = 20 * np.log10((bass_energy_after + 1e-10) / (bass_energy_before + 1e-10))

        metrics.update(
            {
                "bass_energy_before": float(bass_energy_before),  # type: ignore[dict-item]
                "bass_energy_after": float(bass_energy_after),  # type: ignore[dict-item]
                "enhancement_db": float(enhancement_db),  # type: ignore[dict-item]
                "harmonic_gain_db": self.harmonic_gain_db,  # type: ignore[dict-item]
                "mix": self.mix,
                "fallback_used": fallback_used,  # type: ignore[dict-item]
            }
        )

        if audit_log:
            self._audit_log(metrics)

        return result, metrics

    def _audit_log(self, metrics):
        logger.info("[AuditLog][PsychoacousticEnhancer] metrics=%s", metrics)

    def _process_classic(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Klassische Verarbeitung (ohne DL-Inferenz)."""
        is_stereo = audio.ndim == 2
        if is_stereo:
            left = self._enhance_channel(audio[0], sr)
            right = self._enhance_channel(audio[1], sr)
            result = np.stack([left, right], axis=0)
        else:
            result = self._enhance_channel(audio, sr)
        return result  # type: ignore[no-any-return]

    def _enhance_channel(self, channel: np.ndarray, sr: int) -> np.ndarray:
        """
        Enhance single audio channel with psychoacoustic bass.

        Process:
        1. Extract bass frequencies (20-80 Hz)
        2. Generate 2nd and 3rd harmonics
        3. Mix back with original signal
        """
        # Step 1: Extract bass frequencies
        bass_signal = self._extract_bass(channel, sr)

        # Step 2: Generate harmonics using envelope detection + synthesis
        harmonics = self._generate_harmonics(bass_signal, sr)

        # Step 3: Mix with original
        gain_linear = 10 ** (self.harmonic_gain_db / 20.0)
        harmonics_scaled = harmonics * gain_linear

        # Dry/wet mix
        result = channel + (harmonics_scaled * self.mix)

        # NaN/Inf-Guard
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # Final clip
        result = np.clip(result, -1.0, 1.0)

        return result  # type: ignore[no-any-return]

    def _extract_bass(self, channel: np.ndarray, sr: int) -> np.ndarray:
        """Extrahiert bass frequencies using bandpass filter."""
        nyquist = sr / 2
        low = self.bass_freq_range[0] / nyquist
        high = self.bass_freq_range[1] / nyquist

        # Ensure valid frequency range
        low = max(low, 0.001)
        high = min(high, 0.999)

        # Butterworth bandpass filter (4th order)
        sos = butter(4, [low, high], btype="band", output="sos")
        bass_signal = sosfilt(sos, channel)

        # NaN/Inf-Guard
        bass_signal = np.nan_to_num(bass_signal, nan=0.0, posinf=0.0, neginf=0.0)

        return bass_signal  # type: ignore[no-any-return]

    def _generate_harmonics(self, bass_signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Generiert 2nd and 3rd harmonics using envelope detection.

        Technique: Extract envelope, then synthesize harmonics at 2x and 3x frequency.
        """
        # Envelope detection using Hilbert transform
        analytic_signal = np.asarray(hilbert(np.asarray(bass_signal, dtype=np.float64)), dtype=np.complex128)
        envelope = np.sqrt(np.square(analytic_signal.real) + np.square(analytic_signal.imag))

        # Smooth envelope (low-pass filter at 200 Hz)
        nyquist = sr / 2
        cutoff = 200.0 / nyquist
        sos_smooth = butter(2, cutoff, btype="low", output="sos")
        envelope_smooth = sosfilt(sos_smooth, envelope)

        # Generate 2nd harmonic (octave up)
        # Use instantaneous phase for phase-coherent synthesis
        phase = np.unwrap(np.arctan2(analytic_signal.imag, analytic_signal.real))
        phase_2nd = phase * 2  # Double frequency
        harmonic_2nd = envelope_smooth * np.cos(phase_2nd)

        # Generate 3rd harmonic
        phase_3rd = phase * 3  # Triple frequency
        harmonic_3rd = envelope_smooth * np.cos(phase_3rd) * 0.5  # Lower gain for 3rd

        # Combine harmonics
        harmonics = harmonic_2nd + harmonic_3rd

        # Bandpass filter harmonics to audible range (40-240 Hz)
        low = 40.0 / nyquist
        high = 240.0 / nyquist
        sos_bp = butter(4, [low, high], btype="band", output="sos")
        harmonics_filtered = sosfilt(sos_bp, harmonics)

        # NaN/Inf-Guard
        harmonics_filtered = np.nan_to_num(harmonics_filtered, nan=0.0, posinf=0.0, neginf=0.0)

        return harmonics_filtered  # type: ignore[no-any-return]

    def _measure_bass_energy(self, audio: np.ndarray, sr: int) -> float:
        """Misst bass energy in 40-240 Hz range."""
        audio_mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio

        # Extract 40-240 Hz range
        nyquist = sr / 2
        low = 40.0 / nyquist
        high = 240.0 / nyquist

        sos = butter(4, [low, high], btype="band", output="sos")
        bass_band = sosfilt(sos, audio_mono)

        # RMS energy
        energy = np.sqrt(np.mean(bass_band**2))
        return energy  # type: ignore[no-any-return]


def create_psychoacoustic_enhancer(
    bass_freq_range: tuple[float, float] = (20, 80), harmonic_gain_db: float = 6.0, mix: float = 0.5
) -> PsychoacousticEnhancer:
    """Factory-Funktion zum Erstellen einer PsychoacousticEnhancer-Instanz."""
    return PsychoacousticEnhancer(bass_freq_range=bass_freq_range, harmonic_gain_db=harmonic_gain_db, mix=mix)


# Example usage (normkonform)
if __name__ == "__main__":
    import sys

    import soundfile as sf

    from backend.file_import import load_audio_file

    logging.basicConfig(level=logging.INFO)
    try:
        _res = load_audio_file("test_audio.wav")
        audio, sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])  # type: ignore[index]
        enhancer = create_psychoacoustic_enhancer(bass_freq_range=(20, 80), harmonic_gain_db=6.0, mix=0.5)
        enhanced, metrics = enhancer.process(audio, sr, use_deep_learning=True, audit_log=True)
        logger.info("Psychoacoustic Enhancement angewendet:")
        logger.info("  Bass Energy Gain: %.1f dB", metrics["enhancement_db"])
        logger.info("  Harmonic Gain: %.1f dB", metrics["harmonic_gain_db"])
        if metrics.get("fallback_used"):
            logger.info("[INFO] Ersatzpfad auf klassische Methode oder Originalaudio.")
        sf.write("enhanced_psychoacoustic.wav", enhanced, sr)
    except Exception as e:
        logger.error("Fehler im Beispielskript: %s", e)
        sys.exit(1)
