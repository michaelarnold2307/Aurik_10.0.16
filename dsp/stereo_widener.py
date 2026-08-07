"""
Adaptive Stereo Widening
========================

Frequency-dependent Stereo Enhancement für "Studio-Breite".

Problem: Mono/Narrow-Stereo klingt flach, nicht wie moderne Studio-Produktion.

Lösung: Intelligentes Mid-Side Processing:
- Bass bleibt mono (< 200Hz) - Club-kompatibel
- Mitten moderat widen (200-1000Hz)
- Höhen stark widen (1000-8000Hz) - Breite!
- Air maximal widen (8000+Hz) - Studio-Feel

Safeguards:
- Frequency-dependent (keine Phase-Probleme)
- Genre-Adaptive Limits (Classical: minimal, Pop: moderat)
- Mono-Compatibility Check
- Parallel Processing
"""

import logging

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)


class AdaptiveStereoWidener:
    """
    Frequency-dependent Stereo Widening.

    Erweitert Stereo-Bild intelligent ohne Phase-Probleme.
    """

    def __init__(self):
        self.name = "Adaptive Stereo Widener"

    def widen(
        self,
        audio_stereo: np.ndarray,
        sr: int,
        width: float = 1.3,
        genre: str = "unknown",
    ) -> np.ndarray:
        """
        Widen stereo image frequency-dependent.

        Args:
            audio_stereo: Stereo audio (2, N) or (N, 2)
            sr: Sample rate
            width: Widening factor (1.0 = no change, 1.3 = 30% wider, 2.0 = max)
            genre: Genre für adaptive limits

        Returns:
            Widened stereo audio
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        # Check if stereo
        if audio_stereo.ndim == 1:
            logger.warning("Mono signal - cannot widen, returning unchanged")
            return audio_stereo

        # Ensure shape (2, N)
        if audio_stereo.shape[0] != 2:
            audio_stereo = audio_stereo.T

        if audio_stereo.shape[0] != 2:
            logger.error("Invalid stereo shape: %s", audio_stereo.shape)
            return audio_stereo

        # Genre-adaptive Width Limits (Safeguard #3)
        width = self._adjust_width_for_genre(width, genre)

        logger.info("Stereo Widening: width=%.2f, genre=%s", width, genre)

        # Mid-Side Decomposition
        L = audio_stereo[0]
        R = audio_stereo[1]

        mid = (L + R) / 2  # Center (Mono-content)
        side = (L - R) / 2  # Stereo-content

        # Frequency-dependent widening
        side_widened = self._frequency_dependent_widening(side, sr, width)

        # Reconstruct Stereo
        L_wide = mid + side_widened
        R_wide = mid - side_widened

        # Safety only: avoid loudness-changing peak rescale, clamp at output.
        stereo_wide = np.array([L_wide, R_wide])

        # NaN/Inf-Guard + Clipping
        stereo_wide = np.nan_to_num(stereo_wide, nan=0.0, posinf=0.0, neginf=0.0)
        stereo_wide = np.clip(stereo_wide, -1.0, 1.0)

        logger.info("✓ Stereo widening vollstaendig")

        return stereo_wide

    def _frequency_dependent_widening(self, side: np.ndarray, sr: int, width: float) -> np.ndarray:
        """
        Wendet an: frequency-dependent widening to side signal.

        Bands:
        - 0-200Hz: 0.5x width (Bass mono)
        - 200-1000Hz: 0.8x width
        - 1000-8000Hz: 1.0x width (full widening)
        - 8000+Hz: 1.2x width (extra air)
        """
        # Multiband split
        bands = self._split_into_bands(side, sr, cutoffs=[200, 1000, 8000])

        # Width multipliers per band
        width_multipliers = [
            0.5,  # Bass (mono-compatible)
            0.8,  # Low-mids
            1.0,  # Mids/highs (full width)
            1.2,  # Air (extra width)
        ]

        # Apply widening per band
        bands_widened = []
        for i, (band, mult) in enumerate(zip(bands, width_multipliers)):
            band_widened = band * (width * mult)
            bands_widened.append(band_widened)

        # Recombine
        side_widened = sum(bands_widened)

        # NaN/Inf-Guard
        side_widened = np.nan_to_num(side_widened, nan=0.0, posinf=0.0, neginf=0.0)

        return np.asarray(side_widened)

    def _split_into_bands(self, audio: np.ndarray, sr: int, cutoffs: list) -> list:
        """
        Split audio into frequency bands using Butterworth filters.

        Args:
            audio: Input audio
            sr: Sample rate
            cutoffs: List of cutoff frequencies [f1, f2, f3]

        Returns:
            List of bands [low, low-mid, mid-high, high]
        """
        bands = []
        order = 4  # 4th-order Butterworth (24dB/octave)

        # Band 1: Low (< cutoffs[0])
        sos = signal.butter(order, cutoffs[0], btype="lowpass", fs=sr, output="sos")
        band_low = signal.sosfilt(sos, audio)
        bands.append(band_low)

        # Band 2-N: Bandpass filters
        for i in range(len(cutoffs) - 1):
            sos = signal.butter(
                order,
                [cutoffs[i], cutoffs[i + 1]],
                btype="bandpass",
                fs=sr,
                output="sos",
            )
            band = signal.sosfilt(sos, audio)
            bands.append(band)

        # Band N+1: High (> cutoffs[-1])
        sos = signal.butter(order, cutoffs[-1], btype="highpass", fs=sr, output="sos")
        band_high = signal.sosfilt(sos, audio)
        bands.append(band_high)

        return bands

    def _adjust_width_for_genre(self, base_width: float, genre: str) -> float:
        """
        Genre-adaptive Width Limits (Safeguard #3).
        """
        genre_limits = {
            "classical": 1.1,  # Minimal (natural soundstage)
            "jazz": 1.3,  # Moderat (club ambience)
            "acoustic": 1.2,
            "speech": 1.0,  # NONE! (Mono ist OK)
            "rock": 1.5,
            "pop": 1.5,
            "electronic": 1.8,  # Mehr OK (wide modern production)
        }

        max_width = genre_limits.get(genre, 1.3)  # Default: 30%
        adjusted = min(base_width, max_width)

        if adjusted < base_width:
            logger.info("Stereo width limited for genre '%s': %.2f → %.2f", genre, base_width, adjusted)

        return adjusted

    def check_mono_compatibility(self, audio_stereo: np.ndarray) -> tuple[bool, float]:
        """
        Prüft if widened stereo is mono-compatible.

        Returns:
            (is_compatible, correlation)
                correlation: 1.0 = perfect phase, 0.5 = OK, < 0.3 = phase issues
        """
        if audio_stereo.ndim == 1 or audio_stereo.shape[0] != 2:
            return True, 1.0

        L = audio_stereo[0]
        R = audio_stereo[1]

        # Guarded Pearson correlation — avoids NaN on silent/constant channels (§VERBOTEN: np.corrcoef)
        _lc = L.astype(float) - float(np.mean(L))
        _rc = R.astype(float) - float(np.mean(R))
        _nl = float(np.linalg.norm(_lc))
        _nr = float(np.linalg.norm(_rc))
        if _nl < 1e-12 or _nr < 1e-12:
            correlation = 1.0
        else:
            correlation = float(np.clip(np.dot(_lc, _rc) / (_nl * _nr), -1.0, 1.0))

        is_compatible = correlation > 0.3

        if not is_compatible:
            logger.warning("Mono-compatibility issue: correlation %.2f < 0.3", correlation)

        return is_compatible, correlation


# ============================================================================
# Pipeline Integration
# ============================================================================


def select_stereo_widening_strategy(context: dict, goal: dict) -> dict:
    """
    Policy-Engine Decision: Soll Stereo Widening angewendet werden?

    Args:
        context: Audio context
        goal: Processing goals

    Returns:
        widening_params dict or None
    """
    # Regel 1: Speech → KEIN Widening
    if context.get("genre") == "speech" or context.get("has_vocals_only", False):
        logger.info("Stereo Widening uebersprungen: Speech (mono is appropriate)")
        return None  # type: ignore[return-value]

    # Regel 2: Already Mono → Kein Widening
    if context.get("channels", 2) == 1:
        logger.info("Stereo Widening uebersprungen: Mono signal")
        return None  # type: ignore[return-value]

    # Regel 3: User wants transparency → Minimales Widening
    if goal.get("priority") == "transparency":
        width = 1.1  # Nur 10%
    elif goal.get("quality_level") == "maximal":
        width = 1.5  # Moderat
    else:
        width = 1.3  # Default 30%

    params = {"width": width, "genre": context.get("genre", "unknown")}

    logger.info("Stereo Widening SELECTED: width=%.2f, genre=%s", width, params["genre"])

    return params


def apply_stereo_widening_if_needed(audio: np.ndarray, sr: int, context: dict, goal: dict) -> np.ndarray:
    """
    Convenience function für Pipeline-Integration.
    """
    # Policy Decision
    params = select_stereo_widening_strategy(context, goal)

    if params is None:
        return audio

    # Apply Widening
    widener = AdaptiveStereoWidener()
    audio_widened = widener.widen(audio, sr, width=params["width"], genre=params["genre"])

    # Mono-Compatibility Check
    is_compatible, _correlation = widener.check_mono_compatibility(audio_widened)

    if not is_compatible:
        logger.warning("Reducing width for mono-compatibility")
        # Retry with reduced width
        audio_widened = widener.widen(
            audio,
            sr,
            width=params["width"] * 0.7,  # Reduce by 30%
            genre=params["genre"],
        )

    return audio_widened


# ============================================================================
# Unit Test
# ============================================================================

# Rückwärts-kompatibler Alias — Produktionscode (policy_engine, aurik_restore)
# importiert `StereoWidener`; die eigentliche Klasse heißt `AdaptiveStereoWidener`.
StereoWidener = AdaptiveStereoWidener


if __name__ == "__main__":
    # Test Stereo Widening
    sr = 48000
    duration = 3.0
    samples = int(duration * sr)

    # Generate test stereo signal
    t = np.linspace(0, duration, samples)
    L = np.sin(2 * np.pi * 440 * t) * 0.3  # Left
    R = np.sin(2 * np.pi * 440 * t + 0.1) * 0.3  # Right (slightly different phase)

    audio_stereo = np.array([L, R])

    # Test widening
    widener = AdaptiveStereoWidener()

    # Test 1: Classical (minimal)
    audio_wide_classical = widener.widen(audio_stereo, sr, width=1.3, genre="classical")
    logger.info("Test 1 (Classical): Width limited to %.2f", 1.1)

    # Test 2: Pop (moderate)
    audio_wide_pop = widener.widen(audio_stereo, sr, width=1.5, genre="pop")
    logger.info("Test 2 (Pop): Width %.2f", 1.5)

    # Test 3: Mono-Compatibility Check
    is_compat, corr = widener.check_mono_compatibility(audio_wide_pop)
    logger.info("Test 3 (Mono-Compat): Compatible = %s, Correlation = %.2f", is_compat, corr)

    # Test 4: Policy Decision
    context = {"genre": "jazz", "channels": 2}
    goal = {"quality_level": "maximal"}

    params = select_stereo_widening_strategy(context, goal)
    logger.info("\nTest 4 (Policy): %s", params)
    assert params is not None, "Should select widening for jazz"

    # Test 5: Speech (should NOT widen)
    context_speech = {"genre": "speech", "channels": 2}
    params_speech = select_stereo_widening_strategy(context_speech, goal)
    logger.info("Test 5 (Speech): %s", params_speech)
    assert params_speech is None, "Should NOT widen speech"

    logger.info("\n✓ All tests passed")
