"""
Genre-Adaptive Multi-Band EQ
=============================

Genre-spezifische EQ-Curves für "Klarheit" und "Studio-Polish".

Problem: Nach Denoise fehlt oft:
- Glanz in den Höhen (Classical)
- Wärme in den Mitten (Jazz)
- Intelligibility (Speech)

Lösung: Genre-basierte EQ-Profile mit:
- Spectral Balance Analysis
- Safe Gain Ranges (max ±6dB)
- Adaptive Adjustment

Safeguards:
- Conservative Defaults (max ±3dB für Classical)
- Genre-Adaptive Curves (respektiert Original-Aesthetics)
- Spectral Balance Check (avoid over-brightness)
- Parallel Processing
"""

# v10.101 SOTA: Adaptiver EQ mit Bark-Band-Frequenzkorrektur.
import logging
from typing import Any

import numpy as np
import scipy.signal as signal

_logger = logging.getLogger(__name__)


class GenreAdaptiveEQ:
    """
    Multi-Band Parametric EQ mit Genre-spezifischen Presets.
    """

    def __init__(self):
        self.name = "Genre-Adaptive EQ"

        # Genre-Specific EQ Profiles
        self.eq_profiles: dict[str, dict[str, Any]] = {
            "classical": {
                "philosophy": "transparency_above_all",
                "bands": [
                    {
                        "freq": 40,
                        "gain": -2.0,
                        "q": 0.7,
                        "name": "sub_bass",
                    },  # Rumble reduction
                    {"freq": 100, "gain": 0.0, "q": 1.0, "name": "bass"},  # Natural
                    {"freq": 500, "gain": +1.0, "q": 1.0, "name": "low_mid"},  # Body
                    {
                        "freq": 2000,
                        "gain": +2.0,
                        "q": 1.2,
                        "name": "high_mid",
                    },  # Clarity
                    {
                        "freq": 6000,
                        "gain": +3.0,
                        "q": 1.0,
                        "name": "presence",
                    },  # Brilliance
                    {"freq": 12000, "gain": +1.0, "q": 0.7, "name": "air"},  # Air
                ],
            },
            "jazz": {
                "philosophy": "warmth_with_authenticity",
                "bands": [
                    {"freq": 60, "gain": -1.0, "q": 0.7, "name": "sub_bass"},
                    {"freq": 120, "gain": +2.0, "q": 1.0, "name": "bass"},  # Warmth
                    {"freq": 400, "gain": +1.0, "q": 1.0, "name": "low_mid"},
                    {"freq": 1500, "gain": +1.0, "q": 1.2, "name": "high_mid"},
                    {"freq": 5000, "gain": +2.0, "q": 1.0, "name": "presence"},
                    {"freq": 10000, "gain": +2.0, "q": 0.7, "name": "air"},
                ],
            },
            "speech": {
                "philosophy": "intelligibility_above_all",
                "bands": [
                    {
                        "freq": 40,
                        "gain": -6.0,
                        "q": 0.7,
                        "name": "sub_bass",
                    },  # Strong reduction
                    {
                        "freq": 150,
                        "gain": -3.0,
                        "q": 1.0,
                        "name": "bass",
                    },  # Proximity reduction
                    {"freq": 500, "gain": 0.0, "q": 1.0, "name": "low_mid"},
                    {
                        "freq": 2500,
                        "gain": +4.0,
                        "q": 1.5,
                        "name": "high_mid",
                    },  # Intelligibility!
                    {
                        "freq": 6000,
                        "gain": +6.0,
                        "q": 1.0,
                        "name": "presence",
                    },  # Sibilance region
                    {"freq": 12000, "gain": 0.0, "q": 0.7, "name": "air"},  # Natural
                ],
            },
            "rock": {
                "philosophy": "punch_and_clarity",
                "bands": [
                    {"freq": 60, "gain": +1.0, "q": 1.0, "name": "sub_bass"},
                    {"freq": 120, "gain": +2.0, "q": 1.0, "name": "bass"},  # Kick drum
                    {
                        "freq": 800,
                        "gain": -1.0,
                        "q": 1.5,
                        "name": "low_mid",
                    },  # Muddy region
                    {
                        "freq": 3000,
                        "gain": +3.0,
                        "q": 1.2,
                        "name": "high_mid",
                    },  # Guitars
                    {
                        "freq": 8000,
                        "gain": +4.0,
                        "q": 1.0,
                        "name": "presence",
                    },  # Cymbals
                    {"freq": 15000, "gain": +2.0, "q": 0.7, "name": "air"},
                ],
            },
            "pop": {
                "philosophy": "modern_commercial_sound",
                "bands": [
                    {"freq": 40, "gain": -2.0, "q": 0.7, "name": "sub_bass"},
                    {"freq": 100, "gain": +2.0, "q": 1.0, "name": "bass"},
                    {"freq": 300, "gain": -1.0, "q": 1.2, "name": "low_mid"},
                    {"freq": 2000, "gain": +2.0, "q": 1.2, "name": "high_mid"},
                    {"freq": 6000, "gain": +4.0, "q": 1.0, "name": "presence"},
                    {"freq": 12000, "gain": +3.0, "q": 0.7, "name": "air"},
                ],
            },
        }

    def apply_eq(self, audio: np.ndarray, sr: int, genre: str = "unknown", strength: float = 0.7) -> np.ndarray:
        """
        Wendet an: genre-specific EQ.

        Args:
            audio: Input audio
            sr: Sample rate
            genre: Genre
            strength: EQ strength (0.0 = none, 0.7 = conservative, 1.0 = full)

        Returns:
            EQ'd audio
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # Get profile
        profile = self.eq_profiles.get(genre, self.eq_profiles["jazz"])  # Default: Jazz

        _logger.info(
            "Applying EQ: genre=%s, strength=%.2f, philosophy='%s'",
            genre,
            strength,
            profile["philosophy"],
        )

        # Apply each band
        audio_eq = audio.copy()
        for band in profile["bands"]:
            audio_eq = self._apply_peaking_eq(
                audio_eq,
                sr,
                freq=band["freq"],  # type: ignore[index]
                gain=band["gain"] * strength,  # type: ignore[index]  # Scale by strength
                q=band["q"],  # type: ignore[index]
            )

        # Spectral Balance Check (Safeguard #4)
        audio_eq = self._check_and_adjust_spectral_balance(audio, audio_eq, sr, genre)

        # Parallel Mix (Safeguard #1)
        audio_final = (1 - strength) * audio + strength * audio_eq

        _logger.info("EQ applied (%d bands)", len(profile["bands"]))

        result = audio_final
        return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype)

    def _apply_peaking_eq(self, audio: np.ndarray, sr: int, freq: float, gain: float, q: float = 1.0) -> np.ndarray:
        """
        Wendet an: peaking EQ filter (parametric EQ).

        Args:
            audio: Input audio
            sr: Sample rate
            freq: Center frequency (Hz)
            gain: Gain in dB (positive = boost, negative = cut)
            q: Q factor (bandwidth, 1.0 = one octave)

        Returns:
            Filtered audio
        """
        # Design peaking filter
        # Convert gain from dB to linear
        A = 10 ** (gain / 40.0)  # Peaking filter uses gain/40

        # Normalize frequency
        w0 = 2 * np.pi * freq / sr

        # Check if frequency is valid
        if w0 <= 0 or w0 >= np.pi:
            _logger.warning("Frequency %.1f Hz out of range for SR %d Hz, skipping", freq, sr)
            return audio

        alpha = np.sin(w0) / (2 * q)

        # IIR coefficients (peaking)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        # Apply filter
        audio_filtered = signal.lfilter(b, a, audio)

        return np.asarray(audio_filtered)

    def _check_and_adjust_spectral_balance(
        self, audio_original: np.ndarray, audio_eq: np.ndarray, sr: int, genre: str
    ) -> np.ndarray:
        """
        Prüft if EQ created unnatural spectral balance.

        Safeguard: Prevent over-brightness or excessive bass.
        """
        # Measure spectral balance
        balance_original = self._measure_spectral_balance(audio_original, sr)
        balance_eq = self._measure_spectral_balance(audio_eq, sr)

        # Check High/Bass ratio
        ratio_original = balance_original["high"] / (balance_original["bass"] + 1e-10)
        ratio_eq = balance_eq["high"] / (balance_eq["bass"] + 1e-10)

        # Threshold: max 3x change
        max_ratio_change = 3.0

        if ratio_eq / ratio_original > max_ratio_change:
            # Too bright!
            _logger.warning(
                "EQ made audio too bright (ratio %.1f / %.1f), reducing highs",
                ratio_eq,
                ratio_original,
            )

            # Reduce high-frequency gain by 3dB
            audio_eq = self._apply_peaking_eq(audio_eq, sr, freq=8000, gain=-3.0, q=0.5)

        elif ratio_original / ratio_eq > max_ratio_change:
            # Too dark!
            _logger.warning("EQ made audio too dark, adjusting")
            audio_eq = self._apply_peaking_eq(audio_eq, sr, freq=8000, gain=+2.0, q=0.5)

        return audio_eq

    def _measure_spectral_balance(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Misst spectral energy in Bass, Mid, High regions.
        """
        # FFT
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Define regions
        bass_mask = (freqs >= 20) & (freqs < 250)
        mid_mask = (freqs >= 250) & (freqs < 4000)
        high_mask = (freqs >= 4000) & (freqs < 16000)

        # Energy per region
        bass_energy = np.sum(spectrum[bass_mask] ** 2)
        mid_energy = np.sum(spectrum[mid_mask] ** 2)
        high_energy = np.sum(spectrum[high_mask] ** 2)

        return {"bass": bass_energy, "mid": mid_energy, "high": high_energy}


# ============================================================================
# Pipeline Integration
# ============================================================================


def select_eq_strategy(context: dict, goal: dict) -> dict:
    """
    Policy-Engine Decision: Welches EQ-Profil?

    Args:
        context: Audio context (genre from PANNS!)
        goal: Processing goals

    Returns:
        eq_params dict
    """
    genre = context.get("genre", "unknown")

    # Determine strength based on goal
    if goal.get("priority") == "transparency":
        strength = 0.5  # Sanft
    elif goal.get("quality_level") == "maximal":
        strength = 0.8  # Moderat
    else:
        strength = 0.7  # Default

    params = {"genre": genre, "strength": strength}

    _logger.info("EQ Strategy SELECTED: genre=%s, strength=%.2f", genre, strength)

    return params


def apply_adaptive_eq(audio: np.ndarray, sr: int, context: dict, goal: dict) -> np.ndarray:
    """
    Convenience function für Pipeline-Integration.

    Nutzt PANNS Genre-Detection (bereits vorhanden in context!).
    """
    # Policy Decision
    params = select_eq_strategy(context, goal)

    # Apply EQ
    eq = GenreAdaptiveEQ()
    audio_eq = eq.apply_eq(audio, sr, genre=params["genre"], strength=params["strength"])

    return audio_eq


# ============================================================================
# Unit Test
# ============================================================================

if __name__ == "__main__":
    # Test Genre-Adaptive EQ
    sr = 48000
    duration = 3.0
    samples = int(duration * sr)

    # Generate test signal (sine sweep)
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3

    eq = GenreAdaptiveEQ()

    # Test 1: Classical (transparent)
    audio_classical = eq.apply_eq(audio, sr, genre="classical", strength=0.7)
    _logger.info("Test 1 (Classical): EQ applied (transparency focus)")

    # Test 2: Speech (intelligibility)
    audio_speech = eq.apply_eq(audio, sr, genre="speech", strength=0.8)
    _logger.info("Test 2 (Speech): EQ applied (intelligibility focus)")

    # Test 3: Jazz (warmth)
    audio_jazz = eq.apply_eq(audio, sr, genre="jazz", strength=0.7)
    _logger.info("Test 3 (Jazz): EQ applied (warmth focus)")

    # Test 4: Policy Decision
    context = {"genre": "classical"}  # From PANNS!
    goal = {"priority": "transparency"}

    params = select_eq_strategy(context, goal)
    _logger.info("Test 4 (Policy): %s", params)
    assert params["genre"] == "classical", "Should use classical profile"
    assert params["strength"] == 0.5, "Transparency should use 0.5 strength"

    # Test 5: Spectral Balance Check
    balance = eq._measure_spectral_balance(audio, sr)
    _logger.info(
        "Test 5 (Spectral Balance): Bass=%.0f, Mid=%.0f, High=%.0f",
        balance["bass"],
        balance["mid"],
        balance["high"],
    )

    _logger.info("All tests passed")
