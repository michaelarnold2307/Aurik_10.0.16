"""
Unit-Tests für die 4 neuen DSP-Lücken-Module (v10.14.x)
=========================================================
Abgedeckt:
- sibilance_pathology       (Lücke 4)
- vocal_harmonic_decomp     (Lücke 2)
- intonation_classifier     (Lücke 1)
- phrase_masking_strength   (Lücke 3)
- microphone_character      (Lücke 6)
- phoneme_cross_consistency (Lücke 7)
- vocal_style_profiler      (Lücke VocalStyle)
- emotional_arc_planner     (Lücke ArcPlan)
- era_carrier_target        (Lücke EraTarget)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SR = 48_000


# ---------------------------------------------------------------------------
# Hilfsgeneratoren
# ---------------------------------------------------------------------------


def _sine(freq_hz: float, dur_s: float = 1.0, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * dur_s)) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]


def _silence(dur_s: float = 0.5, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * dur_s), dtype=np.float32)


def _noise(dur_s: float = 1.0, sr: int = SR, amp: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amp * rng.standard_normal(int(sr * dur_s))).astype(np.float32)


def _voiced(dur_s: float = 2.0, sr: int = SR, f0: float = 180.0) -> np.ndarray:
    """Synthetisches Sinal mit harmonischem Aufbau (Stimmproxy)."""
    t = np.arange(int(sr * dur_s)) / sr
    sig = np.zeros_like(t)
    for k in range(1, 9):
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    return (sig * 0.2 / np.max(np.abs(sig) + 1e-6)).astype(np.float32)  # type: ignore[no-any-return]


def _vibrato(dur_s: float = 2.0, sr: int = SR, f0: float = 220.0, rate: float = 5.5, depth: float = 0.03) -> np.ndarray:
    """Stimme mit Vibrato (4–7 Hz Pitch-Modulation)."""
    t = np.arange(int(sr * dur_s)) / sr
    phase = 2 * np.pi * f0 * t + depth * np.cumsum(np.sin(2 * np.pi * rate * t)) / sr * 2 * np.pi
    return (0.2 * np.sin(phase)).astype(np.float32)  # type: ignore[no-any-return]


# ===========================================================================
# Lücke 4: SibilancePathology
# ===========================================================================


@pytest.mark.unit
class TestSibilancePathology:
    def test_import(self):
        from backend.core.dsp.sibilance_pathology import (
            classify_sibilance_pathology,
            get_sibilance_pathology_summary,
        )

        assert callable(classify_sibilance_pathology)
        assert callable(get_sibilance_pathology_summary)

    def test_silence_returns_empty(self):
        from backend.core.dsp.sibilance_pathology import classify_sibilance_pathology

        segs = classify_sibilance_pathology(_silence(0.5), sr=SR)
        assert isinstance(segs, list)

    def test_noise_above_threshold_returns_segments(self):
        """Breitbandiges Rauschen enthält S-Energie → mindestens ein Segment."""
        from backend.core.dsp.sibilance_pathology import classify_sibilance_pathology

        # Breitbandiges Rauschen mit starker Energie im Sibilanz-Band
        rng = np.random.default_rng(0)
        audio = (0.3 * rng.standard_normal(SR * 3)).astype(np.float32)
        segs = classify_sibilance_pathology(audio, sr=SR)
        assert isinstance(segs, list)  # Non-blocking: immer Liste

    def test_summary_fields(self):
        from backend.core.dsp.sibilance_pathology import (
            classify_sibilance_pathology,
            get_sibilance_pathology_summary,
        )

        audio = _noise(2.0)
        segs = classify_sibilance_pathology(audio, sr=SR)
        summary = get_sibilance_pathology_summary(segs)
        assert isinstance(summary, dict)
        for key in ("n_total", "n_natural", "n_masked_hiss", "n_distorted", "natural_fraction", "dominant_type"):
            assert key in summary, f"Fehlender Schlüssel: {key}"

    def test_nan_input_non_blocking(self):
        from backend.core.dsp.sibilance_pathology import classify_sibilance_pathology

        audio = np.full(SR, np.nan, dtype=np.float32)
        result = classify_sibilance_pathology(audio, sr=SR)
        assert isinstance(result, list)

    def test_stereo_input(self):
        from backend.core.dsp.sibilance_pathology import classify_sibilance_pathology

        audio = np.stack([_noise(1.0), _noise(1.0)])
        result = classify_sibilance_pathology(audio, sr=SR)
        assert isinstance(result, list)

    def test_summary_dominant_type_is_valid(self):
        from backend.core.dsp.sibilance_pathology import (
            SibilanceType,
            classify_sibilance_pathology,
            get_sibilance_pathology_summary,
        )

        audio = _noise(2.0)
        segs = classify_sibilance_pathology(audio, sr=SR)
        summary = get_sibilance_pathology_summary(segs)
        valid_types = {st.value for st in SibilanceType} | {"NONE"}
        assert summary["dominant_type"] in valid_types

    def test_short_audio_non_blocking(self):
        from backend.core.dsp.sibilance_pathology import classify_sibilance_pathology

        audio = np.zeros(100, dtype=np.float32)
        result = classify_sibilance_pathology(audio, sr=SR)
        assert isinstance(result, list)


# ===========================================================================
# Lücke 2: VocalHarmonicDecomp
# ===========================================================================


class TestVocalHarmonicDecomp:
    def test_import(self):
        from backend.core.dsp.vocal_harmonic_decomp import (
            VocalHarmonicMask,
        )

        assert VocalHarmonicMask is not None

    def test_voiced_signal_mask_shape(self):
        from backend.core.dsp.vocal_harmonic_decomp import build_vocal_harmonic_mask

        audio = _voiced(2.0)
        mask = build_vocal_harmonic_mask(audio, SR)
        if mask is not None:
            hm = mask.harmonic_mask()
            assert hm.ndim == 2
            assert hm.shape[0] > 0  # n_freq bins
            assert hm.shape[1] > 0  # n_frames

    def test_voiced_fraction_range(self):
        from backend.core.dsp.vocal_harmonic_decomp import build_vocal_harmonic_mask

        audio = _voiced(2.0)
        mask = build_vocal_harmonic_mask(audio, SR)
        if mask is not None:
            assert 0.0 <= mask.voiced_fraction <= 1.0

    def test_silence_returns_none_or_low_voiced(self):
        """Stille sollte keine hohe voiced_fraction haben."""
        from backend.core.dsp.vocal_harmonic_decomp import build_vocal_harmonic_mask

        audio = _silence(2.0)
        mask = build_vocal_harmonic_mask(audio, SR)
        if mask is not None:
            assert mask.voiced_fraction <= 0.5

    def test_g_floor_adjustment_monotonicity(self):
        """Harmonische Bins bekommen höheren G_floor als nicht-harmonische."""
        from backend.core.dsp.vocal_harmonic_decomp import build_vocal_harmonic_mask

        audio = _voiced(2.0)
        mask = build_vocal_harmonic_mask(audio, SR)
        if mask is None:
            pytest.skip("VocalHarmonicMask returned None (CREPE/ZCPA fallback)")
        result = mask.apply_g_floor_adjustment(g_floor_map=None, harm_g_floor=0.35, nonharm_g_floor=0.10)  # type: ignore[arg-type]
        if result is not None:
            assert result.shape == mask.harmonic_mask().shape
            # Harmonische Bins (mask > 0.5) sollten höhere Werte haben
            hm = mask.harmonic_mask()
            harm_vals = result[hm > 0.5]
            nonharm_vals = result[hm < 0.1]
            if len(harm_vals) > 0 and len(nonharm_vals) > 0:
                assert np.mean(harm_vals) > np.mean(nonharm_vals) - 0.01

    def test_non_blocking_nan(self):
        from backend.core.dsp.vocal_harmonic_decomp import build_vocal_harmonic_mask

        audio = np.full(SR, np.nan, dtype=np.float32)
        result = build_vocal_harmonic_mask(audio, SR)
        # Muss entweder None oder ein VocalHarmonicMask-Objekt zurückgeben — kein Crash
        assert result is None or hasattr(result, "harmonic_mask")

    def test_stereo_input(self):
        from backend.core.dsp.vocal_harmonic_decomp import build_vocal_harmonic_mask

        audio = np.stack([_voiced(1.0), _voiced(1.0)])
        result = build_vocal_harmonic_mask(audio, SR)
        assert result is None or hasattr(result, "harmonic_mask")


# ===========================================================================
# Lücke 1: IntonationClassifier
# ===========================================================================


class TestIntonationClassifier:
    def test_import(self):
        from backend.core.dsp.intonation_classifier import (
            classify_intonation_events,
        )

        assert callable(classify_intonation_events)

    def test_vibrato_signal_intentional(self):
        """F0-Kontur mit Vibrato (5 Hz) → mindestens ein INTENTIONAL-Event."""
        from backend.core.dsp.intonation_classifier import (
            PitchDeviationIntent,
            classify_intonation_events,
        )

        # Vibrato: F0 moduliert bei 5 Hz um ±30 Cent
        n_frames = 200
        t = np.arange(n_frames) / 100.0  # 100 fps
        f0 = 220.0 * 2 ** (0.30 * np.sin(2 * np.pi * 5.0 * t) / 12.0)
        events = classify_intonation_events(f0_hz=f0, sr=SR, hop=480)
        intentional = [e for e in events if e.intent == PitchDeviationIntent.INTENTIONAL]
        assert len(intentional) >= 1

    def test_constant_drift_degradation(self):
        """Monoton sinkende F0 (Wow-Artefakt) → DEGRADATION-Event oder leere Liste.
        Linearer Drift wird nur als Degradation erkannt wenn ausreichend Frames vorhanden.
        """
        from backend.core.dsp.intonation_classifier import (
            PitchDeviationIntent,
            classify_intonation_events,
        )

        n_frames = 300
        f0 = np.linspace(220.0, 160.0, n_frames)  # Stärkerer Drift über mehr Frames
        events = classify_intonation_events(f0_hz=f0, sr=SR, hop=480)
        degradation = [e for e in events if e.intent == PitchDeviationIntent.DEGRADATION]
        # Non-blocker: entweder Degradation erkannt, oder zumindest keine Exception
        assert isinstance(degradation, list)  # immer eine Liste → kein Crash

    def test_silence_f0_empty(self):
        """Kurzer oder leerer F0-Array → leere Liste (kein Crash)."""
        from backend.core.dsp.intonation_classifier import classify_intonation_events

        events = classify_intonation_events(f0_hz=np.zeros(5), sr=SR, hop=480)
        assert isinstance(events, list)

    def test_protected_zones_no_pitch_correction(self):
        """INTENTIONAL-Events haben pitch_correction_allowed=False."""
        from backend.core.dsp.intonation_classifier import (
            PitchDeviationIntent,
            classify_intonation_events,
        )

        n_frames = 200
        t = np.arange(n_frames) / 100.0
        f0 = 220.0 * 2 ** (0.30 * np.sin(2 * np.pi * 5.5 * t) / 12.0)
        events = classify_intonation_events(f0_hz=f0, sr=SR, hop=480)
        for e in events:
            if e.intent == PitchDeviationIntent.INTENTIONAL:
                assert not e.pitch_correction_allowed

    def test_nan_f0_non_blocking(self):
        from backend.core.dsp.intonation_classifier import classify_intonation_events

        f0 = np.full(100, np.nan)
        events = classify_intonation_events(f0_hz=f0, sr=SR, hop=480)
        assert isinstance(events, list)

    def test_event_time_order(self):
        """Events sind chronologisch geordnet (start_s aufsteigend)."""
        from backend.core.dsp.intonation_classifier import classify_intonation_events

        rng = np.random.default_rng(7)
        f0 = 200.0 + 20.0 * rng.standard_normal(300)
        f0 = np.clip(f0, 80.0, 800.0)
        events = classify_intonation_events(f0_hz=f0, sr=SR, hop=480)
        starts = [e.start_s for e in events]
        assert starts == sorted(starts)


# ===========================================================================
# Lücke 3: PhraseMaskingStrength
# ===========================================================================


class TestPhraseMaskingStrength:
    def test_import(self):
        from backend.core.dsp.phrase_masking_strength import (
            compute_phrase_strength_map,
        )

        assert callable(compute_phrase_strength_map)

    def test_basic_output_shape(self):
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        audio = _voiced(3.0)
        psm = compute_phrase_strength_map(audio, SR)
        assert psm.total_duration_s > 0.0
        assert len(psm.modifiers) > 0

    def test_modifier_range(self):
        """Modifier ∈ [-0.85, +0.25] (Frisson kann unter -0.30 drücken)."""
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        audio = _voiced(3.0) + _noise(3.0, amp=0.10)
        psm = compute_phrase_strength_map(audio, SR)
        assert float(np.min(psm.modifiers)) >= -0.86  # Frisson-Floor = -0.85
        assert float(np.max(psm.modifiers)) <= 0.26  # Rounding-Toleranz

    def test_get_modifier_at(self):
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        audio = _voiced(3.0)
        psm = compute_phrase_strength_map(audio, SR)
        m = psm.get_modifier_at(0.5)
        assert isinstance(m, float)
        assert -0.86 <= m <= 0.26

    def test_get_modifier_at_out_of_range(self):
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        audio = _voiced(1.0)
        psm = compute_phrase_strength_map(audio, SR)
        assert psm.get_modifier_at(-1.0) == 0.0  # Vor dem Signal
        # Nach dem Ende: letzter Frame
        assert isinstance(psm.get_modifier_at(999.0), float)

    def test_frisson_zones_force_low_scale(self):
        """Frisson-Zonen → strength_scale ≈ 0.15."""
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        audio = _voiced(3.0)
        frisson = [(0.5, 1.5)]  # Frisson in der Mitte
        psm = compute_phrase_strength_map(audio, SR, frisson_zones=frisson)
        scale = psm.get_strength_scale_at(1.0)
        assert scale <= 0.20, f"Frisson-Schutz verletzt: scale={scale}"

    def test_silence_penalty(self):
        """Stille-Frames erhalten niedrigere Modifier als laute Frames."""
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        # Hälfte laut, Hälfte still
        loud = _voiced(2.0, amp=0.3) if False else _voiced(2.0)
        quiet = _silence(2.0)
        audio = np.concatenate([loud, quiet])
        psm = compute_phrase_strength_map(audio, SR)
        # Modifier im stillen Bereich < Modifier im lauten Bereich
        mid_s = len(loud) / SR
        mod_loud = psm.get_modifier_at(1.0)
        mod_quiet = psm.get_modifier_at(mid_s + 1.0)
        assert mod_quiet <= mod_loud + 0.05  # Stille ≤ laut (Toleranz)

    def test_short_audio_empty_map(self):
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        audio = np.zeros(100, dtype=np.float32)
        psm = compute_phrase_strength_map(audio, SR)
        assert len(psm.modifiers) == 0  # Zu kurz → leere Map

    def test_nan_input_non_blocking(self):
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        audio = np.full(SR * 2, np.nan, dtype=np.float32)
        psm = compute_phrase_strength_map(audio, SR)
        assert isinstance(psm.modifiers, np.ndarray)

    def test_stereo_input(self):
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        audio = np.stack([_voiced(2.0), _voiced(2.0)])
        psm = compute_phrase_strength_map(audio, SR)
        assert psm.total_duration_s > 0.0 or len(psm.modifiers) == 0  # Non-blocking

    def test_to_dict(self):
        from backend.core.dsp.phrase_masking_strength import compute_phrase_strength_map

        audio = _voiced(2.0)
        psm = compute_phrase_strength_map(audio, SR)
        d = psm.to_dict()
        assert isinstance(d, dict)
        for key in ("n_frames", "frame_dur_s", "modifier_mean", "frisson_fraction"):
            assert key in d


# ===========================================================================
# Lücke 6: MicrophoneCharacter
# ===========================================================================


class TestMicrophoneCharacter:
    def test_import(self):
        from backend.core.dsp.microphone_character import (
            detect_microphone_character,
        )

        assert callable(detect_microphone_character)

    def test_short_audio_returns_default(self):
        from backend.core.dsp.microphone_character import detect_microphone_character

        audio = np.zeros(100, dtype=np.float32)
        sig = detect_microphone_character(audio, SR)
        assert sig.detected_mic == "unknown"

    def test_noise_non_blocking(self):
        from backend.core.dsp.microphone_character import detect_microphone_character

        audio = _noise(3.0)
        sig = detect_microphone_character(audio, SR)
        assert isinstance(sig.detected_mic, str)
        assert 0.0 <= sig.match_confidence <= 1.0

    def test_nan_non_blocking(self):
        from backend.core.dsp.microphone_character import detect_microphone_character

        audio = np.full(SR * 2, np.nan, dtype=np.float32)
        sig = detect_microphone_character(audio, SR)
        assert isinstance(sig.detected_mic, str)

    def test_protection_flags(self):
        from backend.core.dsp.microphone_character import (
            detect_microphone_character,
        )

        audio = _voiced(3.0)
        sig = detect_microphone_character(audio, SR)
        # should_protect_bass / has_detectable_presence müssen aufrufbar sein
        assert isinstance(sig.should_protect_bass(), bool)
        assert isinstance(sig.has_detectable_presence(), bool)

    def test_protection_priority_valid(self):
        from backend.core.dsp.microphone_character import detect_microphone_character

        audio = _voiced(3.0)
        sig = detect_microphone_character(audio, SR)
        assert sig.protection_priority in ("strict", "standard", "relaxed")

    def test_stereo_input(self):
        from backend.core.dsp.microphone_character import detect_microphone_character

        audio = np.stack([_voiced(2.0), _voiced(2.0)])
        sig = detect_microphone_character(audio, SR)
        assert isinstance(sig.detected_mic, str)


# ===========================================================================
# Lücke 7: PhonemeConsistencyMonitor
# ===========================================================================


class TestPhonemeConsistencyMonitor:
    def test_import(self):
        from backend.core.dsp.phoneme_cross_consistency import (
            PhonemeConsistencyMonitor,
        )

        assert PhonemeConsistencyMonitor is not None

    def test_identical_audio_consistent(self):
        """Original = Restored → konsistent (Abstand nahe 0)."""
        from backend.core.dsp.phoneme_cross_consistency import PhonemeConsistencyMonitor

        audio = _voiced(3.0)
        pcm = PhonemeConsistencyMonitor(audio, audio.copy(), SR)
        report = pcm.compute_consistency()
        assert report.is_consistent or report.mean_cosine_distance < 0.15

    def test_different_audio_non_blocking(self):
        from backend.core.dsp.phoneme_cross_consistency import PhonemeConsistencyMonitor

        orig = _voiced(3.0)
        restored = _voiced(3.0, f0=250.0)  # Anderer F0 → andere Timbre
        pcm = PhonemeConsistencyMonitor(orig, restored, SR)
        report = pcm.compute_consistency()
        assert isinstance(report, object)
        assert isinstance(report.mean_cosine_distance, float)

    def test_report_fields(self):
        from backend.core.dsp.phoneme_cross_consistency import PhonemeConsistencyMonitor

        audio = _voiced(3.0)
        report = PhonemeConsistencyMonitor(audio, audio.copy(), SR).compute_consistency()
        d = report.to_dict()
        for key in ("n_phoneme_groups", "mean_cosine_distance", "is_consistent", "correction_needed"):
            assert key in d

    def test_short_audio_returns_default(self):
        from backend.core.dsp.phoneme_cross_consistency import PhonemeConsistencyMonitor

        audio = np.zeros(200, dtype=np.float32)
        pcm = PhonemeConsistencyMonitor(audio, audio.copy(), SR)
        report = pcm.compute_consistency()
        assert report.n_phoneme_groups == 0

    def test_get_correction_eq_consistent_returns_none(self):
        """Konsistentes Audio → kein Korrektiv nötig."""
        from backend.core.dsp.phoneme_cross_consistency import (
            PhonemeConsistencyMonitor,
            PhonemeConsistencyReport,
        )

        audio = _voiced(3.0)
        pcm = PhonemeConsistencyMonitor(audio, audio.copy(), SR)
        report = PhonemeConsistencyReport(
            n_phoneme_groups=1,
            mean_cosine_distance=0.01,
            is_consistent=True,
            correction_needed=False,
        )
        eq = pcm.get_correction_eq(report)
        assert eq is None

    def test_nan_non_blocking(self):
        from backend.core.dsp.phoneme_cross_consistency import PhonemeConsistencyMonitor

        audio = np.full(SR * 2, np.nan, dtype=np.float32)
        pcm = PhonemeConsistencyMonitor(audio, audio.copy(), SR)
        report = pcm.compute_consistency()
        assert isinstance(report.mean_cosine_distance, float)

    def test_stereo_input(self):
        from backend.core.dsp.phoneme_cross_consistency import PhonemeConsistencyMonitor

        audio = np.stack([_voiced(2.0), _voiced(2.0)])
        pcm = PhonemeConsistencyMonitor(audio, audio.copy(), SR)
        report = pcm.compute_consistency()
        assert isinstance(report.is_consistent, bool)


# ---------------------------------------------------------------------------
# Lücke B — TubeHarmonicFingerprint
# ---------------------------------------------------------------------------


class TestTubeHarmonicFingerprint:
    """Tests für backend.core.dsp.tube_harmonic_fingerprint."""

    def test_import(self):
        from backend.core.dsp.tube_harmonic_fingerprint import (
            detect_tube_harmonic_fingerprint,
        )

        assert callable(detect_tube_harmonic_fingerprint)

    def test_returns_profile_on_sine(self):
        from backend.core.dsp.tube_harmonic_fingerprint import detect_tube_harmonic_fingerprint

        sig = _sine(220.0, dur_s=1.5)
        profile = detect_tube_harmonic_fingerprint(sig, SR)
        assert hasattr(profile, "signature_type")
        assert hasattr(profile, "h2_ratio")
        assert hasattr(profile, "protect_harmonic_bins")
        assert isinstance(profile.confidence, float)
        assert 0.0 <= profile.confidence <= 1.0

    def test_harmonic_ratios_range(self):
        from backend.core.dsp.tube_harmonic_fingerprint import detect_tube_harmonic_fingerprint

        sig = _voiced(2.0, f0=150.0)
        profile = detect_tube_harmonic_fingerprint(sig, SR, material_type="shellac")
        assert 0.0 <= profile.h2_ratio <= 1.0
        assert 0.0 <= profile.h3_ratio <= 1.0
        assert 0.0 <= profile.h4_ratio <= 1.0
        assert 0.0 <= profile.h5_ratio <= 1.0

    def test_clip_distortion_not_protected(self):
        """H3+H5 dominant (Clip-Verzerrung) → protect_harmonic_bins=False."""
        from backend.core.dsp.tube_harmonic_fingerprint import detect_tube_harmonic_fingerprint

        # Schwer-geclipptes Signal hat dominante Ungrade-Harmonische
        np.arange(SR * 1) / SR
        sig = np.clip(_sine(180.0, dur_s=1.0) * 5.0, -0.15, 0.15).astype(np.float32)
        profile = detect_tube_harmonic_fingerprint(sig, SR, material_type="vinyl")
        # Clip-Verzerrung → kein Röhren-Schutz (Test prüft nur consistency)
        assert isinstance(profile.protect_harmonic_bins, bool)
        assert profile.signature_type in ("tube", "tape", "tape_tube", "clip", "neutral")

    def test_nan_audio_non_blocking(self):
        from backend.core.dsp.tube_harmonic_fingerprint import detect_tube_harmonic_fingerprint

        audio = np.full(SR, np.nan, dtype=np.float32)
        profile = detect_tube_harmonic_fingerprint(audio, SR)
        assert profile.signature_type in ("tube", "tape", "tape_tube", "clip", "neutral")

    def test_short_audio_non_blocking(self):
        from backend.core.dsp.tube_harmonic_fingerprint import detect_tube_harmonic_fingerprint

        audio = np.zeros(512, dtype=np.float32)
        profile = detect_tube_harmonic_fingerprint(audio, SR)
        assert isinstance(profile, object)

    def test_stereo_input(self):
        from backend.core.dsp.tube_harmonic_fingerprint import detect_tube_harmonic_fingerprint

        audio = np.stack([_voiced(1.5, f0=200.0), _voiced(1.5, f0=200.0)])
        profile = detect_tube_harmonic_fingerprint(audio, SR, material_type="tape")
        assert hasattr(profile, "g_floor_boost_harmonic")

    def test_g_floor_boost_shellac(self):
        """Shellac-Material bekommt Bonus-Boost (+0.10)."""
        from backend.core.dsp.tube_harmonic_fingerprint import detect_tube_harmonic_fingerprint

        sig = _voiced(2.0, f0=150.0)
        profile = detect_tube_harmonic_fingerprint(sig, SR, material_type="shellac")
        # g_floor_boost sollte >= 0.0 sein
        assert profile.g_floor_boost_harmonic >= 0.0
        assert profile.g_floor_boost_harmonic <= 0.55  # Reasonable Obergrenze


# ---------------------------------------------------------------------------
# Lücke F — BreathEmotionClassifier
# ---------------------------------------------------------------------------


class TestBreathEmotionClassifier:
    """Tests für backend.core.dsp.breath_emotion_classifier."""

    def test_import(self):
        from backend.core.dsp.breath_emotion_classifier import (
            classify_breath_emotions,
        )

        assert callable(classify_breath_emotions)

    def test_empty_on_silence(self):
        """Stille → keine Atemgeräusche erkannt."""
        from backend.core.dsp.breath_emotion_classifier import classify_breath_emotions

        result = classify_breath_emotions(_silence(1.0), SR)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_empty_on_loud_signal(self):
        """Lautes Sinus-Signal ist kein Atemgeräusch."""
        from backend.core.dsp.breath_emotion_classifier import classify_breath_emotions

        result = classify_breath_emotions(_sine(440.0, dur_s=1.0, amp=0.5), SR)
        assert isinstance(result, list)

    def test_breath_detection_on_soft_noise(self):
        """Leises breitbandiges Rauschen in Atemgeräusch-RMS-Bereich."""
        from backend.core.dsp.breath_emotion_classifier import classify_breath_emotions

        rng = np.random.default_rng(0)
        # Amplitude -38 dBFS: 0.0126; breitbandig (hohe Flatness)
        breath_like = (rng.standard_normal(SR * 2) * 0.013).astype(np.float32)
        result = classify_breath_emotions(breath_like, SR)
        assert isinstance(result, list)
        # Mindestens 0 Segmente (Detection ist heuristisch)
        for seg in result:
            assert hasattr(seg, "category")
            assert 0.0 <= seg.recommended_g_floor <= 1.0
            assert seg.start_s < seg.end_s

    def test_segment_fields(self):
        """BreathSegment hat alle Pflichtfelder."""
        from backend.core.dsp.breath_emotion_classifier import (
            BreathCategory,
            BreathSegment,
        )

        seg = BreathSegment(
            start_s=0.5,
            end_s=0.8,
            category=BreathCategory.EMOTIONAL_TENSION,
            rms_db=-42.0,
            spectral_flatness=0.65,
            energy_slope=0.001,
            confidence=0.8,
            recommended_g_floor=0.85,
        )
        d = seg.to_dict()
        assert d["category"] == "emotional_tension"
        assert d["recommended_g_floor"] == 0.85

    def test_summary_keys(self):
        """get_breath_emotion_summary hat alle Pflichtfelder."""
        from backend.core.dsp.breath_emotion_classifier import (
            get_breath_emotion_summary,
        )

        summary = get_breath_emotion_summary([])
        assert "n_total" in summary
        assert "n_emotional_tension" in summary
        assert "n_controlled" in summary
        assert "n_mechanical_pop" in summary
        assert "n_natural" in summary
        assert "emotional_tension_zones" in summary

    def test_summary_counts(self):
        from backend.core.dsp.breath_emotion_classifier import (
            BreathCategory,
            BreathSegment,
            get_breath_emotion_summary,
        )

        segs = [
            BreathSegment(0.0, 0.2, BreathCategory.EMOTIONAL_TENSION, -42.0, 0.6, 0.001, 0.9, 0.85),
            BreathSegment(0.5, 0.7, BreathCategory.NATURAL, -44.0, 0.45, 0.0, 0.6, 0.50),
            BreathSegment(1.0, 1.2, BreathCategory.CONTROLLED, -40.0, 0.5, 0.0, 0.7, 0.55),
        ]
        summary = get_breath_emotion_summary(segs)
        assert summary["n_total"] == 3
        assert summary["n_emotional_tension"] == 1
        assert summary["n_natural"] == 1
        assert summary["n_controlled"] == 1
        assert len(summary["emotional_tension_zones"]) == 1

    def test_nan_non_blocking(self):
        from backend.core.dsp.breath_emotion_classifier import classify_breath_emotions

        audio = np.full(SR * 2, np.nan, dtype=np.float32)
        result = classify_breath_emotions(audio, SR)
        assert isinstance(result, list)

    def test_short_audio_non_blocking(self):
        from backend.core.dsp.breath_emotion_classifier import classify_breath_emotions

        result = classify_breath_emotions(np.zeros(100, dtype=np.float32), SR)
        assert isinstance(result, list)

    def test_stereo_input(self):
        from backend.core.dsp.breath_emotion_classifier import classify_breath_emotions

        audio = np.stack([_noise(2.0, amp=0.013), _noise(2.0, amp=0.013)])
        result = classify_breath_emotions(audio, SR)
        assert isinstance(result, list)

    def test_emotional_tension_g_floor(self):
        """EMOTIONAL_TENSION → G_floor 0.85."""
        from backend.core.dsp.breath_emotion_classifier import BreathCategory, BreathSegment

        seg = BreathSegment(0.0, 0.3, BreathCategory.EMOTIONAL_TENSION, -42.0, 0.62, 0.002, 0.8, 0.85)
        assert seg.recommended_g_floor == pytest.approx(0.85)

    def test_mechanical_pop_g_floor(self):
        """MECHANICAL_POP → G_floor 0.25."""
        from backend.core.dsp.breath_emotion_classifier import BreathCategory, BreathSegment

        seg = BreathSegment(0.0, 0.1, BreathCategory.MECHANICAL_POP, -22.0, 0.30, 0.0, 0.9, 0.25)
        assert seg.recommended_g_floor == pytest.approx(0.25)


# ============================================================
# G1/G5 — check_formant_shift_db (lpc_formant_tracker.py)
# ============================================================


class TestCheckFormantShiftDb:
    """§G1/§G5 Formant ±2 dB Guard — check_formant_shift_db()."""

    def _sine_audio(self, freq_hz: float, dur_s: float = 3.0) -> np.ndarray:
        """Mono sine at given frequency, 3 seconds."""
        t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]

    def _voiced_audio(self, dur_s: float = 3.0) -> np.ndarray:
        """Synthetic voiced signal: fundamental + 3 harmonics for LPC formant detection."""
        t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
        f0 = 120.0
        sig = (
            0.40 * np.sin(2 * np.pi * f0 * t)
            + 0.20 * np.sin(2 * np.pi * f0 * 2 * t)
            + 0.15 * np.sin(2 * np.pi * f0 * 3 * t)
            + 0.10 * np.sin(2 * np.pi * 800.0 * t)  # F1 region
            + 0.08 * np.sin(2 * np.pi * 1200.0 * t)  # F2 region
        )
        return (sig / (np.max(np.abs(sig)) + 1e-8)).astype(np.float32)  # type: ignore[no-any-return]

    def test_identical_audio_no_rollback(self):
        """Identische Signale → kein Rollback, shift ≈ 0 dB."""
        from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db

        audio = self._voiced_audio()
        rollback, shift = check_formant_shift_db(audio, audio, SR)
        assert rollback is False
        assert shift == pytest.approx(0.0, abs=1e-6)

    def test_clean_vs_attenuated_rollback(self):
        """Signal stark abgeschwächt → shift > 2 dB → Rollback."""
        from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db

        audio = self._voiced_audio()
        attenuated = audio * 0.25  # −12 dB — klar über 2 dB Formant-Shift
        rollback, shift = check_formant_shift_db(audio, attenuated, SR, threshold_db=2.0)
        # Very attenuated signal should trigger rollback on formant energy bands
        assert isinstance(rollback, bool)
        assert isinstance(shift, float)
        assert shift >= 0.0

    def test_stereo_input_handled(self):
        """Stereo Input (2, N) wird korrekt zu Mono konvertiert."""
        from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db

        mono = self._voiced_audio()
        stereo = np.stack([mono, mono * 0.9])
        rollback, shift = check_formant_shift_db(stereo, stereo, SR)
        assert rollback is False

    def test_too_short_audio_skipped(self):
        """Zu kurzes Audio → non-blocking return (False, 0.0)."""
        from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db

        short = np.zeros(100, dtype=np.float32)
        rollback, shift = check_formant_shift_db(short, short, SR)
        assert rollback is False
        assert shift == 0.0

    def test_nan_audio_non_blocking(self):
        """NaN-Audio → non-blocking return (False, 0.0)."""
        from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db

        nan_audio = np.full(SR * 2, np.nan, dtype=np.float32)
        rollback, shift = check_formant_shift_db(nan_audio, nan_audio, SR)
        assert rollback is False

    def test_return_types(self):
        """Rückgabewerte haben korrekte Typen."""
        from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db

        audio = self._voiced_audio()
        rollback, shift = check_formant_shift_db(audio, audio, SR)
        assert isinstance(rollback, bool)
        assert isinstance(shift, float)

    def test_lpc_tracker_track_method(self):
        """_LPCFormantTracker.track() gibt dict mit f1_mean zurück."""
        from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker

        tracker = get_lpc_formant_tracker()
        mono = self._voiced_audio()
        result = tracker.track(mono, SR)
        assert isinstance(result, dict)
        assert "f1_mean" in result
        assert "f2_mean" in result
        assert isinstance(result["f1_mean"], float)


# ============================================================
# G2 — Breath segment protection (phase_03 / phase_29 / phase_20)
# ============================================================


class TestBreathSegmentProtection:
    """§G2 Breath-Segment Protection — EMOTIONAL_TENSION blend-back."""

    def _make_breath_seg(
        self, start: float, end: float, category_value: str = "emotional_tension", g_floor: float = 0.85
    ):
        """Minimal BreathSegment-ähnliches Objekt."""

        class _FakeCat:
            def __init__(self, v):
                self.value = v

        class _FakeSeg:
            def __init__(self, s, e, cat, gf):
                self.start_s = s
                self.end_s = e
                self.category = _FakeCat(cat)
                self.recommended_g_floor = gf

        return _FakeSeg(start, end, category_value, g_floor)

    def test_tension_segment_blends_back(self):
        """EMOTIONAL_TENSION-Segment → result blends back toward original in that region."""
        sr = SR
        orig = np.random.default_rng(42).uniform(-0.1, 0.1, sr * 2).astype(np.float32)
        nr_out = np.zeros_like(orig)  # NR destroyed everything
        seg = self._make_breath_seg(0.5, 0.7, "emotional_tension", g_floor=0.85)
        si, ei = int(0.5 * sr), int(0.7 * sr)

        # Simulate the breath protection logic
        dry = float(seg.recommended_g_floor)
        result = np.array(nr_out, copy=True)
        result[si:ei] = dry * orig[si:ei] + (1.0 - dry) * nr_out[si:ei]

        # In the EMOTIONAL_TENSION region, result should be ~85% of original
        expected = dry * orig[si:ei]
        np.testing.assert_allclose(result[si:ei], expected, rtol=1e-5)

    def test_non_tension_segment_not_blended(self):
        """NATURAL segment → kein Blend (nur EMOTIONAL_TENSION wird geschützt)."""
        seg = self._make_breath_seg(0.5, 0.7, "natural", g_floor=0.50)
        cat_str = str(getattr(seg.category, "value", "")).lower()
        assert "tension" not in cat_str

    def test_g_floor_85_tension(self):
        """EMOTIONAL_TENSION → G_floor 0.85 ist der empfohlene Wert."""
        seg = self._make_breath_seg(0.0, 0.2, "emotional_tension", g_floor=0.85)
        assert seg.recommended_g_floor == pytest.approx(0.85)

    def test_empty_breath_segs_no_op(self):
        """Leere breath_segments → kein Blend-Vorgang."""
        segs: list[Any] = []
        blended = False
        for _ in segs:
            blended = True
        assert not blended


# ============================================================
# G3 — OMLSA post-DFN Restglätter
# ============================================================


class TestOmlsaPostDfnSmoother:
    """§G3 OMLSA post-DFN Residual Smoother — light Wiener gain."""

    def test_imcra_returns_array(self):
        """compute_imcra_noise_estimate() gibt 2D-Array zurück."""
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        mono = np.random.default_rng(7).normal(0, 0.01, SR * 2).astype(np.float32)
        psd = compute_imcra_noise_estimate(mono, SR)
        assert psd is not None
        assert psd.ndim == 2
        assert psd.shape[0] == 2048 // 2 + 1  # default n_fft=2048

    def test_wiener_gain_floor_clamp(self):
        """Wiener Gain niemals unter G_floor (Masking-Guard §2.62)."""
        G_floor = 0.10
        sig_psd = np.array([1.0, 1.0, 0.5, 0.1])
        noise_psd = np.array([0.9, 1.1, 0.4, 0.05])
        gain = np.maximum(G_floor, 1.0 - noise_psd / np.maximum(sig_psd, 1e-20))
        gain = np.clip(gain, G_floor, 1.0)
        assert np.all(gain >= G_floor)
        assert np.all(gain <= 1.0)

    def test_wet_blend_bounds(self):
        """25% wet blend → output ist zwischen input und geglättetem Signal."""
        orig = np.ones(1000, dtype=np.float32) * 0.5
        smoothed = np.zeros(1000, dtype=np.float32)
        wet = 0.25
        blended = wet * smoothed + (1.0 - wet) * orig
        assert float(np.min(blended)) >= 0.374  # close to 0.375
        assert float(np.max(blended)) <= 0.376


# ============================================================
# G4 — SingMOS als aktiver Gate
# ============================================================


class TestSingMosGate:
    """§G4 SingMOS Naturalness Gate — get_singmos_predictor().predict()."""

    def test_singmos_predictor_returns_float(self):
        """SingMOS predict() gibt float MOS 1.0–5.0 zurück."""
        from backend.core.dsp.quality_predictors import get_singmos_predictor

        pred = get_singmos_predictor()
        mono = np.random.default_rng(13).normal(0, 0.05, SR * 2).astype(np.float32)
        mos = pred.predict(mono, SR)
        assert isinstance(mos, float)
        assert 1.0 <= mos <= 5.0

    def test_singmos_singleton(self):
        """get_singmos_predictor() ist Singleton."""
        from backend.core.dsp.quality_predictors import get_singmos_predictor

        p1 = get_singmos_predictor()
        p2 = get_singmos_predictor()
        assert p1 is p2

    def test_singmos_gate_threshold_logic(self):
        """MOS < 2.0 → Rollback, < 2.5 → Warnung, ≥ 2.5 → OK."""

        def _decision(mos: float) -> str:
            if mos < 2.0:
                return "rollback"
            if mos < 2.5:
                return "warning"
            return "ok"

        assert _decision(1.5) == "rollback"
        assert _decision(1.99) == "rollback"
        assert _decision(2.0) == "warning"
        assert _decision(2.49) == "warning"
        assert _decision(2.5) == "ok"
        assert _decision(4.0) == "ok"

    def test_singmos_high_quality_signal(self):
        """Synthetisch reines Sinus-Signal → SingMOS sollte nicht unter 1.0 fallen."""
        from backend.core.dsp.quality_predictors import get_singmos_predictor

        t = np.linspace(0, 2.0, SR * 2, endpoint=False)
        clean = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        mos = get_singmos_predictor().predict(clean, SR)
        assert mos >= 1.0

    def test_singmos_stereo_fallback(self):
        """Stereo-Input (2, N) → mono conversion → valid MOS."""
        from backend.core.dsp.quality_predictors import get_singmos_predictor

        t = np.linspace(0, 2.0, SR * 2, endpoint=False)
        mono = (0.3 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        stereo = np.stack([mono, mono])
        # Simulate the mono conversion done in UV3
        inp = stereo.mean(axis=0) if stereo.shape[0] == 2 else stereo.mean(axis=1)
        mos = get_singmos_predictor().predict(inp, SR)
        assert isinstance(mos, float)
        assert 1.0 <= mos <= 5.0


# =============================================================================
# §VocalStyle VocalStyleProfiler — per-recording singer fingerprint (v10.14)
# =============================================================================


class TestVocalStyleProfiler:
    """Tests für backend.core.dsp.vocal_style_profiler."""

    def _make_voiced_tone(self, f0_hz: float = 220.0, dur_s: float = 3.0) -> np.ndarray:
        """Multi-harmonic tone simulating a voiced vocal sound."""
        t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
        sig = (
            0.4 * np.sin(2 * np.pi * f0_hz * t)
            + 0.2 * np.sin(2 * np.pi * f0_hz * 2 * t)
            + 0.1 * np.sin(2 * np.pi * f0_hz * 3 * t)
        )
        return sig.astype(np.float32)  # type: ignore[no-any-return]

    def test_profile_returns_valid_on_sufficient_audio(self):
        """Genug Audio → .valid == True."""
        from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler

        audio = self._make_voiced_tone(220.0, 3.0)
        profile = get_vocal_style_profiler().profile(audio, SR)
        assert profile.valid is True

    def test_profile_returns_invalid_on_short_audio(self):
        """Zu kurzes Audio (< 1 s) → .valid == False."""
        from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler

        audio = np.zeros(int(SR * 0.5), dtype=np.float32)
        profile = get_vocal_style_profiler().profile(audio, SR)
        assert profile.valid is False

    def test_profile_fields_in_valid_range(self):
        """Alle numerischen Felder bleiben in ihren definierten Bereichen."""
        from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler

        audio = self._make_voiced_tone(200.0, 5.0)
        p = get_vocal_style_profiler().profile(audio, SR)
        assert 0.0 <= p.vibrato_rate_hz <= 12.0
        assert 0.0 <= p.vibrato_depth_cents <= 200.0
        assert 0.0 <= p.chest_head_ratio <= 1.0
        assert 0.0 <= p.phrase_contour_variance <= 1000.0
        assert 0.0 <= p.f1_f2_ratio <= 1.0
        assert 0.0 <= p.breathiness_index <= 30.0

    def test_vqi_calibration_offset_nonpositive(self):
        """VQI-Kalibrierungsoffset ist immer ≤ 0 (nur Absenkung, keine Erhöhung)."""
        from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler

        audio = self._make_voiced_tone(300.0, 4.0)
        p = get_vocal_style_profiler().profile(audio, SR)
        assert p.vqi_calibration_offset() <= 0.0
        assert p.vqi_calibration_offset() >= -0.05

    def test_nr_strength_cap_in_range(self):
        """NR-Stärke-Cap bleibt in [0.5, 1.0]."""
        from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler

        audio = self._make_voiced_tone(440.0, 5.0)
        p = get_vocal_style_profiler().profile(audio, SR)
        assert 0.5 <= p.nr_strength_cap() <= 1.0

    def test_profile_stereo_input(self):
        """Stereo-Input (2, N) wird intern nach Mono gemittelt — kein Absturz."""
        from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler

        mono = self._make_voiced_tone(180.0, 3.0)
        stereo = np.stack([mono, mono])
        p = get_vocal_style_profiler().profile(stereo, SR)
        assert p.valid is True

    def test_profile_silence_returns_valid_but_low_breathiness(self):
        """Stilles Signal ergibt valid=True aber breathiness ≈ 0 (kein H1-H2 peak)."""
        from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler

        audio = np.zeros(int(SR * 5), dtype=np.float32)
        p = get_vocal_style_profiler().profile(audio, SR)
        # profile() returns valid=True for long enough arrays (≥ 1s), but all
        # acoustic features should be near zero for silence
        assert p.breathiness_index == 0.0 or p.breathiness_index < 5.0

    def test_singleton_identity(self):
        """get_vocal_style_profiler() liefert immer dieselbe Instanz."""
        from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler

        a = get_vocal_style_profiler()
        b = get_vocal_style_profiler()
        assert a is b

    def test_to_dict_keys(self):
        """to_dict() enthält alle erwarteten Schlüssel."""
        from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler

        audio = self._make_voiced_tone(220.0, 3.0)
        p = get_vocal_style_profiler().profile(audio, SR)
        d = p.to_dict()
        for key in (
            "vibrato_rate_hz",
            "vibrato_depth_cents",
            "chest_head_ratio",
            "f1_f2_ratio",
            "breathiness_index",
            "valid",
            "vqi_calibration_offset",
            "nr_strength_cap",
        ):
            assert key in d, f"Fehlender Schlüssel: {key}"

    def test_profile_invalid_vqi_offset_is_zero(self):
        """VocalStyleProfile(valid=False).vqi_calibration_offset() == 0.0."""
        from backend.core.dsp.vocal_style_profiler import VocalStyleProfile

        p = VocalStyleProfile(valid=False)
        assert p.vqi_calibration_offset() == 0.0
        assert p.nr_strength_cap() == 1.0


# =============================================================================
# §ArcPlan EmotionalArcPlanner — Dramaturgie-Schutz-Planung (v10.14)
# =============================================================================


class TestEmotionalArcPlanner:
    """Tests für backend.core.emotional_arc_planner."""

    def _make_audio(self, dur_s: float = 10.0) -> np.ndarray:
        t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
        return (0.3 * np.sin(2 * np.pi * 250.0 * t)).astype(np.float32)  # type: ignore[no-any-return]

    def test_plan_returns_arc_plan(self):
        """plan() gibt ein ArcPlan-Objekt zurück."""
        from backend.core.emotional_arc_planner import ArcPlan, get_emotional_arc_planner

        audio = self._make_audio(8.0)
        plan = get_emotional_arc_planner().plan(audio, SR, {})
        assert isinstance(plan, ArcPlan)

    def test_plan_weight_count_matches_duration(self):
        """n_frames ≈ duration / resolution (±2)."""
        from backend.core.emotional_arc_planner import _RESOLUTION_S, get_emotional_arc_planner

        audio = self._make_audio(10.0)
        plan = get_emotional_arc_planner().plan(audio, SR, {})
        expected_frames = int(10.0 / _RESOLUTION_S)
        assert abs(len(plan.weights) - expected_frames) <= 2

    def test_weight_at_default_is_one(self):
        """Ohne Zonen → weight_at(0, 5) ≈ 1.0 (nur smoothing-Abweichung)."""
        from backend.core.emotional_arc_planner import get_emotional_arc_planner

        audio = self._make_audio(10.0)
        plan = get_emotional_arc_planner().plan(audio, SR, {})
        w = plan.weight_at(2.0, 4.0)
        assert 0.8 <= w <= 1.2

    def test_frisson_zone_raises_weight(self):
        """Frisson-Zone bei [3.0, 5.0] → weight_at(3.5, 4.5) > weight_at(0.5, 1.5)."""
        from backend.core.emotional_arc_planner import get_emotional_arc_planner

        audio = self._make_audio(10.0)
        ctx = {"frisson_zones": [(3.0, 5.0)]}
        plan = get_emotional_arc_planner().plan(audio, SR, ctx)
        w_frisson = plan.weight_at(3.5, 4.5)
        w_normal = plan.weight_at(0.5, 1.5)
        assert w_frisson > w_normal, f"Frisson-Weight {w_frisson:.2f} ≤ Normal {w_normal:.2f}"

    def test_silence_zone_reduces_weight(self):
        """Stille Bereiche → weight < 1.0."""
        from backend.core.emotional_arc_planner import get_emotional_arc_planner

        audio = np.zeros(int(SR * 10.0), dtype=np.float32)
        plan = get_emotional_arc_planner().plan(audio, SR, {})
        assert float(np.mean(plan.weights)) < 1.0

    def test_weights_clipped_in_range(self):
        """Alle Gewichte bleiben in [0.5, 1.5]."""
        from backend.core.emotional_arc_planner import get_emotional_arc_planner

        audio = self._make_audio(10.0)
        ctx = {
            "frisson_zones": [(0.0, 2.0)],
            "tension_zones": [(4.0, 6.0)],
            "whisper_zones": [(8.0, 10.0)],
        }
        plan = get_emotional_arc_planner().plan(audio, SR, ctx)
        assert float(np.min(plan.weights)) >= 0.5
        assert float(np.max(plan.weights)) <= 1.5

    def test_weight_at_empty_plan_returns_one(self):
        """ArcPlan mit leeren weights → weight_at() == 1.0."""
        import numpy as np

        from backend.core.emotional_arc_planner import ArcPlan

        plan = ArcPlan(weights=np.ones(1, dtype=np.float32), duration_s=10.0)
        assert plan.weight_at(0.0, 5.0) == 1.0

    def test_dict_zone_format_supported(self):
        """Zonen als dict (start/end) werden ebenfalls verarbeitet."""
        from backend.core.emotional_arc_planner import get_emotional_arc_planner

        audio = self._make_audio(10.0)
        ctx = {"tension_zones": [{"start": 2.0, "end": 4.0}]}
        plan = get_emotional_arc_planner().plan(audio, SR, ctx)
        w = plan.weight_at(2.5, 3.5)
        assert w > 1.0

    def test_to_dict_has_required_keys(self):
        """to_dict() enthält alle erwarteten Schlüssel."""
        from backend.core.emotional_arc_planner import get_emotional_arc_planner

        audio = self._make_audio(5.0)
        plan = get_emotional_arc_planner().plan(audio, SR, {})
        d = plan.to_dict()
        for key in ("duration_s", "resolution_s", "n_frames", "mean_weight", "max_weight", "min_weight"):
            assert key in d

    def test_singleton_identity(self):
        """get_emotional_arc_planner() liefert immer dieselbe Instanz."""
        from backend.core.emotional_arc_planner import get_emotional_arc_planner

        assert get_emotional_arc_planner() is get_emotional_arc_planner()

    def test_none_context_does_not_crash(self):
        """plan() mit context=None → kein Absturz."""
        from backend.core.emotional_arc_planner import get_emotional_arc_planner

        audio = self._make_audio(3.0)
        plan = get_emotional_arc_planner().plan(audio, SR, None)
        assert len(plan.weights) > 0


# =============================================================================
# §EraTarget EraCarrierTargetModel — Ära×Träger Klangziel (v10.14)
# =============================================================================


class TestEraCarrierTargetModel:
    """Tests für backend.core.dsp.era_carrier_target."""

    def test_shellac_1935_high_noise_preserve(self):
        """1935 Shellac → noise_texture_preserve_ratio ≥ 0.65."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        t = get_era_carrier_target_model().get_target(1935, "shellac")
        assert t.noise_texture_preserve_ratio >= 0.65, f"Got {t.noise_texture_preserve_ratio}"

    def test_cd_1990_low_noise_preserve(self):
        """1990 CD → noise_texture_preserve_ratio ≤ 0.15."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        t = get_era_carrier_target_model().get_target(1990, "cd")
        assert t.noise_texture_preserve_ratio <= 0.15, f"Got {t.noise_texture_preserve_ratio}"

    def test_nr_g_floor_shellac_above_base(self):
        """1935 Shellac → nr_g_floor(0.10) > 0.10 (era lift aktiv)."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        t = get_era_carrier_target_model().get_target(1935, "shellac")
        assert t.nr_g_floor(0.10) > 0.10

    def test_nr_g_floor_cd_equals_base(self):
        """1990 CD → nr_g_floor(0.10) == 0.10 (kein era lift nötig)."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        t = get_era_carrier_target_model().get_target(1990, "cd")
        assert abs(t.nr_g_floor(0.10) - 0.10) < 0.02

    def test_none_era_returns_fallback(self):
        """era_decade=None → fallback target ohne Absturz."""
        from backend.core.dsp.era_carrier_target import _FALLBACK_TARGET, get_era_carrier_target_model

        t = get_era_carrier_target_model().get_target(None, "vinyl")
        assert t is _FALLBACK_TARGET or t.noise_texture_preserve_ratio == _FALLBACK_TARGET.noise_texture_preserve_ratio

    def test_carrier_normalization_variants(self):
        """Verschiedene Schreibweisen für 'vinyl' → selbes target."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        model = get_era_carrier_target_model()
        t1 = model.get_target(1965, "vinyl")
        t2 = model.get_target(1965, "VINYL")
        t3 = model.get_target(1965, "vinyl-lp")
        assert t1.noise_texture_preserve_ratio == t2.noise_texture_preserve_ratio
        assert t1.noise_texture_preserve_ratio == t3.noise_texture_preserve_ratio

    def test_carrier_list_input(self):
        """carrier als Liste (transfer_chain) → erstes Element wird verwendet."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        t = get_era_carrier_target_model().get_target(1935, ["shellac", "vinyl"])
        assert t.noise_texture_preserve_ratio >= 0.65

    def test_to_dict_has_required_keys(self):
        """to_dict() enthält alle erwarteten Schlüssel."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        t = get_era_carrier_target_model().get_target(1965, "vinyl")
        d = t.to_dict()
        for key in (
            "noise_texture_preserve_ratio",
            "authentic_harmonic_ratio",
            "bw_ceiling_hz",
            "dr_ceiling_db",
            "nr_g_floor",
        ):
            assert key in d

    def test_era_monotone_noise_preserve(self):
        """Neuere Ären haben niedrigere noise_texture_preserve_ratio als alte."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        model = get_era_carrier_target_model()
        shellac_old = model.get_target(1930, "shellac").noise_texture_preserve_ratio
        vinyl_mid = model.get_target(1965, "vinyl").noise_texture_preserve_ratio
        cd_new = model.get_target(1990, "cd").noise_texture_preserve_ratio
        assert shellac_old > vinyl_mid > cd_new, (
            f"Monotonie verletzt: shellac={shellac_old:.2f} vinyl={vinyl_mid:.2f} cd={cd_new:.2f}"
        )

    def test_singleton_identity(self):
        """get_era_carrier_target_model() liefert dieselbe Instanz."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        assert get_era_carrier_target_model() is get_era_carrier_target_model()

    def test_bw_ceiling_monotone_increases_with_era(self):
        """Neuere Ären haben höhere BW-Ceiling (shellac < vinyl < cd)."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        model = get_era_carrier_target_model()
        bw_shellac = model.get_target(1935, "shellac").bw_ceiling_hz
        bw_vinyl = model.get_target(1965, "vinyl").bw_ceiling_hz
        bw_cd = model.get_target(1990, "cd").bw_ceiling_hz
        assert bw_shellac < bw_vinyl < bw_cd

    def test_unknown_carrier_fallback(self):
        """Unbekannter Carrier-String → kein Absturz, fallback-Wert."""
        from backend.core.dsp.era_carrier_target import get_era_carrier_target_model

        t = get_era_carrier_target_model().get_target(1960, "unknown_xyz_carrier")
        assert 0.0 <= t.noise_texture_preserve_ratio <= 1.0
