import pytest

"""
tests/unit/test_iad_musical_residuum.py — IntroducedArtifactDetector §2.23 Test-Suite
§2.46e Relative-Harmonicity-Guard (v10.0.0) — artifact_freedom = 0.95 adhesion fix.

Alle Tests synthetisch, kein ML-Modell erforderlich.
"""

import numpy as np

SR = 48_000


def _harmonic_signal(dur: float = 6.0, amp: float = 0.1, f0: float = 220.0) -> np.ndarray:
    """Synthethisches Sinus-Fundamental + 4 Obertöne (harmonisches Vokal-Proxy)."""
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    sig = sum((amp / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 5))
    return np.clip(sig, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _noise(dur: float = 6.0, amp: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amp * rng.standard_normal(int(dur * SR))).astype(np.float32)


# ---------------------------------------------------------------------------
# Import + Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.introduced_artifact_detector import (
        IntroducedArtifactDetector,
        detect_introduced_artifacts,
        get_iad,
    )

    assert IntroducedArtifactDetector is not None
    assert detect_introduced_artifacts is not None
    assert get_iad is not None


def test_01_singleton():
    from backend.core.introduced_artifact_detector import get_iad

    a = get_iad()
    b = get_iad()
    assert a is b


# ---------------------------------------------------------------------------
# §2.46e Relative-Harmonicity-Guard — Kern-Fix
# ---------------------------------------------------------------------------


def test_02_vocal_denoise_delta_not_flagged():
    """Kleine harmonische Restorierungs-Deltas auf vokalen Abschnitten dürfen NICHT
    als ML-Halluzination geflaged werden (artifact_freedom = 0.95 Adhesion-Fix).

    Scenario: NR entfernt leises Rauschen (-40 dBFS) und modifiziert minimal die
    Vokal-Harmonik (0.5 % des Signals). Residuum ist harmonisch (erbt Vokal-Struktur),
    aber NICHT mehr harmonisch als das Original → kein Flag.
    """
    from backend.core.introduced_artifact_detector import IntroducedArtifactDetector

    iad = IntroducedArtifactDetector()
    dur = 6.0
    original = _harmonic_signal(dur=dur, amp=0.10, f0=220.0) + _noise(dur=dur, amp=0.015)
    # Restored: noise removed + minimal harmonic modification (0.5 %)
    restored = _harmonic_signal(dur=dur, amp=0.10 * 0.995, f0=220.0)

    residuum = np.nan_to_num(restored[: len(original)] - original[: len(original)])
    orig_mono = original[: len(residuum)]
    results = iad._detect_ml_hallucinations(orig_mono, residuum, SR)
    assert len(results) == 0, f"Legitimate vocal restoration delta falsely flagged as ML hallucination: {results}"


def test_03_bw_extension_residuum_not_flagged():
    """Bandwidth-Extension-Residuum (neue HF-Harmonik, niedrige Amplitude) darf NICHT
    als Halluzination geflaged werden, wenn Energie < HALLUCINATION_MIN_RMS.

    Scenario: BW extension fügt HF-Harmonik bei ~-40 dBFS hinzu.
    """
    from backend.core.introduced_artifact_detector import IntroducedArtifactDetector

    iad = IntroducedArtifactDetector()
    dur = 6.0
    original = _harmonic_signal(dur=dur, amp=0.10, f0=110.0)
    # Residuum = added HF harmonics at very low amplitude (< HALLUCINATION_MIN_RMS = 0.032)
    hf_residuum = _harmonic_signal(dur=dur, amp=0.008, f0=880.0)  # -42 dBFS
    residuum = hf_residuum[: len(original)]
    orig_mono = original[: len(residuum)]

    results = iad._detect_ml_hallucinations(orig_mono, residuum, SR)
    assert len(results) == 0, f"Low-amplitude BW-extension falsely flagged: {results}"


def test_04_true_hallucination_flagged():
    """Echte ML-Halluzination: Residuum ist stark harmonisch in einer Sektion,
    wo das Original NICHT harmonisch ist (Rauschbereich). MUSS geflaged werden.

    Scenario: Original = Rauschen (h_orig ~0.05), Residuum = starkes harmonisches
    Signal (h_res ~0.80). Residuum >> h_orig + 0.20 → True Hallucination.
    """
    from backend.core.introduced_artifact_detector import IntroducedArtifactDetector

    iad = IntroducedArtifactDetector()
    dur = 6.0
    original = _noise(dur=dur, amp=0.05)  # broadband noise, low harmonicity
    # Residuum: strong harmonic signal (hallucinated) at high amplitude
    residuum = _harmonic_signal(dur=dur, amp=0.10, f0=220.0)
    orig_mono = original[: len(residuum)]

    results = iad._detect_ml_hallucinations(orig_mono, residuum, SR)
    assert len(results) > 0, "True ML hallucination (harmonic residuum on noise original) was NOT flagged"


def test_05_detect_convenience_wrapper_vocal_no_hallucination():
    """detect_introduced_artifacts() auf harmonsicher Restaurierung:
    Fraction muss << 0.05 bleiben → kein 0.95-Adhesion-Bug.
    """
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    dur = 10.0
    original = _harmonic_signal(dur=dur, amp=0.10, f0=220.0) + _noise(dur=dur, amp=0.015)
    # Mild restoration: slight denoise + minimal harmonic modification
    restored = _harmonic_signal(dur=dur, amp=0.10 * 0.998, f0=220.0) + _noise(dur=dur, amp=0.002)
    restored = restored[: len(original)].astype(np.float32)

    result = detect_introduced_artifacts(original[: len(restored)].astype(np.float32), restored, SR)
    # After fix: fraction should be very small (no false hallucination flags)
    assert result.total_contaminated_fraction < 0.04, (
        f"Vocal restoration incorrectly produces high contamination fraction: "
        f"{result.total_contaminated_fraction:.4f} (adhesion bug regression)"
    )


def test_06_iad_result_penalty_above_095():
    """IAD-Penalty für saubere Restaurierung muss > 0.95 sein.
    Testet die direkte Ursache des artifact_freedom = 0.95 Adhesion-Bugs.
    """
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    dur = 10.0
    original = _harmonic_signal(dur=dur, amp=0.10, f0=220.0) + _noise(dur=dur, amp=0.015)
    restored = _harmonic_signal(dur=dur, amp=0.10 * 0.998, f0=220.0) + _noise(dur=dur, amp=0.002)
    restored = restored[: len(original)].astype(np.float32)

    result = detect_introduced_artifacts(original[: len(restored)].astype(np.float32), restored, SR)
    penalty = 1.0 - result.total_contaminated_fraction
    assert penalty > 0.95, (
        f"IAD penalty should be > 0.95 for legitimate vocal restoration, got {penalty:.4f} "
        f"(fraction={result.total_contaminated_fraction:.4f})"
    )


def test_07_relative_margin_constant():
    """HALLUCINATION_RELATIVE_MARGIN muss 0.20 sein (§2.46e v10.0.0)."""
    from backend.core.introduced_artifact_detector import IntroducedArtifactDetector

    iad = IntroducedArtifactDetector()
    assert iad.HALLUCINATION_RELATIVE_MARGIN == 0.20


def test_08_harmonicity_method_vocal():
    """_harmonicity() muss für ein vokal-artiges Signal > HARMONICITY_THRESHOLD liefern."""
    from backend.core.introduced_artifact_detector import IntroducedArtifactDetector

    iad = IntroducedArtifactDetector()
    frame = _harmonic_signal(dur=2.0, amp=0.1, f0=220.0)
    if len(frame) >= 8192 and SR >= 32000:
        frame_eval = frame[::2]
        sr_eval = SR // 2
    else:
        frame_eval = frame
        sr_eval = SR
    h = iad._harmonicity(frame_eval, sr_eval)
    assert h > 0.60, f"Expected vocal harmonicity > 0.60, got {h:.4f}"


def test_09_harmonicity_method_noise():
    """_harmonicity() muss für weißes Rauschen < 0.30 liefern."""
    from backend.core.introduced_artifact_detector import IntroducedArtifactDetector

    iad = IntroducedArtifactDetector()
    rng = np.random.default_rng(99)
    frame = (0.05 * rng.standard_normal(int(2.0 * SR))).astype(np.float32)
    if len(frame) >= 8192 and SR >= 32000:
        frame_eval = frame[::2]
        sr_eval = SR // 2
    else:
        frame_eval = frame
        sr_eval = SR
    h = iad._harmonicity(frame_eval, sr_eval)
    assert h < 0.30, f"Expected noise harmonicity < 0.30, got {h:.4f}"


def test_10_silence_no_artifacts():
    """Stille → keine Artefakte."""
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    silence = np.zeros(int(6.0 * SR), dtype=np.float32)
    result = detect_introduced_artifacts(silence, silence, SR)
    assert not result.has_artifacts
    assert result.total_contaminated_fraction == 0.0


def test_11_identical_signals_no_artifacts():
    """Identische Signale → Residuum = 0 → keine Artefakte."""
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    sig = _harmonic_signal(dur=6.0, amp=0.10, f0=300.0).astype(np.float32)
    result = detect_introduced_artifacts(sig, sig, SR)
    # Identical signals produce zero residuum → fraction must be 0
    assert result.total_contaminated_fraction == 0.0
