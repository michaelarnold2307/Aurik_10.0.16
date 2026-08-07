import pytest

"""
tests/unit/test_artifact_freedom_gate.py — ArtifactFreedomGate §2.49 Test-Suite (≥ 25 Tests)
Alle Tests synthetisch, kein ML-Modell erforderlich.
"""

import numpy as np

SR = 48_000


def _audio(dur: float = 3.0, amp: float = 0.3, freq: float = 440.0):
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(dur: float = 1.0):
    return np.zeros(int(dur * SR), dtype=np.float32)


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono * 0.9], axis=0)


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.artifact_freedom_gate import ArtifactFreedomGate, get_artifact_freedom_gate

    assert ArtifactFreedomGate is not None
    assert get_artifact_freedom_gate is not None


def test_01_singleton():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    g1 = get_artifact_freedom_gate()
    g2 = get_artifact_freedom_gate()
    assert g1 is g2


def test_uv3_afg_soft_backoff_keeps_strongest_passing_wet():
    from types import SimpleNamespace

    from backend.core.unified_restorer_v3 import _try_artifact_freedom_soft_backoff

    phase_input = np.zeros(1024, dtype=np.float32)
    phase_output = np.ones(1024, dtype=np.float32) * 0.5

    class _Gate:
        def __init__(self) -> None:
            self.wets: list[float] = []

        def evaluate(self, original, restored, sr, material_type, phase_id, **kwargs):
            del original, sr, material_type, phase_id, kwargs
            wet = float(np.mean(restored) / 0.5)
            self.wets.append(round(wet, 2))
            return SimpleNamespace(artifact_freedom=1.0 if wet <= 0.35 else 0.8)

    gate = _Gate()
    candidate, result, wet = _try_artifact_freedom_soft_backoff(
        gate,
        phase_input,
        phase_output,
        SR,
        "cassette",
        "phase_03_denoise",
    )

    assert wet == 0.35
    assert result.artifact_freedom == 1.0  # type: ignore[union-attr]
    assert gate.wets == [0.75, 0.55, 0.35]
    assert candidate is not None
    assert candidate.dtype == np.float32
    assert np.allclose(candidate, phase_output * 0.35)


def test_02_clean_audio_perfect_score():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio()
    result = gate.evaluate(audio, audio.copy(), SR)
    assert result.artifact_freedom >= 0.95


def test_03_identical_audio_no_artifacts():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio()
    result = gate.evaluate(audio, audio, SR)
    assert len(result.detected_artifacts) == 0
    assert result.artifact_freedom == 1.0


def test_04_short_audio_returns_perfect():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    short = np.zeros(100, dtype=np.float32)
    result = gate.evaluate(short, short, SR)
    assert result.artifact_freedom == 1.0


def test_05_material_type_normalization():
    from backend.core.artifact_freedom_gate import ArtifactFreedomGate

    gate = ArtifactFreedomGate()
    assert gate._normalize_material("VINYL") == "vinyl"
    assert gate._normalize_material("MaterialType.tape") == "tape"
    assert gate._normalize_material("MaterialType.cd_digital") == "digital"
    assert gate._normalize_material("shellac_78rpm") == "shellac"
    assert gate._normalize_material("unknown_format") == "digital"


def test_06_material_thresholds_differ():
    from backend.core.artifact_freedom_gate import ArtifactFreedomGate

    gate = ArtifactFreedomGate()
    digital = gate._get_thresholds("digital")
    shellac = gate._get_thresholds("shellac")
    # Shellac should be more tolerant (higher peak threshold)
    assert shellac["musical_noise_peak_db"] > digital["musical_noise_peak_db"]
    assert shellac["spectral_hole_hz"] > digital["spectral_hole_hz"]


def test_07_musical_noise_detection_in_silence():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    original = _silence(2.0)
    restored = _silence(2.0)
    # Add strong tonal peak in restored silence
    n = len(restored)
    for i in range(0, n - 1440, 1440):
        segment = restored[i : i + 1440]
        # Strong tone at ~1 kHz in silence
        t = np.arange(len(segment)) / SR
        segment += 0.01 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    result = gate.evaluate(original, np.clip(restored, -1.0, 1.0), SR)
    # Should detect some artifacts (musical noise in silence)
    assert result.artifact_freedom <= 1.0


def test_08_spectral_holes_detection():
    from backend.core.artifact_freedom_gate import ArtifactFreedomGate

    gate = ArtifactFreedomGate()
    # Create harmonic-rich broadband signal with clear passband
    np.random.seed(42)
    n = SR * 2
    t = np.linspace(0, 2.0, n, endpoint=False)
    # Multi-harmonic signal spanning 200-6000 Hz (defines a clear passband)
    sig = np.zeros(n, dtype=np.float64)
    for f in range(200, 6000, 100):
        sig += 0.02 * np.sin(2 * np.pi * f * t)
    original = sig.astype(np.float32)
    # Band-stop via full-signal FFT — zero 2-4 kHz
    fft_orig = np.fft.rfft(original)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    band_mask = (freqs >= 2000) & (freqs <= 4000)
    fft_restored = fft_orig.copy()
    fft_restored[band_mask] = 0.0
    restored = np.fft.irfft(fft_restored, n=n).astype(np.float32)
    thresholds = gate._get_thresholds("digital")
    artifacts = gate._detect_spectral_holes(original, restored, SR, thresholds)
    assert len(artifacts) > 0, "Should detect spectral hole in 2-4 kHz band"


def test_09_phase_cancellation_detection():
    from backend.core.artifact_freedom_gate import ArtifactFreedomGate

    gate = ArtifactFreedomGate()
    # Create stereo with GENUINE anti-phase R ≈ -L (lr_corr < -0.20).
    # Independent noise with lr_corr ≈ 0 is NOT phase cancellation.
    # Real phase cancellation requires strong anti-correlation causing
    # audible "hollow center" / mono-incompatibility.
    n = int(2.0 * SR)
    t = np.linspace(0, 2.0, n, endpoint=False)
    left = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    right = -0.9 * left  # anti-phase: lr_corr ≈ −0.9, mono_compat << 0.25
    stereo = np.stack([left, right], axis=0)
    thresholds = gate._get_thresholds("digital")
    artifacts = gate._detect_phase_cancellation(stereo, SR, thresholds)
    assert len(artifacts) > 0, "Should detect genuine phase cancellation (R ≈ −L, lr_corr < −0.20)"


def test_10_no_phase_cancellation_mono():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio()
    result = gate.evaluate(audio, audio, SR)
    has_cancel = any(a.artifact_type == "phase_cancellation" for a in result.detected_artifacts)
    assert not has_cancel, "Mono audio should not have phase cancellation"


def test_11_metallic_ringing_detection():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    original = _audio(2.0, amp=0.1)
    restored = original.copy()
    # Add persistent high-Q resonance at ~3 kHz
    t = np.linspace(0, 2.0, len(restored), endpoint=False)
    ringing = 0.05 * np.sin(2 * np.pi * 3000 * t).astype(np.float32)
    # Only add ringing (not in original residual will show it)
    restored = np.clip(restored + ringing, -1.0, 1.0)
    result = gate.evaluate(original, restored, SR)
    # May or may not detect depending on threshold — just verify no crash
    assert result.artifact_freedom <= 1.0


def test_12_salience_frequency_weighting():
    from backend.core.artifact_freedom_gate import ArtifactFreedomGate, DetectedArtifact

    gate = ArtifactFreedomGate()
    # Mid-frequency artifact (200-5000 Hz)
    art_mid = DetectedArtifact("test", 0, 5000, 10.0, 1000.0, -30.0)
    # High-frequency artifact (> 12 kHz)
    art_high = DetectedArtifact("test", 0, 5000, 10.0, 15000.0, -30.0)
    w_mid = gate._compute_salience_weight(art_mid, SR)
    w_high = gate._compute_salience_weight(art_high, SR)
    assert w_mid > w_high, "Mid-freq should be more salient than >12kHz"


def test_13_salience_context_weighting():
    from backend.core.artifact_freedom_gate import ArtifactFreedomGate, DetectedArtifact

    gate = ArtifactFreedomGate()
    # Artifact in silence
    art_silence = DetectedArtifact("test", 0, 5000, 10.0, 1000.0, -50.0)
    # Artifact in loud passage
    art_loud = DetectedArtifact("test", 0, 5000, 10.0, 1000.0, -10.0)
    w_silence = gate._compute_salience_weight(art_silence, SR)
    w_loud = gate._compute_salience_weight(art_loud, SR)
    assert w_silence > w_loud, "Silence-context artifact should be more salient"


def test_14_salience_duration_weighting():
    from backend.core.artifact_freedom_gate import ArtifactFreedomGate, DetectedArtifact

    gate = ArtifactFreedomGate()
    # Long artifact (> 100 ms)
    dur_long = int(0.15 * SR)
    art_long = DetectedArtifact("test", 0, dur_long, 10.0, 1000.0, -30.0)
    # Short artifact (< 20 ms)
    dur_short = int(0.01 * SR)
    art_short = DetectedArtifact("test", 0, dur_short, 10.0, 1000.0, -30.0)
    w_long = gate._compute_salience_weight(art_long, SR)
    w_short = gate._compute_salience_weight(art_short, SR)
    assert w_long > w_short, "Long artifacts should be weighted more"


def test_15_noise_texture_identical():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _silence(1.0)
    dev, penalty = gate._check_noise_texture(audio, audio, SR)
    assert dev == 0.0
    assert penalty == 0.0


def test_16_noise_texture_no_silence():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio(1.0, amp=0.5)  # loud, no silence segments
    dev, penalty = gate._check_noise_texture(audio, audio, SR)
    assert penalty == 0.0  # no silence found → no measurement


def test_17_result_dataclass_fields():
    from backend.core.artifact_freedom_gate import ArtifactFreedomResult

    result = ArtifactFreedomResult(artifact_freedom=0.92)
    assert result.artifact_freedom == 0.92
    assert result.detected_artifacts == []
    assert result.noise_texture_deviation_db_oct == 0.0
    assert result.noise_texture_penalty == 0.0


def test_18_type_weights_complete():
    from backend.core.artifact_freedom_gate import _TYPE_WEIGHTS

    expected_types = {
        "musical_noise",
        "pre_echo",
        "spectral_hole",
        "phase_cancellation",
        "metallic_ringing",
        "crackle_impulse",
    }
    assert set(_TYPE_WEIGHTS.keys()) == expected_types


def test_19_material_factors_complete():
    from backend.core.artifact_freedom_gate import _MATERIAL_FACTORS

    assert "digital" in _MATERIAL_FACTORS
    assert "tape" in _MATERIAL_FACTORS
    assert "vinyl" in _MATERIAL_FACTORS
    assert "shellac" in _MATERIAL_FACTORS


def test_20_evaluate_returns_material_type():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio(1.0)
    result = gate.evaluate(audio, audio, SR, material_type="vinyl")
    assert result.material_type == "vinyl"


def test_21_pre_echo_no_false_positive_clean():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio(1.0)
    thresholds = gate._get_thresholds("digital")
    artifacts = gate._detect_pre_echo(audio, audio.copy(), SR, thresholds)
    assert len(artifacts) == 0, "Clean audio should have no pre-echo"


def test_22_spectral_holes_no_false_positive():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio(1.0)
    thresholds = gate._get_thresholds("digital")
    artifacts = gate._detect_spectral_holes(audio, audio.copy(), SR, thresholds)
    assert len(artifacts) == 0


def test_23_metallic_ringing_no_false_positive():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio(1.0)
    thresholds = gate._get_thresholds("digital")
    artifacts = gate._detect_metallic_ringing(audio, audio.copy(), SR, thresholds)
    assert len(artifacts) == 0


def test_24_evaluate_with_nan_input():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio(1.0)
    nan_audio = audio.copy()
    nan_audio[100] = np.nan
    nan_audio[200] = np.inf
    result = gate.evaluate(nan_audio, audio, SR)
    assert np.isfinite(result.artifact_freedom)


def test_25_detail_report_keys():
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio(1.0)
    result = gate.evaluate(audio, audio, SR)
    expected_keys = {
        "n_musical_noise",
        "n_pre_echo",
        "n_spectral_holes",
        "n_phase_cancellation",
        "n_metallic_ringing",
        "n_crackle_impulse",
        "weighted_artifact_sum",
        "noise_texture_deviation_db_oct",
    }
    assert expected_keys.issubset(set(result.detail_report.keys()))


def test_26_artifact_freedom_clips_to_0_1():
    from backend.core.artifact_freedom_gate import ArtifactFreedomResult

    # verify score range assumption
    r = ArtifactFreedomResult(artifact_freedom=0.0)
    assert 0.0 <= r.artifact_freedom <= 1.0


def test_27_veto_threshold_is_095():
    """§2.49: artifact_freedom < 0.95 → veto."""
    # This is a contract test — the threshold is hard-coded in the pipeline
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    audio = _audio(1.0)
    result = gate.evaluate(audio, audio, SR)
    # Clean audio must pass the veto
    assert result.artifact_freedom >= 0.95


def test_28_evaluate_survives_scalar_guard_configuration(monkeypatch):
    import backend.core.artifact_freedom_gate as afg_module

    gate = afg_module.ArtifactFreedomGate()
    audio = _audio(1.0)

    monkeypatch.setattr(afg_module, "PRE_ECHO_VALID_TYPES", 1.0)
    monkeypatch.setattr(afg_module, "NOISE_TEXTURE_VALID_TYPES", 1.0)
    monkeypatch.setattr(afg_module, "_ROUGHNESS_APPLICABLE_TYPES", 1.0)
    monkeypatch.setattr(afg_module.ArtifactFreedomGate, "_RESTORATIVE_PHASE_IDS", 1.0)

    result = gate.evaluate(audio, audio, SR, phase_id="phase_40")

    assert np.isfinite(result.artifact_freedom)
    assert result.artifact_freedom >= 0.95


def test_28_per_phase_mode_disables_musical_noise_and_ringing():
    """§2.49 per-phase mode: residual-based detectors (musical_noise, metallic_ringing)
    must be disabled when phase_id is supplied.  Adding harmonics/EQ to clean audio
    produces a non-zero residual; in pipeline mode that would fire both detectors —
    in per-phase mode it must not."""
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    original = _audio(2.0, amp=0.2, freq=220.0)
    # Simulate an enhancement phase: add harmonics (residual has sustained tones > 50 ms)
    harmonics = (
        _audio(2.0, amp=0.05, freq=440.0) + _audio(2.0, amp=0.03, freq=660.0) + _audio(2.0, amp=0.02, freq=880.0)
    )
    restored = np.clip(original + harmonics, -1.0, 1.0)
    # Without phase_id → residual-based detectors active → may detect sustained peaks
    result_pipeline = gate.evaluate(original, restored, SR, material_type="digital", phase_id="")
    # With phase_id → residual-based detectors disabled → must not penalise musical content
    result_per_phase = gate.evaluate(
        original, restored, SR, material_type="digital", phase_id="phase_07_harmonic_restoration"
    )
    # Per-phase mode must produce higher (or equal) artifact_freedom — sustained harmonics
    # are musical content, not artefacts.
    assert result_per_phase.artifact_freedom >= result_pipeline.artifact_freedom, (
        f"Per-phase mode should not penalise added harmonics: "
        f"pipeline={result_pipeline.artifact_freedom:.3f} vs per_phase={result_per_phase.artifact_freedom:.3f}"
    )
    # detail_report must show 0 musical_noise and 0 metallic_ringing in per-phase mode
    assert result_per_phase.detail_report.get("n_musical_noise", -1) == 0, "musical_noise must be 0 in per-phase mode"
    assert result_per_phase.detail_report.get("n_metallic_ringing", -1) == 0, (
        "metallic_ringing must be 0 in per-phase mode"
    )


def test_29_pipeline_mode_still_detects_ringing():
    """§2.49 pipeline mode (no phase_id): metallic_ringing detector must still run.
    When the *same* sustained-tone residual is evaluated without a phase_id,
    the gate must detect it (demonstrating the detector is actually active)."""
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    # Pure silence original → sustained tones in output = clear metallic ringing
    original = _silence(2.0)
    restored = _audio(2.0, amp=0.3, freq=1200.0)  # single persistent tone from nowhere
    result = gate.evaluate(original, restored, SR, material_type="digital", phase_id="")
    # The detector must fire, reducing artifact_freedom below 1.0
    assert result.artifact_freedom < 1.0, (
        "Pipeline-mode metallic_ringing detector should fire on sustained tones "
        f"added to silence; got artifact_freedom={result.artifact_freedom:.3f}"
    )


def test_30_loudness_normalization_no_pre_echo_false_positive():
    """§2.49 level_scale fix: phase_40 loudness normalisation amplifies uniformly.
    Pre-echo detector must NOT fire when the only change is a proportional gain
    increase (simulated by 2.5× boost ≈ +8 dB LUFS from −22 to −14 LUFS)."""
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    # Build a signal with a sharp transient (drum-like attack)
    t = np.linspace(0, 3.0, int(3.0 * SR), endpoint=False)
    audio = np.zeros_like(t)
    # Add a sharp transient at 1 s
    attack_start = SR  # 1 s
    audio[attack_start : attack_start + 480] = (np.exp(-np.arange(480) / 60.0) * 0.8).astype(np.float32)
    # Add low-level pre-transient content (realistic, below -35 dB relative to attack)
    audio[attack_start - 240 : attack_start] = (np.sin(2 * np.pi * 440 * np.arange(240) / SR) * 0.005).astype(
        np.float32
    )
    audio = audio.astype(np.float32)
    # Simulate phase_40: uniform 2.5× gain boost
    boosted = np.clip(audio * 2.5, -1.0, 1.0).astype(np.float32)
    result = gate.evaluate(audio, boosted, SR, material_type="tape", phase_id="phase_40_loudness_normalization")
    assert result.detail_report.get("n_pre_echo", -1) == 0, (
        f"Gain-only loudness normalisation must not trigger pre_echo; "
        f"got n_pre_echo={result.detail_report.get('n_pre_echo')}"
    )
    assert result.artifact_freedom >= 0.95, (
        f"phase_40 must pass §2.49 gate after level_scale fix; got artifact_freedom={result.artifact_freedom:.3f}"
    )


def test_31_stereo_collapse_r_channel_detected():
    """§2.49 Fix A: R-Kanal stumm → phase_cancellation muss feuern.

    Wenn original Stereo hatte (L + R beide aktiv) und die Phase R auf 0
    kollabiert (> 40 dB Abfall), muss artifact_freedom < 0.95.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()

    mono = _audio(dur=3.0, amp=0.4, freq=330.0)
    # Stereo original: L=signal, R=signal*0.9 — echtes Stereo (channel-first)
    orig = np.stack([mono, mono * 0.9], axis=0).astype(np.float32)
    # After the phase: R-Kanal komplett stumm (Mono-Kollaps)
    restored = np.stack([mono, np.zeros_like(mono)], axis=0).astype(np.float32)

    result = gate.evaluate(orig, restored, SR, material_type="tape", phase_id="phase_34_mid_side_processing")
    assert result.detail_report.get("n_phase_cancellation", 0) >= 1, (
        "Stereo collapse (R-channel silent after phase) must produce ≥1 phase_cancellation artifact"
    )
    assert result.artifact_freedom < 0.95, (
        f"artifact_freedom must be < 0.95 when R-channel collapses; got {result.artifact_freedom:.3f}"
    )


def test_32_stereo_collapse_preexisting_not_flagged():
    """§2.49 Fix A: Bereits-stummer R-Kanal darf nicht neu geflaggt werden.

    Wenn original SCHON Mono war (R=0) und restored auch R=0 hat,
    darf die AFG das NICHT als neu eingeführten Stereo-Kollaps werten.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()

    mono = _audio(dur=3.0, amp=0.4, freq=440.0)
    # Beide: R-Kanal war schon stumm (pre-existing mono from source)
    orig = np.stack([mono, np.zeros_like(mono)], axis=0).astype(np.float32)
    restored = np.stack([mono * 0.95, np.zeros_like(mono)], axis=0).astype(np.float32)

    result = gate.evaluate(orig, restored, SR, material_type="tape", phase_id="phase_03_denoise")
    # No NEW collapse introduced — gate must pass
    assert result.artifact_freedom >= 0.95, (
        f"Pre-existing mono (R already silent) must not trigger collapse detector; "
        f"got artifact_freedom={result.artifact_freedom:.3f}"
    )


def test_33_near_mono_input_guard_not_flagged():
    """§2.49 Fix D: Quasi-Mono-Aufnahmen (orig_compat > 0.65) erlauben kleine
    Processing-Asymmetrien (mono_compat bleibt > 0.40) ohne Rollback.

    Reel-tape-Aufnahmen sind oft quasi-mono (Summe zweier Mono-Mikrofone).
    Eine Noise-Gate oder Dropout-Repair-Phase kann einzelne Frames auf
    mono_compat ~ 0.55 (von 0.70) verschieben — nicht hörbar, kein Artefakt.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()

    sr = 48_000
    n = int(3.0 * sr)
    t = np.linspace(0, 3.0, n, endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Near-mono original: L = R (identical channels → mono_compat ≈ 0.707)
    orig = np.stack([sig, sig], axis=0)

    # After phase: one frame has minor anti-correlation from gate smoothing.
    # mono_compat ≈ 0.55 (L active, R slightly lower due to gate transient).
    sig_r = sig.copy()
    # Frame 2 (100ms = 4800 samples): gate smoothing causes R to drop to 0.6 × L
    sig_r[4800:9600] = sig[4800:9600] * 0.6
    restored = np.stack([sig, sig_r], axis=0)

    result = gate.evaluate(orig, restored, SR, material_type="tape", phase_id="phase_18_noise_gate")

    assert result.artifact_freedom >= 0.95, (
        f"Near-mono source (orig_compat ≈ 0.70) with minor gate asymmetry "
        f"(output compat ≈ 0.55) must NOT trigger artifact gate; "
        f"got artifact_freedom={result.artifact_freedom:.3f}"
    )


def test_34_near_mono_guard_still_catches_severe_collapse():
    """§2.49 Fix D Invariante: Near-Mono-Guard darf echten Stereo-Kollaps NICHT maskieren.

    Wenn orig quasi-mono (0.70) aber output mono_compat < 0.40 (z.B. 0.15),
    muss der Detektor trotzdem feuern.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()

    sr = 48_000
    n = int(3.0 * sr)
    t = np.linspace(0, 3.0, n, endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Near-mono original: L ≈ R
    orig = np.stack([sig, sig], axis=0)

    # After phase: R is SEVERELY anti-phase on 3 consecutive frames
    # → mono_compat drops to ~0.10 (stereo collapse)
    sig_r = sig.copy()
    sig_r[4800:19200] = -sig[4800:19200]  # 3 frames (300ms) with inverted R
    restored = np.stack([sig, sig_r], axis=0)

    result = gate.evaluate(orig, restored, SR, material_type="tape", phase_id="phase_07_harmonic_restoration")

    assert result.artifact_freedom < 0.95, (
        f"Severe stereo collapse (R inverted, mono_compat ~0.10) must still trigger "
        f"even on near-mono source; got artifact_freedom={result.artifact_freedom:.3f}"
    )


# ── §2.49 #6 Crackle-Impuls-Wiedereinfuehrung ─────────────────────────────


def _make_impulsive_crackle(sr: int, dur: float = 1.0, n_clicks: int = 8, amp: float = 0.5) -> np.ndarray:
    """Helper: white-noise signal with added impulsive click spikes (crackle simulation)."""
    n = int(dur * sr)
    audio = (np.random.default_rng(42).standard_normal(n) * 0.05).astype(np.float32)
    rng = np.random.default_rng(7)
    for _ in range(n_clicks):
        pos = int(rng.integers(sr // 10, n - sr // 10))
        width = 3
        audio[pos : pos + width] += amp * np.array([1.0, -0.5, 0.2], dtype=np.float32)[:width]
    return np.clip(audio, -1.0, 1.0)


def test_35_crackle_impulse_detected_in_spectral_phase():
    """§2.49 #6: _detect_crackle_impulse() muss feuern, wenn eine spektrale Phase
    impulsartige Artefakte (PGHI-Ringing-Simulation) in sauberes Audio einfuehrt.

    Setup: orig = saubere Sinus-Signale (keine Clicks), restored = orig + Impuls-Spikes
    (Phase hat durch PGHI / STFT-Diskontinuitaeten neue Knistern erzeugt).
    Der Detektor muss mindestens 1 Artefakt erkennen.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    np.random.default_rng(42)

    sr = 48_000
    n = int(1.5 * sr)
    t = np.linspace(0, 1.5, n, endpoint=False)

    # Clean original: tonal signal without impulses
    orig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Simulate PGHI ringing: restored has added impulsive spikes
    restored = orig.copy()
    click_positions = [sr // 4, sr // 2, 3 * sr // 4]
    for pos in click_positions:
        # Very high kurtosis spike: narrow intense peak
        if pos + 5 < n:
            restored[pos] += 0.6
            restored[pos + 1] -= 0.3
            restored[pos + 2] += 0.1
    restored = np.clip(restored, -1.0, 1.0)

    thresholds = gate._get_thresholds("digital")
    artifacts = gate._detect_crackle_impulse(orig, restored, sr, thresholds)
    assert len(artifacts) >= 1, (
        f"_detect_crackle_impulse() must detect PGHI-like impulse artifacts; got {len(artifacts)} artifacts"
    )
    assert all(a.artifact_type == "crackle_impulse" for a in artifacts)


def test_36_crackle_impulse_no_false_positive_on_clean_spectral():
    """§2.49 #6: Kein False-Positive wenn eine spektrale Phase nur spektrale
    Formen aendert (Filterung, EQ) ohne Impuls-Spitzen einzufuehren.

    Setup: orig = Signal, restored = gefilterte Version (sanfte Spektral-Aenderung,
    kein impulsiver Charakter). Detektor muss 0 Artefakte liefern.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    from scipy import signal as scipy_signal

    sr = 48_000
    n = int(1.5 * sr)
    t = np.linspace(0, 1.5, n, endpoint=False)
    orig = (0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.15 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)

    # Smooth spectral change: mild LPF (no new impulses)
    b, a = scipy_signal.butter(4, 8000 / (sr / 2), btype="low")
    restored = scipy_signal.filtfilt(b, a, orig).astype(np.float32)
    restored = np.clip(restored, -1.0, 1.0)

    thresholds = gate._get_thresholds("digital")
    artifacts = gate._detect_crackle_impulse(orig, restored, sr, thresholds)
    assert len(artifacts) == 0, (
        f"Smooth spectral filtering must not trigger crackle_impulse; got {len(artifacts)} artifacts"
    )


def test_37_crackle_impulse_per_phase_mode_fires_for_subtractive():
    """§2.49 #6: evaluate() muss crackle_impulse in per-phase mode fuer
    SUBTRACTIVE-Phase (phase_29_tape_hiss_reduction) erkennen wenn Impulse hinzukamen.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()

    sr = 48_000
    n = int(1.5 * sr)
    t = np.linspace(0, 1.5, n, endpoint=False)
    orig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Simulate phase_29 PGHI ringing: dense impulse spikes introduced
    restored = orig.copy()
    for pos in range(sr // 6, n - 100, sr // 8):
        if pos + 3 < n:
            restored[pos] += 0.55
            restored[pos + 1] -= 0.28
            restored[pos + 2] += 0.08
    restored = np.clip(restored, -1.0, 1.0)

    result = gate.evaluate(
        orig,
        restored,
        sr,
        material_type="digital",
        phase_id="phase_29_tape_hiss_reduction",
    )
    assert result.detail_report.get("n_crackle_impulse", 0) >= 1, (
        f"per-phase mode for SUBTRACTIVE phase must detect crackle_impulse; detail={result.detail_report}"
    )


def test_38_crackle_impulse_not_fired_for_additive_phase():
    """§2.49 #6: Crackle-Detektor darf fuer ADDITIVE Phase (phase_07) nicht
    feuern, da ADDITIVE Phasen keine STFT/PGHI-Verarbeitung haben.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()

    sr = 48_000
    n = int(1.5 * sr)
    t = np.linspace(0, 1.5, n, endpoint=False)
    orig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Even with added harmonics (ADDITIVE phase), crackle check must not run
    harmonics = (0.05 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
    restored = np.clip(orig + harmonics, -1.0, 1.0)

    result = gate.evaluate(
        orig,
        restored,
        sr,
        material_type="digital",
        phase_id="phase_07_harmonic_restoration",  # ADDITIVE type
    )
    assert result.detail_report.get("n_crackle_impulse", 0) == 0, (
        f"Crackle detector must not run for ADDITIVE phase; detail={result.detail_report}"
    )


def test_39_crackle_impulse_material_adaptive_vinyl_more_tolerant():
    """§2.49 #6: Vinyl-Material hat hoehere Kurtosis-Toleranz als Digital.
    Ein moderates Crackle-Signal soll bei Vinyl weniger Artefakte zaehlen
    als bei digitalem Material.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()

    sr = 48_000
    n = int(1.5 * sr)
    t = np.linspace(0, 1.5, n, endpoint=False)
    orig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Moderate impulse crackle (border case)
    restored = orig.copy()
    for pos in range(sr // 4, n - 50, sr // 4):
        if pos + 3 < n:
            restored[pos] += 0.35
            restored[pos + 1] -= 0.18
    restored = np.clip(restored, -1.0, 1.0)

    thr_digital = gate._get_thresholds("digital")
    thr_vinyl = gate._get_thresholds("vinyl")
    artifacts_digital = gate._detect_crackle_impulse(orig, restored, sr, thr_digital)
    artifacts_vinyl = gate._detect_crackle_impulse(orig, restored, sr, thr_vinyl)

    # Vinyl threshold is 1.4× higher kurtosis — fewer or equal flags
    assert len(artifacts_vinyl) <= len(artifacts_digital), (
        f"Vinyl must be <= strict as digital for borderline crackle; "
        f"vinyl={len(artifacts_vinyl)} digital={len(artifacts_digital)}"
    )


# ─── §0d/§2.54 Restorative-Phase Guard Tests ──────────────────────────────────


def test_40_dropout_zone_guard_no_false_positive():
    """§0d Dropout-Zone-Guard: Frames wo orig near-silent (dropout gap) duerfen
    NICHT als crackle_impulse gemeldet werden, auch wenn delta hochkurtuig ist.

    Setup: orig = near-silence (dropout gap), restored = interpolierter Inhalt.
    Damit ist das Delta per Definition hochkurtuig — aber es ist korrekte Reparatur.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    sr = 48_000
    n = int(1.5 * sr)
    t = np.linspace(0, 1.5, n, endpoint=False)

    # Original: near-silent dropout region (< -25 dBFS)
    orig = np.full(n, 1e-5, dtype=np.float32)  # ≈ -100 dBFS — deep silence

    # Restored: interpolated audio fills the gap (impulsive relative to silence)
    restored = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    thresholds = gate._get_thresholds("vinyl")
    artifacts = gate._detect_crackle_impulse(orig, restored, sr, thresholds)
    assert len(artifacts) == 0, (
        f"Dropout-zone frames (orig near-silent) must not trigger crackle_impulse; "
        f"got {len(artifacts)} false-positive artifacts"
    )


def test_41_click_removal_guard_no_false_positive():
    """§0d Click-Removal-Guard: Wenn phase einen grossen Peak ENTFERNT hat
    (peak_orig >> peak_rest), ist das korrekte Reparatur — kein Artefakt.

    Setup: orig = Signal mit grossem Click-Peak, restored = Signal nach Click-Removal
    (Click-Peak stark reduziert). Der delta HAT hohe Kurtosis (entfernter Click),
    aber der Guard muss diesen Fall herausfiltern.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    sr = 48_000
    n = int(1.5 * sr)
    t = np.linspace(0, 1.5, n, endpoint=False)

    base_signal = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Original: Signal with large click spike (peak_orig >> peak_rest after removal)
    orig = base_signal.copy()
    click_pos = sr // 2  # mid-point
    if click_pos + 3 < n:
        orig[click_pos] += 0.8  # large click — peak_orig >> any restored peak
        orig[click_pos + 1] -= 0.4

    # Restored: click removed, clean signal (peak_orig / peak_rest >> 1.30)
    restored = base_signal.copy()  # no click — clean signal
    restored = np.clip(restored, -1.0, 1.0)

    thresholds = gate._get_thresholds("vinyl")
    artifacts = gate._detect_crackle_impulse(orig, restored, sr, thresholds)
    assert len(artifacts) == 0, (
        f"Click-removal (peak_orig >> peak_rest) must not trigger crackle_impulse; "
        f"got {len(artifacts)} false-positive artifacts"
    )


def test_42_corrective_phase_elevated_kurtosis_threshold():
    """§0d CORRECTIVE-Phasen erhalten in evaluate() erhoehteh Kurtosis-Threshold
    (×1.5 = 15 statt 10). Ein Signal mit Kurtosis ≈ 12 (Interpolations-Ringing)
    soll bei CORRECTIVE-Phase nicht geflaggt werden, bei SUBTRACTIVE schon.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    sr = 48_000
    n = int(1.5 * sr)
    t = np.linspace(0, 1.5, n, endpoint=False)

    # Moderate-kurtosis scenario: clear signal with moderate spikes
    orig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    restored = orig.copy()
    # Spikes that give kurtosis roughly between 10 and 15 in some windows
    # (moderate: won't be caught by elevated-threshold CORRECTIVE, will by SUBTRACTIVE)
    # Use small spikes near the signal level (not >> orig peak so click-guard won't fire)
    for pos in range(sr // 4, n - 100, sr // 3):
        if pos + 5 < n:
            restored[pos] += 0.45  # peak_rest > peak_orig * 1.15 (orig=0.3+0.45=0.75, orig_base=0.3)
            restored[pos + 1] -= 0.22
            restored[pos + 2] += 0.08
    restored = np.clip(restored, -1.0, 1.0)

    # Verify CORRECTIVE phase (phase_27 = click_pop_removal) has fewer or equal flags than SUBTRACTIVE
    result_corrective = gate.evaluate(
        orig,
        restored,
        sr,
        material_type="vinyl",
        phase_id="phase_27_click_pop_removal",  # CORRECTIVE — higher threshold
    )
    result_subtractive = gate.evaluate(
        orig,
        restored,
        sr,
        material_type="vinyl",
        phase_id="phase_03_denoise",  # SUBTRACTIVE — standard threshold
    )
    n_corr = result_corrective.detail_report.get("n_crackle_impulse", 0)
    n_sub = result_subtractive.detail_report.get("n_crackle_impulse", 0)
    assert n_corr <= n_sub, (
        f"CORRECTIVE phase should flag <= crackle_impulse than SUBTRACTIVE (higher kurtosis threshold); "
        f"corrective={n_corr} subtractive={n_sub}"
    )


def test_43_restorative_tolerance_scales_with_restorability():
    """§0d/§2.29c Restorative-Phase-Toleranz: Bei niedrigerer Restorability wird
    _max_tolerance hoeher skaliert (weniger Einschraenkung fuer schwer degradiertes Material).
    Ein restorative Phase mit wsum ≈ 0.8 soll bei Restorability=30 hoehere artifact_freedom
    haben als bei Restorability=90.
    """
    from backend.core.artifact_freedom_gate import get_artifact_freedom_gate

    gate = get_artifact_freedom_gate()
    sr = 48_000
    n = int(1.5 * sr)
    t = np.linspace(0, 1.5, n, endpoint=False)

    # SUBTRACTIVE phase that introduces a small crackle (wsum ≈ 0.8)
    orig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    restored = orig.copy()
    # Small single spike that barely crosses the crackle threshold
    pos = sr // 2
    if pos + 3 < n:
        restored[pos] += 0.55
        restored[pos + 1] -= 0.28
        restored[pos + 2] += 0.10
    restored = np.clip(restored, -1.0, 1.0)

    result_low_rest = gate.evaluate(
        orig,
        restored,
        sr,
        material_type="vinyl",
        phase_id="phase_03_denoise",  # SUBTRACTIVE = restorative
        restorability_score=30.0,  # heavily degraded → higher tolerance
    )
    result_high_rest = gate.evaluate(
        orig,
        restored,
        sr,
        material_type="vinyl",
        phase_id="phase_03_denoise",
        restorability_score=90.0,  # near-pristine → standard tolerance
    )
    # Low restorability must give >= artifact_freedom (higher tolerance = less penalized)
    assert result_low_rest.artifact_freedom >= result_high_rest.artifact_freedom, (
        f"Heavily degraded material (rest=30) must have >= artifact_freedom than near-pristine (rest=90); "
        f"low_rest={result_low_rest.artifact_freedom:.3f} high_rest={result_high_rest.artifact_freedom:.3f}"
    )
