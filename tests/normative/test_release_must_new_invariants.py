from __future__ import annotations

"""Normative gates for newly introduced RELEASE_MUST invariants.

Covers:
- §0l [RELEASE_MUST] Per-Phase-Strength-Orakel und 15-Ziele-Teamarbeit
- [RELEASE_MUST] Frontend-Version-Anzeige-Invariante
- [RELEASE_MUST] ROCm-TorchAudio-ABI-Invariante
"""


from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UV3_FILE = ROOT / "backend" / "core" / "unified_restorer_v3.py"
MAIN_FILE = ROOT / "Aurik10" / "main.py"
WINDOW_FILE = ROOT / "Aurik10" / "ui" / "modern_window.py"
SPLASH_FILE = ROOT / "Aurik10" / "ui" / "splash_screen.py"
RUN_SCRIPT = ROOT / "run_aurik.sh"
AFG_FILE = ROOT / "backend" / "core" / "artifact_freedom_gate.py"
PHASE_18_FILE = ROOT / "backend" / "core" / "phases" / "phase_18_noise_gate.py"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_must_0l_phase_strength_oracle_and_goal_teamwork_wired() -> None:
    """§0l contract must be represented in runtime wiring tokens."""
    src = UV3_FILE.read_text(encoding="utf-8")

    required_tokens = {
        "phase_strength_oracle_rollout": "UV3 must expose phase-strength-oracle rollout control.",
        "_resolve_phase_strength_oracle_rollout_mode": "UV3 must resolve oracle rollout mode.",
        "_is_phase_strength_oracle_enabled_for_phase": "UV3 must gate oracle activation per phase.",
        "goal_weights": "UV3 must carry song goal weights for team-objective optimization.",
        "effective_goal_targets": "UV3 must carry effective goal targets into runtime context.",
    }

    for token, message in required_tokens.items():
        assert token in src, message


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_must_frontend_version_display_invariant_is_wired_to_single_source() -> None:
    """All required frontend version display paths must be present and bound to __version__."""
    main_src = MAIN_FILE.read_text(encoding="utf-8")
    window_src = WINDOW_FILE.read_text(encoding="utf-8")
    splash_src = SPLASH_FILE.read_text(encoding="utf-8")

    assert "from Aurik10 import __version__" in main_src, "Aurik10/main.py must import __version__ from Aurik10."
    assert "setApplicationVersion(__version__)" in main_src, "Aurik10/main.py must set app version from __version__."

    assert "from Aurik10 import __version__ as _AURIK_VERSION" in window_src, (
        "Aurik10/ui/modern_window.py must derive title version from Aurik10.__version__."
    )
    assert 'setWindowTitle(f"AURIK Professional v{_AURIK_VERSION}")' in window_src, (
        "Aurik10/ui/modern_window.py must expose version in window title."
    )

    assert "from Aurik10 import __version__ as _VERSION" in splash_src, (
        "Aurik10/ui/splash_screen.py must import __version__ for splash badge."
    )
    assert 'vt = f"v{_VERSION}"' in splash_src, "Aurik10/ui/splash_screen.py must render visible version badge."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_must_rocm_torchaudio_abi_invariant_is_enforced_in_launcher() -> None:
    """run_aurik.sh must validate and repair torch/torchaudio ROCm ABI before launch."""
    src = RUN_SCRIPT.read_text(encoding="utf-8")

    required_tokens = {
        "check_rocm_torchaudio_abi()": "Launcher must define ROCm ABI preflight check.",
        "import torch": "Preflight must import torch.",
        "import torchaudio": "Preflight must import torchaudio.",
        "ROCM_STACK_ERR build mismatch": "Preflight must detect build-tag mismatch.",
        "repair_rocm_torchaudio()": "Launcher must provide torchaudio repair path.",
        "torchaudio==$torch_version": "Repair must pin torchaudio to exact torch version.",
        "check_rocm_torchaudio_abi": "Launcher must execute ABI preflight before app start.",
        "AURIK_TORCHAUDIO_DEGRADED=1": "torchaudio-only failure must trigger selective degraded mode.",
        "GPU bleibt AKTIV": "torchaudio-only failure must keep GPU active.",
        "Fallback auf CPU-venv": "torch base-stack failure must fallback to CPU launcher.",
    }

    for token, message in required_tokens.items():
        assert token in src, message


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_startup_suppresses_expected_framework_and_remediation_warnings() -> None:
    """Release startup must not surface expected framework/remediation noise as warnings."""
    main_src = MAIN_FILE.read_text(encoding="utf-8")
    run_src = RUN_SCRIPT.read_text(encoding="utf-8")
    afg_src = AFG_FILE.read_text(encoding="utf-8")
    uv3_src = UV3_FILE.read_text(encoding="utf-8")
    phase18_src = PHASE_18_FILE.read_text(encoding="utf-8")

    assert 'os.environ.setdefault("MIOPEN_LOG_LEVEL", "1")' in main_src
    assert 'export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-1}"' in run_src
    assert "warnings.filterwarnings" in main_src
    assert "Importing from timm\\.models\\.layers is deprecated" in main_src
    assert "torch\\.meshgrid: in an upcoming release" in main_src

    assert 'logger.info(\n                        "§2.50 Quellmaterial-Baseline' in afg_src
    assert 'logger.info(\n                        "§2.50 Stereo-Notfall-Remediation' in uv3_src
    assert 'logger.info("§V19 phase_18: noise_texture_dist=' in phase18_src
    assert 'logger.info(\n                            "§SFT ArtifactRescue' in uv3_src
    assert 'logger.info(\n                    "§2.45a QuietZone-Guard' in uv3_src
    assert 'logger.info(\n                        "ActiveIntervention %s REJECTED: no beneficial score delta' in uv3_src
    assert 'logger.info(\n                    "ActiveIntervention %s REJECTED: quiet-zone target unmet' in uv3_src
    assert 'logger.info(\n                            "§Wall-Time-Budget: %.0f s non-exempt > %.0f s' in uv3_src


# ═══════════════════════════════════════════════════════════════════════════
# §2.8 SOTA Gender Detection (Spec 19)
# ═══════════════════════════════════════════════════════════════════════════

LPC_TRACKER_FILE = ROOT / "backend" / "core" / "dsp" / "lpc_formant_tracker.py"
VOCAL_AI_FILE = ROOT / "backend" / "core" / "vocal_ai_enhancement.py"
PHASE_19_FILE = ROOT / "backend" / "core" / "phases" / "phase_19_de_esser.py"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_gender_sota_classify_gender_via_formants_exists() -> None:
    """§2.8a: classify_gender_via_formants must exist on LPCFormantTracker."""
    src = LPC_TRACKER_FILE.read_text(encoding="utf-8")
    assert "def classify_gender_via_formants" in src, (
        "§2.8a classify_gender_via_formants fehlt in lpc_formant_tracker.py"
    )
    assert "_scan_f0_voiced" in src, "§2.8a _scan_f0_voiced (scanning F0 helper) fehlt"
    assert "_estimate_formants_from_voiced" in src, "§2.8a _estimate_formants_from_voiced (LPC scanning helper) fehlt"
    assert "_GENDER_RANGES" in src, "§2.8a _GENDER_RANGES (formant ranges) fehlt"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_gender_sota_detect_f0_scans_audio() -> None:
    """§2.8b: GenderDetector._detect_f0 must scan, not only check first 100ms."""
    src = VOCAL_AI_FILE.read_text(encoding="utf-8")
    # The old code used audio[:max_samples]; the new code uses a scanning loop
    assert "chunk_samples" in src and "hop_samples" in src, (
        "§2.8b _detect_f0: scanning implementation missing (chunk_samples/hop_samples)"
    )
    assert "best_f0" in src and "best_peak_height" in src, (
        "§2.8b _detect_f0: best_f0 tracking missing — still only checks first window?"
    )
    # Verify the old brittle pattern is gone
    assert "segment = audio[:max_samples]" not in src, (
        "§2.8b _detect_f0: old audio[:max_samples] pattern still present — must scan!"
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_gender_sota_no_dead_methods_in_phase19() -> None:
    """§2.8c: No dead/stub gender methods in phase_19_de_esser.py."""
    src = PHASE_19_FILE.read_text(encoding="utf-8")
    # These stubs were removed
    assert (
        '    def _detect_gender_timeline(self, audio, sample_rate, hop_length=256):\n        """Time-varying gender detection (returns empty on fallback)."""\n        return []'
        not in src
    ), "§2.8c Dead stub _detect_gender_timeline (return []) still present!"
    assert "def _process_per_gender_segments(self, audio, sample_rate, gender_segments, **kwargs):" not in src, (
        "§2.8c Dead stub _process_per_gender_segments still present!"
    )
    assert (
        'def _apply_formant_preservation(\n        self, original, processed, sample_rate, formant_low, formant_high, protection_factor\n    ):\n        """Preserve formant regions by blending original back."""\n        return processed'
        not in src
    ), "§2.8c Dead stub _apply_formant_preservation still present!"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_gender_sota_methods_not_nested_in_build_union() -> None:
    """§2.8d: _detect_gender_* must be on DeEsserPhase, not nested in _build_union_vocal_profile."""
    src = PHASE_19_FILE.read_text(encoding="utf-8")
    # Verify DeEsserPhase has the methods
    from backend.core.phases.phase_19_de_esser import DeEsserPhase

    dp = DeEsserPhase()
    for method_name in [
        "_detect_gender_robust",
        "_detect_gender_simple",
        "_detect_gender_timeline",
        "_process_per_gender_segments",
        "_apply_formant_preservation",
    ]:
        assert hasattr(dp, method_name), (
            f"§2.8d {method_name} fehlt auf DeEsserPhase — vermutlich noch in _build_union_vocal_profile gefangen!"
        )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_gender_sota_lpc_fallback_in_robust_chain() -> None:
    """§2.8e: _detect_gender_robust must have LPC formant tracker fallback."""
    src = PHASE_19_FILE.read_text(encoding="utf-8")
    assert "classify_gender_via_formants" in src, (
        "§2.8e _detect_gender_robust: LPC classify_gender_via_formants fallback fehlt!"
    )
    assert "Burg-LPC fallback" in src or "LPC Formant Gender" in src, (
        "§2.8e _detect_gender_robust: LPC fallback log message fehlt!"
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_gender_sota_detect_gender_simple_scans() -> None:
    """§2.8f: _detect_gender_simple must scan audio, not only first 5 seconds."""
    src = PHASE_19_FILE.read_text(encoding="utf-8")
    # New scanning pattern
    assert "win_samples = sample_rate * 2" in src, "§2.8f _detect_gender_simple: scanning windows missing"
    assert "best_f0" in src and "best_peak_height" in src, "§2.8f _detect_gender_simple: best_f0 tracking missing"
    # Old brittle pattern must be gone
    assert "max_samples = sample_rate * 5" not in src, (
        "§2.8f _detect_gender_simple: old max_samples=5s pattern still present — must scan!"
    )
