"""§v10.994: Model-Zoo-Aktivierung — Potenzial wird Wirkung.

Pins die drei Aktivierungs-Ebenen:
  1. RepairPlanner aktiviert SGMSE+/MP-SENet kontextabhängig (vocal_confidence)
  2. _run_denoise führt die Opt-In-Kette aus — immer mit DSP-Fallback
  3. MP-SENet-Norm-Kalibrierung ist skalenfest (Peak-99 + Gain + Loudness-Guard)
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.coordinated_repair import (
    _denormalize_amp,
    _guard_amp_loudness,
    _normalize_amp_peak99,
)
from backend.core.defect_consensus_pipeline import DefectCategory, DefectHypothesis, DefectManifest


def _manifest_with(category: DefectCategory) -> DefectManifest:
    return DefectManifest(
        defects=[
            DefectHypothesis(
                category=category,
                start_sample=0,
                end_sample=48000,
                confidence=0.8,
                severity=0.4,
                source_module="test",
            )
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Kontextabhängige Aktivierung im RepairPlanner
# ═══════════════════════════════════════════════════════════════════════════════


def _plan_params(category: DefectCategory, vocal_confidence: float | None) -> dict:
    from backend.core.coordinated_repair import RepairPlanner

    metadata = None if vocal_confidence is None else {"vocal_confidence": vocal_confidence}
    plan = RepairPlanner().plan(_manifest_with(category), 48000, metadata)
    for step in plan.steps:
        if step.defect_category == category.value:
            return step.parameters
    return {}


def test_planner_activates_sgmse_and_mp_senet_for_strong_vocals():
    params = _plan_params(DefectCategory.HISS, 0.75)
    assert params.get("use_sgmse") is True
    assert params.get("use_mp_senet") is True


def test_planner_activates_only_sgmse_for_moderate_vocals():
    params = _plan_params(DefectCategory.HISS, 0.55)
    assert params.get("use_sgmse") is True
    assert params.get("use_mp_senet") is not True  # konservativ: erst ab 0.65


def test_planner_stays_conservative_without_metadata():
    params = _plan_params(DefectCategory.HISS, None)
    assert params.get("use_sgmse") is not True
    assert params.get("use_mp_senet") is not True


def test_planner_uses_lower_sigma_for_reverb():
    params = _plan_params(DefectCategory.REVERB_TAIL, 0.6)
    assert params.get("use_sgmse") is True
    assert params.get("sgmse_sigma") == 0.4


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Opt-In-Kette in _run_denoise — nie stiller Ausfall
# ═══════════════════════════════════════════════════════════════════════════════


def _make_step(**params: object) -> object:
    from backend.core.coordinated_repair import RepairPriority, RepairStep

    return RepairStep(
        phase_id="phase_03_denoise",
        priority=RepairPriority.BREITBAND,
        defect_category="hiss",
        affected_samples=[],
        parameters=params,
    )


def _fake_dsp(audio: np.ndarray, *, factor: float = 0.5) -> np.ndarray:
    return (np.asarray(audio) * factor).astype(np.float32)


def test_denoise_uses_sgmse_when_opted_in(monkeypatch):
    from backend.core.coordinated_repair import CoordinatedRepair

    calls: list = []
    monkeypatch.setattr(
        "plugins.sgmse_plugin.enhance_sgmse",
        lambda audio, sr, sigma: (
            type("R", (), {"audio": _fake_dsp(audio, factor=0.3)})() if calls.append(sigma) is None else None
        ),
    )
    audio = np.ones(4096, dtype=np.float32) * 0.5
    step = _make_step(use_sgmse=True, sgmse_sigma=0.5)
    out = CoordinatedRepair()._run_denoise(audio, step, None, 48000)
    assert calls == [0.5]
    assert np.allclose(out, audio * 0.3)


def test_denoise_falls_back_to_dsp_when_sgmse_raises(monkeypatch):
    from backend.core.coordinated_repair import CoordinatedRepair

    def _boom(audio, sr, sigma):
        raise RuntimeError("modell weg")

    monkeypatch.setattr("plugins.sgmse_plugin.enhance_sgmse", _boom)
    # DSP-Pipeline patchen (leichter Fake statt echtes SOTA-Denoising)
    import backend.core.sota_denoise_pipeline as _sota_mod

    class _FakePipeline:
        def process(self, audio, sr, override_strength=None):
            return type("R", (), {"audio": _fake_dsp(audio, factor=0.7)})()

    monkeypatch.setattr(_sota_mod, "SOTADenoisePipeline", _FakePipeline)

    audio = np.ones(4096, dtype=np.float32) * 0.5
    step = _make_step(use_sgmse=True)
    out = CoordinatedRepair()._run_denoise(audio, step, None, 48000)
    assert np.allclose(out, audio * 0.7)  # DSP-Fallback griff


def test_denoise_skips_sgmse_without_flag(monkeypatch):
    from backend.core.coordinated_repair import CoordinatedRepair

    monkeypatch.setattr(
        "plugins.sgmse_plugin.enhance_sgmse",
        lambda *a, **kw: pytest.fail("SGMSE+ darf ohne Opt-In nicht laufen"),
    )
    import backend.core.sota_denoise_pipeline as _sota_mod

    class _FakePipeline:
        def process(self, audio, sr, override_strength=None):
            return type("R", (), {"audio": _fake_dsp(audio, factor=0.8)})()

    monkeypatch.setattr(_sota_mod, "SOTADenoisePipeline", _FakePipeline)

    audio = np.ones(4096, dtype=np.float32) * 0.5
    step = _make_step()  # kein Flag
    out = CoordinatedRepair()._run_denoise(audio, step, None, 48000)
    assert np.allclose(out, audio * 0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MP-SENet Norm-Kalibrierung — Skalenfestigkeit
# ═══════════════════════════════════════════════════════════════════════════════


def test_amp_norm_roundtrip_is_scale_invariant():
    rng = np.random.default_rng(7)
    amp = np.abs(rng.standard_normal((32, 201))).astype(np.float32) * 0.1

    norm, scale = _normalize_amp_peak99(amp)
    assert scale > 0
    assert float(np.percentile(norm, 99.0)) == pytest.approx(1.0, abs=1e-4)
    # Roundtrip reproduziert das Original
    restored = _denormalize_amp(norm, scale)
    assert np.allclose(restored, amp, atol=1e-5)


def test_amp_norm_silence_is_safe():
    amp = np.zeros((4, 201), dtype=np.float32)
    norm, scale = _normalize_amp_peak99(amp)
    assert scale == 1.0  # kein Division-by-Zero
    assert np.allclose(norm, amp)


def test_loudness_guard_caps_overshoot():
    original = np.full((16, 201), 0.5, dtype=np.float32)
    denoised = np.full((16, 201), 0.9, dtype=np.float32)  # +80% Overshoot
    capped = _guard_amp_loudness(denoised, original)
    assert float(np.max(capped)) == pytest.approx(0.5 * 1.05, abs=1e-4)

    # Kein Overshoot → unverändert
    fine = _guard_amp_loudness(original * 0.9, original)
    assert np.allclose(fine, original * 0.9)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Zoo-Registry spiegelt die Aktivierung
# ═══════════════════════════════════════════════════════════════════════════════


def test_zoo_registry_reflects_activation():
    from backend.core.model_zoo_registry import get_model

    sgmse = get_model("sgmse_plus")
    assert sgmse is not None and sgmse.status == "active"
    assert "_run_denoise" in (sgmse.integration or "")

    mp = get_model("mp_senet")
    assert mp is not None
    assert "Kalibrierung" in mp.notes  # nicht mehr "offen"

    mel = get_model("melbandroformer")
    assert mel is not None
    assert "bs_roformer_plugin" in mel.notes  # präzise statt vage "Kalibrierung offen"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.998: Die Kassetten-Katastrophe — Null-Schwere-Fehlalarme + Energy-Collapse
# ═══════════════════════════════════════════════════════════════════════════════


def test_planner_skips_zero_severity_false_alarms():
    """severity < 0.05 darf keine Phase triggern (Kassetten-Diagnose-Befund)."""
    from backend.core.coordinated_repair import RepairPlanner

    manifest = DefectManifest(
        defects=[
            DefectHypothesis(
                category=DefectCategory.HUM,
                start_sample=0,
                end_sample=48000,
                confidence=0.26,
                severity=0.0,
                source_module="x",
            ),
            DefectHypothesis(
                category=DefectCategory.CLIPPING,
                start_sample=0,
                end_sample=48000,
                confidence=0.79,
                severity=0.0,
                source_module="x",
            ),
            DefectHypothesis(
                category=DefectCategory.HISS,
                start_sample=0,
                end_sample=48000,
                confidence=0.8,
                severity=0.4,
                source_module="x",
            ),
        ]
    )
    plan = RepairPlanner().plan(manifest, 48000)
    phase_ids = [s.phase_id for s in plan.steps]
    assert "phase_02_hum_removal" not in phase_ids  # sev 0.0 → kein Hum-Schritt
    assert "phase_07_declipper" not in phase_ids  # sev 0.0 trotz conf 0.79
    assert "phase_03_denoise" in phase_ids  # sev 0.4 → läuft


def test_hum_handler_passes_strength_from_step(monkeypatch):
    """Ohne strength-Durchreichung lief Phase 02 mit voller Stärke auf Fehlalarm."""
    import numpy as np

    from backend.core.coordinated_repair import CoordinatedRepair, RepairPriority, RepairStep

    captured: dict = {}

    class _FakePhase:
        def _detect_musical_content(self, audio, freq):
            return False  # §v10.998: Do-no-harm-Gate — kein Musik-Befund

        def process(self, **kwargs):
            captured.update(kwargs)
            return type("R", (), {"audio": kwargs["audio"]})()

    monkeypatch.setattr("backend.core.phases.phase_02_hum_removal.HumRemovalPhase", _FakePhase)
    step = RepairStep(
        phase_id="phase_02_hum_removal",
        priority=RepairPriority.TONAL,
        defect_category="hum",
        affected_samples=[],
        parameters={"strength": 0.02},
    )
    audio = np.ones(4096, dtype=np.float32) * 0.1
    CoordinatedRepair()._run_hum_removal(audio, step, None, 48000)
    assert captured.get("strength") == 0.02  # volle Stärke 1.0 wäre der alte Bug


def test_energy_collapse_guard_reverts_destruction(monkeypatch):
    """RMS < 25% des Eingangs → Schritt wird vollständig zurückgerollt."""
    import numpy as np

    from backend.core.coordinated_repair import CoordinatedRepair, RepairPlan, RepairPriority, RepairStep

    plan = RepairPlan(
        steps=[
            RepairStep(
                phase_id="phase_99_collapse_test",
                priority=RepairPriority.TRANSIENT,
                defect_category="test",
                affected_samples=[],
            ),
        ]
    )
    audio = np.ones(48000, dtype=np.float32) * 0.5
    executor = CoordinatedRepair()
    monkeypatch.setattr(
        executor,
        "_execute_step",
        lambda audio, step, manifest, sr, n_channels: audio * 0.01,  # kollabiert auf 1%
    )
    out, report = executor.execute(audio, plan, None, 48000)
    assert np.allclose(out, audio)  # revert
    assert report.guard_violations.get("energy_collapse") == 1
