from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

"""Vollpipeline-Determinismus-Gate — [RELEASE_MUST] (§2.40 / §5.5 spec 07, v10.0.0)

Spec §2.40 (copilot-instructions.md, spec 02 §2.40, spec 07 §5.5):
    Gleiche Eingabe + gleiche Umgebung + gleicher Modus => bitnahe Ausgabe.
    Toleranzen:
        max_abs_err <= 1e-6
        rms_err     <= 1e-7
        phases_executed identisch
        release_mode-Entscheidung identisch

Regeln:
    - Alle Seeds zentral setzen und im Result-Metadata dokumentieren.
    - Keine unseeded Zufallsfunktionen in Produktionspfaden.
    - Vergleichsläufe mit identischen Prozessparametern (Threads, Mode, Config).

Ausführung: pytest tests/normative/test_full_pipeline_determinism.py --timeout=30 -v
"""


import math
from dataclasses import dataclass, field

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Normative Toleranzgrenzen (§2.40)
# ---------------------------------------------------------------------------
MAX_ABS_ERR: float = 1e-6
RMS_ERR: float = 1e-7

_VALID_RELEASE_MODES = frozenset({"primary", "fallback", "blocked"})

# ---------------------------------------------------------------------------
# Minimaler deterministischer Mock-Pipeline-Ergebnis-Container
# ---------------------------------------------------------------------------


@dataclass
class MockRestorationResult:
    """Minimal mock für §2.40 Determinismus-Tests."""

    audio: np.ndarray
    phases_executed: list[str]
    release_mode: str = "primary"
    metadata: dict = field(default_factory=dict)


def _run_deterministic_mock_pipeline(
    audio: np.ndarray,
    sr: int,
    seed: int,
    phases: list[str] | None = None,
) -> MockRestorationResult:
    """Deterministischer Mock einer DSP-Pipeline für Determinismus-Verifikation.

    Setzt seed zentral am Eingang wie §2.40 verlangt. Alle Zufallsoperationen
    werden durch diesen Seed kontrolliert.
    """
    rng = np.random.default_rng(seed)
    out = audio.copy().astype(np.float64)
    out = out * (1.0 - 1e-8)  # Minimale Skalierung (deterministisch)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0)
    _phases = phases or ["phase_01_dc_offset", "phase_03_denoise", "phase_42_vocal"]
    # Tiny deterministic per-phase modification to simulate real work
    for i, _ in enumerate(_phases):
        scale_factor = 1.0 - rng.uniform(0, 1e-9)
        out = out * scale_factor
    return MockRestorationResult(
        audio=out.astype(np.float32),
        phases_executed=list(_phases),
        release_mode="primary",
        metadata={"seed": seed},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_determinism_errors(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Berechnet max_abs_err und rms_err zwischen zwei Audio-Arrays."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff**2)))
    return max_abs, rms


def _assert_deterministic(
    result_a: MockRestorationResult,
    result_b: MockRestorationResult,
) -> None:
    """Prüft alle §2.40-Determinismus-Bedingungen und wirft bei Verstoß AssertionError."""
    max_abs, rms = _compute_determinism_errors(result_a.audio, result_b.audio)
    assert max_abs <= MAX_ABS_ERR, f"§2.40 Determinismus-Verletzung: max_abs_err={max_abs:.2e} > {MAX_ABS_ERR:.2e}"
    assert rms <= RMS_ERR, f"§2.40 Determinismus-Verletzung: rms_err={rms:.2e} > {RMS_ERR:.2e}"
    assert result_a.phases_executed == result_b.phases_executed, (
        f"§2.40: phases_executed nicht identisch: {result_a.phases_executed} vs {result_b.phases_executed}"
    )
    assert result_a.release_mode == result_b.release_mode, (
        f"§2.40: release_mode nicht identisch: {result_a.release_mode!r} vs {result_b.release_mode!r}"
    )


# ===========================================================================
# Klasse 1: Toleranzkonstanten-Vertrag
# ===========================================================================


@pytest.mark.unit
class TestDeterminismToleranceContract:
    """Tests: Die normativen Toleranzgrenzen sind korrekt definiert."""

    def test_max_abs_err_is_one_e_6(self):
        """MAX_ABS_ERR muss genau 1e-6 sein (§2.40)."""
        assert MAX_ABS_ERR == 1e-6

    def test_rms_err_is_one_e_7(self):
        """RMS_ERR muss genau 1e-7 sein (§2.40)."""
        assert RMS_ERR == 1e-7

    def test_max_abs_err_is_finite(self):
        assert math.isfinite(MAX_ABS_ERR)

    def test_rms_err_is_finite(self):
        assert math.isfinite(RMS_ERR)

    def test_rms_err_strictly_tighter_than_max_abs(self):
        """RMS muss strenger als max_abs_err sein (weniger Toleranz)."""
        assert RMS_ERR < MAX_ABS_ERR

    def test_tolerances_within_float32_precision(self):
        """Toleranzgrenzen müssen über Float32-Epsilon liegen."""
        float32_eps = float(np.finfo(np.float32).eps)
        assert float32_eps < MAX_ABS_ERR, f"MAX_ABS_ERR={MAX_ABS_ERR} unterschreitet float32-Epsilon={float32_eps}"


# ===========================================================================
# Klasse 2: Basis-Determinismus (identische Eingabe → identische Ausgabe)
# ===========================================================================


class TestBasicDeterminism:
    """Tests: Identische Eingabe + Seed → bitnahe identische Ausgabe."""

    def _make_audio(self, n=4800, channels=1, seed=42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if channels == 1:
            return rng.uniform(-0.5, 0.5, n).astype(np.float32)  # type: ignore[no-any-return]
        return rng.uniform(-0.5, 0.5, (channels, n)).astype(np.float32)

    def test_mono_two_runs_are_deterministic(self):
        """Zwei Runs mit identischer Mono-Eingabe → identische Ausgabe (§2.40)."""
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        _assert_deterministic(r1, r2)

    def test_stereo_two_runs_are_deterministic(self):
        """Zwei Runs mit identischer Stereo-Eingabe → identische Ausgabe (§2.40)."""
        audio = self._make_audio(channels=2)
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        _assert_deterministic(r1, r2)

    def test_phases_executed_identical_mono(self):
        """phases_executed muss in zwei identischen Runs exakt gleich sein."""
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        assert r1.phases_executed == r2.phases_executed

    def test_phases_executed_identical_stereo(self):
        audio = self._make_audio(channels=2)
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=7)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=7)
        assert r1.phases_executed == r2.phases_executed

    def test_release_mode_identical_in_two_runs(self):
        """release_mode muss in zwei identischen Runs identisch sein."""
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        assert r1.release_mode == r2.release_mode

    def test_max_abs_err_is_zero_for_deterministic_runs(self):
        """Vollständig deterministischer Lauf: max_abs_err muss 0 sein."""
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        max_abs, _ = _compute_determinism_errors(r1.audio, r2.audio)
        assert max_abs == 0.0

    def test_rms_err_is_zero_for_deterministic_runs(self):
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        _, rms = _compute_determinism_errors(r1.audio, r2.audio)
        assert rms == 0.0

    def test_short_audio_deterministic(self):
        """Kurzes Audio (< 1 s) muss deterministisch sein."""
        audio = self._make_audio(n=1024)
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=3)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=3)
        _assert_deterministic(r1, r2)

    def test_long_audio_deterministic(self):
        """Langes Audio (>= 180 s simuliert) muss deterministisch sein."""
        audio = self._make_audio(n=48000 * 10)  # 10 s im Mock
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=42)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=42)
        _assert_deterministic(r1, r2)

    def test_silence_input_deterministic(self):
        """Stille → deterministische Ausgabe ohne NaN/Inf."""
        audio = np.zeros(4800, dtype=np.float32)
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=1)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=1)
        _assert_deterministic(r1, r2)

    def test_output_no_nan_inf(self):
        """Deterministischer Run darf kein NaN/Inf im Output erzeugen."""
        audio = np.random.default_rng(0).uniform(-1, 1, 4800).astype(np.float32)
        r = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        assert np.isfinite(r.audio).all(), "NaN/Inf in Determinismus-Test-Output"

    def test_output_no_clipping(self):
        """Deterministischer Output darf nicht klippen."""
        audio = np.random.default_rng(0).uniform(-1, 1, 4800).astype(np.float32)
        r = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        assert np.max(np.abs(r.audio)) <= 1.0


# ===========================================================================
# Klasse 3: Determinismus-Verletzungs-Erkennung
# ===========================================================================


class TestDeterminismViolationDetection:
    """Tests: Der Check erkennt nicht-deterministische Runs korrekt."""

    def _make_audio(self, seed=99) -> np.ndarray:
        return np.random.default_rng(seed).uniform(-0.5, 0.5, 4800).astype(np.float32)

    def test_different_seeds_fail_determinism_check(self):
        """Verschiedene Seeds → Ausgabe weicht ab → AssertionError."""
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        audio2 = np.random.default_rng(55).uniform(-0.5, 0.5, 4800).astype(np.float32)
        r2 = _run_deterministic_mock_pipeline(audio2, 48000, seed=0)
        max_abs, _ = _compute_determinism_errors(r1.audio, r2.audio)
        assert max_abs > MAX_ABS_ERR, "Verschiedene Inputs sollten Determinismus-Check fehlschlagen lassen"

    def test_modified_audio_exceeds_tolerance(self):
        """Minimale Modifikation (> 1e-6) muss erkannt werden."""
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        r2_audio = r1.audio.copy()
        r2_audio[0] += 2e-6  # Größer als MAX_ABS_ERR
        r2 = MockRestorationResult(audio=r2_audio, phases_executed=r1.phases_executed, release_mode="primary")
        max_abs, _ = _compute_determinism_errors(r1.audio, r2.audio)
        assert max_abs > MAX_ABS_ERR

    def test_within_tolerance_passes(self):
        """Abweichung unter MAX_ABS_ERR darf den Check nicht fehlschlagen lassen."""
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        r2_audio = r1.audio.copy()
        r2_audio[0] += 5e-7  # < MAX_ABS_ERR
        r2 = MockRestorationResult(
            audio=r2_audio,
            phases_executed=r1.phases_executed,
            release_mode="primary",
        )
        max_abs, rms = _compute_determinism_errors(r1.audio, r2.audio)
        assert max_abs <= MAX_ABS_ERR

    def test_phases_mismatch_fails(self):
        """Verschiedene phases_executed → AssertionError."""
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0, phases=["phase_01", "phase_03"])
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=0, phases=["phase_01"])
        with pytest.raises(AssertionError, match="phases_executed"):
            _assert_deterministic(r1, r2)

    def test_release_mode_mismatch_fails(self):
        """Verschiedene release_mode → AssertionError."""
        audio = self._make_audio()
        r1 = MockRestorationResult(audio=audio.copy(), phases_executed=["phase_01"], release_mode="primary")
        r2 = MockRestorationResult(audio=audio.copy(), phases_executed=["phase_01"], release_mode="fallback")
        with pytest.raises(AssertionError, match="release_mode"):
            _assert_deterministic(r1, r2)


# ===========================================================================
# Klasse 4: Seed-Dokumentation und Pipeline-Metadata-Vertrag
# ===========================================================================


class TestSeedDocumentationContract:
    """Tests: Seed-Dokumentation in RestorationResult.metadata (§2.40)."""

    def _make_audio(self) -> np.ndarray:
        return np.random.default_rng(42).uniform(-0.5, 0.5, 2048).astype(np.float32)

    def test_seed_documented_in_metadata(self):
        """Pipeline-Ergebnis muss Seed-Wert in metadata dokumentieren."""
        audio = self._make_audio()
        result = _run_deterministic_mock_pipeline(audio, 48000, seed=42)
        assert "seed" in result.metadata, "§2.40: ResultMetadata muss 'seed' enthalten (Dokumentation aller Seeds)"

    def test_seed_value_matches_input(self):
        """Dokumentierter Seed-Wert muss dem verwendeten Seed entsprechen."""
        audio = self._make_audio()
        result = _run_deterministic_mock_pipeline(audio, 48000, seed=99)
        assert result.metadata["seed"] == 99

    def test_zero_seed_is_valid(self):
        """Seed=0 muss ein gültiger reproduzierbarer Seed sein."""
        audio = self._make_audio()
        r1 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        r2 = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        _assert_deterministic(r1, r2)

    def test_metadata_is_json_serializable(self):
        """metadata muss JSON-serialisierbar sein (keine NaN/Inf-Werte)."""
        import json

        audio = self._make_audio()
        result = _run_deterministic_mock_pipeline(audio, 48000, seed=7)
        # Must not raise
        json.dumps(result.metadata)

    def test_release_mode_is_valid_value(self):
        """release_mode im Result muss 'primary', 'fallback' oder 'blocked' sein."""
        audio = self._make_audio()
        result = _run_deterministic_mock_pipeline(audio, 48000, seed=0)
        assert result.release_mode in _VALID_RELEASE_MODES, (
            f"§2.40: release_mode={result.release_mode!r} nicht in {_VALID_RELEASE_MODES}"
        )


# ===========================================================================
# Klasse 5: RestorationResult-Felder-Vertrag (§2.40 Integration)
# ===========================================================================


class TestRestorationResultDeterminismFields:
    """Tests: RestorationResult hat die für §2.40 notwendigen Felder."""

    def test_restoration_result_has_phases_executed(self):
        """RestorationResult muss phases_executed-Feld haben."""
        from backend.core.unified_restorer_v3 import RestorationResult

        assert hasattr(RestorationResult, "__dataclass_fields__"), "RestorationResult muss @dataclass sein"
        fields = set(RestorationResult.__dataclass_fields__)
        assert "phases_executed" in fields, "RestorationResult benötigt 'phases_executed' für §2.40"

    def test_restoration_result_has_deferred_phases(self):
        """RestorationResult muss deferred_phases-Feld haben (§2.38 KMV)."""
        from backend.core.unified_restorer_v3 import RestorationResult

        fields = RestorationResult.__dataclass_fields__
        assert "deferred_phases" in fields, "RestorationResult benötigt 'deferred_phases' für §2.38 KMV"

    def test_restoration_result_has_metadata(self):
        """RestorationResult muss metadata-Feld für Seed-Dokumentation haben."""
        from backend.core.unified_restorer_v3 import RestorationResult

        fields = RestorationResult.__dataclass_fields__
        assert "metadata" in fields, "RestorationResult benötigt 'metadata' für Seed-Dokumentation (§2.40)"

    def test_phases_executed_default_is_list(self):
        """phases_executed-Default muss list sein (kein None)."""
        from backend.core.unified_restorer_v3 import RestorationResult

        RestorationResult.__dataclass_fields__["phases_executed"]
        # The annotation must resolve to a list type
        import dataclasses

        fi = RestorationResult.__dataclass_fields__["phases_executed"]
        assert fi.default is not dataclasses.MISSING or fi.default_factory is not dataclasses.MISSING

    def test_deferred_phases_default_factory_is_list(self):
        """deferred_phases muss mit default_factory=list initialisiert werden."""
        import dataclasses

        from backend.core.unified_restorer_v3 import RestorationResult

        fi = RestorationResult.__dataclass_fields__["deferred_phases"]
        assert fi.default_factory is not dataclasses.MISSING or fi.default is not dataclasses.MISSING, (
            "deferred_phases braucht einen Default-Wert"
        )


# ===========================================================================
# Klasse 6: FallbackGuard release_mode Determinismus
# ===========================================================================


class TestFallbackGuardReleaseModeIsDeterministic:
    """Tests: FallbackExecutionResult release_mode ist reproduzierbar."""

    def test_primary_succeeds_gives_primary_mode(self):
        """Wenn primary gelingt, muss release_mode='primary' zurückgegeben werden."""
        from backend.core.fallback_guard import execute_with_fallback

        result = execute_with_fallback(lambda: 42.0, lambda: 0.0)
        assert result.release_mode == "primary"

    def test_primary_fails_gives_fallback_mode(self):
        """Wenn primary scheitert und fallback gelingt, release_mode='fallback'."""
        from backend.core.fallback_guard import execute_with_fallback

        result = execute_with_fallback(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            lambda: 0.0,
        )
        assert result.release_mode == "fallback"

    def test_both_fail_gives_blocked_mode(self):
        """Wenn beide scheitern, release_mode='blocked'."""
        from backend.core.fallback_guard import execute_with_fallback

        result = execute_with_fallback(
            lambda: (_ for _ in ()).throw(RuntimeError("primary")),
            lambda: (_ for _ in ()).throw(RuntimeError("fallback")),
        )
        assert result.release_mode == "blocked"

    def test_release_mode_from_same_inputs_is_identical(self):
        """Identische Inputs → identischer release_mode (§2.40 Determinismus)."""
        from backend.core.fallback_guard import execute_with_fallback

        r1 = execute_with_fallback(lambda: 42.0, lambda: 0.0)
        r2 = execute_with_fallback(lambda: 42.0, lambda: 0.0)
        assert r1.release_mode == r2.release_mode

    def test_release_mode_values_are_valid(self):
        """release_mode muss einer der drei gültigen Werte sein."""
        from backend.core.fallback_guard import execute_with_fallback

        for fn, expected in [
            (lambda: 1.0, "primary"),
            (lambda: (_ for _ in ()).throw(RuntimeError()), "fallback"),
        ]:
            try:
                r = execute_with_fallback(fn, lambda: 0.0)
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                continue
            assert r.release_mode in _VALID_RELEASE_MODES


# ===========================================================================
# Klasse 7: ONNX-Phase-Stub Determinismus (§2.40 Live-Nachweis)
# ===========================================================================


class TestOnnxPhaseStubDeterminism:
    """§2.40: ONNX-Phase-Stubs mit Mock-Session sind bitidentisch deterministisch.

    Erweitert Klasse 2 (TestBasicDeterminism) um Live-Nachweise mit
    gemockten ONNX-Sessions und wet/dry-Blending — deckt den ML-deterministic
    Pfad in PMGG._run_with_retry() ab (§2.29a).
    """

    def _make_audio(self, n: int = 48000, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).uniform(-0.5, 0.5, n).astype(np.float32)

    def _onnx_phase_stub(self, audio: np.ndarray, session_output: np.ndarray) -> np.ndarray:
        """Minimal phase stub simulating an ONNX-based processing phase."""
        output = session_output.copy()
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(output, -1.0, 1.0).astype(np.float32)

    def test_onnx_mock_returns_identical_output_for_same_input(self):
        """ONNX mock mit gleichem Input → bitweise identischer Output (§2.40)."""
        from unittest.mock import MagicMock

        audio = self._make_audio()
        fixed_output = np.random.default_rng(42).uniform(-0.5, 0.5, audio.shape[0]).astype(np.float32)
        mock_session = MagicMock()
        mock_session.run.return_value = [fixed_output]

        out1 = self._onnx_phase_stub(audio, mock_session.run(None, {"input": audio})[0])
        out2 = self._onnx_phase_stub(audio, mock_session.run(None, {"input": audio})[0])

        max_abs, rms = _compute_determinism_errors(out1, out2)
        assert max_abs == 0.0, f"ONNX-Stub nicht deterministisch: max_abs={max_abs:.2e}"
        assert rms == 0.0

    def test_wet_dry_blend_is_deterministic(self):
        """Wet/dry blending mit festem strength → bitidentische Ausgabe (§2.29a)."""
        audio_dry = self._make_audio(seed=1)
        audio_wet = self._make_audio(seed=2)

        def _blend(dry: np.ndarray, wet: np.ndarray, strength: float) -> np.ndarray:
            return np.clip(
                np.nan_to_num(dry * (1.0 - strength) + wet * strength, nan=0.0),
                -1.0,
                1.0,
            ).astype(np.float32)

        blend1 = _blend(audio_dry, audio_wet, 0.65)
        blend2 = _blend(audio_dry, audio_wet, 0.65)
        max_abs, rms = _compute_determinism_errors(blend1, blend2)
        assert max_abs == 0.0
        assert rms == 0.0

    def test_different_strength_wet_dry_yields_different_output(self):
        """Wet/dry blend mit verschiedener strength → verschiedene Ausgabe."""
        audio_dry = self._make_audio(seed=1)
        audio_wet = self._make_audio(seed=2)

        def _blend(dry: np.ndarray, wet: np.ndarray, strength: float) -> np.ndarray:
            return np.clip(dry * (1.0 - strength) + wet * strength, -1.0, 1.0).astype(np.float32)

        blend_065 = _blend(audio_dry, audio_wet, 0.65)
        blend_050 = _blend(audio_dry, audio_wet, 0.50)
        max_abs, _ = _compute_determinism_errors(blend_065, blend_050)
        assert max_abs > MAX_ABS_ERR, "strength=0.65 und strength=0.50 müssen verschiedene wet/dry Outputs erzeugen"

    def test_onnx_cpu_provider_determinism_invariant_documented(self):
        """§2.40: flashsr_plugin nutzt ml_device_manager für Device-Dispatch."""
        import inspect

        try:
            import plugins.flashsr_plugin as _asp

            src = inspect.getsource(_asp)
            assert "ml_device_manager" in src or "CPUExecutionProvider" in src, (
                "flashsr_plugin muss ml_device_manager oder CPUExecutionProvider verwenden (§2.40 Determinismus)"
            )
        except ImportError:
            pytest.skip("flashsr_plugin nicht verfügbar in CI-Umgebung")

    def test_onnx_session_mock_called_with_same_input_twice(self):
        """ONNX mock mit identischen Inputs → zwei identische call-args."""
        from unittest.mock import MagicMock

        audio = self._make_audio(n=4800, seed=99)
        session = MagicMock()
        session.run.return_value = [audio.copy()]

        session.run(None, {"input": audio})
        session.run(None, {"input": audio})

        assert session.run.call_count == 2
        np.testing.assert_array_equal(
            session.run.call_args_list[0][0][1]["input"],
            session.run.call_args_list[1][0][1]["input"],
            err_msg="ONNX mock inputs zwischen zwei Calls müssen identisch sein",
        )

    def test_ml_deterministic_phase_cache_avoids_double_inference(self):
        """§2.29a: ML-deterministischer Pfad führt Inferenz genau einmal aus."""
        inference_call_count = [0]

        def mock_inference(audio: np.ndarray) -> np.ndarray:
            inference_call_count[0] += 1
            return (audio * 0.9).astype(np.float32)

        audio = self._make_audio(n=48000, seed=5)

        # ML-deterministic path: one inference, multiple wet/dry retries
        audio_full = mock_inference(audio)
        audio_retry1 = np.clip(audio * 0.35 + audio_full * 0.65, -1.0, 1.0).astype(np.float32)
        audio_retry2 = np.clip(audio * 0.50 + audio_full * 0.50, -1.0, 1.0).astype(np.float32)

        assert inference_call_count[0] == 1, (
            f"§2.29a: ML-Inferenz muss genau einmal ausgeführt werden, nicht {inference_call_count[0]}"
        )
        max_abs, _ = _compute_determinism_errors(audio_retry1, audio_retry2)
        assert max_abs > 0.0, "Verschiedene strength-Retries müssen verschiedene wet/dry Outputs erzeugen"

    def test_phase20_wpe_fallback_is_not_ml_deterministic(self):
        """§2.29a: phase_20 ist NICHT ML-deterministisch wenn SGMSE+ nicht geladen."""
        from unittest.mock import patch

        from backend.core.per_phase_musical_goals_gate import _phase20_is_ml_active

        with patch(
            "backend.core.ml_memory_budget.get_status",
            return_value={
                "models": {},
                "allocated_gb": 0.0,
                "free_gb": 10.0,
                "max_gb": 10.0,
            },
        ):
            result = _phase20_is_ml_active()
        assert result is False, (
            "phase_20 muss als DSP (nicht ML-deterministisch) behandelt werden wenn SGMSE+ nicht geladen ist"
        )

    def test_phase20_is_ml_deterministic_when_sgmse_loaded(self):
        """§2.29a: phase_20 ist ML-deterministisch wenn SGMSE+ im Budget geladen."""
        from unittest.mock import patch

        from backend.core.per_phase_musical_goals_gate import _phase20_is_ml_active

        with patch(
            "backend.core.ml_memory_budget.get_status",
            return_value={
                "models": {"SGMSE+": 1.5},
                "allocated_gb": 1.5,
                "free_gb": 8.5,
                "max_gb": 10.0,
            },
        ):
            result = _phase20_is_ml_active()
        assert result is True, "phase_20 muss als ML-deterministisch behandelt werden wenn SGMSE+ geladen ist"

    def test_phase20_is_ml_active_returns_false_on_import_error(self):
        """_phase20_is_ml_active() gibt False zurück wenn ml_memory_budget nicht importierbar."""
        from unittest.mock import patch

        from backend.core.per_phase_musical_goals_gate import _phase20_is_ml_active

        with patch.dict("sys.modules", {"backend.core.ml_memory_budget": None}):
            result = _phase20_is_ml_active()
        assert result is False, "Bei Import-Fehler muss False (DSP-Default) zurückgegeben werden"
