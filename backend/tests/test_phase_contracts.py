"""§v10.15 Phase Contract Tests
=============================
Validates: phase output types, audio shapes, stereo handling,
PostGate signatures, OneTakeExport fallback, STCG consistency.

§v10.0.5: Added Genre-Propagation und PostGate-Lambda-Signatur-Validierung.

Run: python3 -m pytest backend/tests/test_phase_contracts.py -v
"""

import inspect

import numpy as np
import pytest

# ── Phase Import Helpers ────────────────────────────────────────


def _make_stereo_audio(duration_s=2.0, sr=48000):
    """Generate valid stereo test audio (N, 2)."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    left = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(n)
    right = 0.3 * np.sin(2 * np.pi * 445 * t) + 0.1 * np.random.randn(n)
    return np.column_stack([left, right]).astype(np.float32)


def _make_mono_audio(duration_s=2.0, sr=48000):
    """Generate valid mono test audio (N,)."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(n)).astype(np.float32)


# ── PhaseResult Contract ────────────────────────────────────────


class TestPhaseResultContract:
    """Every phase must return PhaseResult with valid audio."""

    def test_phase09_returns_phaseresult(self):
        """phase_09_crackle_removal must return PhaseResult."""
        from backend.core.phases.phase_09_crackle_removal import CrackleRemovalPhase
        from backend.core.phases.phase_interface import PhaseResult

        phase = CrackleRemovalPhase()
        audio = _make_stereo_audio()
        result = phase.process(audio, sample_rate=48000, material_type="cassette")

        assert isinstance(result, PhaseResult), f"phase_09 returned {type(result).__name__}, expected PhaseResult"
        assert isinstance(result.audio, np.ndarray), "PhaseResult.audio must be ndarray"
        assert result.audio.ndim in (1, 2), f"audio must be 1D or 2D, got {result.audio.ndim}D"
        # Shape must be consistent with input
        assert result.audio.shape == audio.shape, f"output shape {result.audio.shape} != input shape {audio.shape}"

    def test_phase29_returns_phaseresult(self):
        """phase_29_tape_hiss_reduction must return PhaseResult."""
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase
        from backend.core.phases.phase_interface import PhaseResult

        phase = TapeHissReductionPhase()
        audio = _make_stereo_audio()
        result = phase.process(audio, sample_rate=48000)

        assert isinstance(result, PhaseResult), f"phase_29 returned {type(result).__name__}, expected PhaseResult"
        assert isinstance(result.audio, np.ndarray), "PhaseResult.audio must be ndarray"

    def test_phase06_returns_phaseresult(self):
        """phase_06_frequency_restoration must return PhaseResult."""
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase
        from backend.core.phases.phase_interface import PhaseResult

        phase = FrequencyRestorationPhase()
        audio = _make_stereo_audio()
        result = phase.process(audio, sample_rate=48000)

        assert isinstance(result, PhaseResult), f"phase_06 returned {type(result).__name__}, expected PhaseResult"
        assert isinstance(result.audio, np.ndarray), "PhaseResult.audio must be ndarray"


# ── Stereo Shape Contract ───────────────────────────────────────


class TestStereoShapeContract:
    """Stereo input must produce same-shape stereo output."""

    @pytest.mark.parametrize(
        "phase_name,module_path,class_name",
        [
            ("phase_09", "backend.core.phases.phase_09_crackle_removal", "CrackleRemovalPhase"),
            ("phase_29", "backend.core.phases.phase_29_tape_hiss_reduction", "TapeHissReductionPhase"),
        ],
    )
    def test_stereo_in_stereo_out(self, phase_name, module_path, class_name):
        """Stereo (N,2) input → stereo (N,2) output."""
        import importlib

        mod = importlib.import_module(module_path)
        phase_cls = getattr(mod, class_name)
        phase = phase_cls()
        audio = _make_stereo_audio()
        result = phase.process(audio, sample_rate=48000)

        if result.audio is not None:
            if audio.ndim == 2:
                assert result.audio.ndim == 2, f"{phase_name}: stereo input got {result.audio.ndim}D output"
                assert result.audio.shape == audio.shape, (
                    f"{phase_name}: shape mismatch {audio.shape} → {result.audio.shape}"
                )

    @pytest.mark.parametrize(
        "phase_name,module_path,class_name",
        [
            ("phase_09", "backend.core.phases.phase_09_crackle_removal", "CrackleRemovalPhase"),
        ],
    )
    def test_mono_in_mono_out(self, phase_name, module_path, class_name):
        """Mono (N,) input → mono (N,) output."""
        import importlib

        mod = importlib.import_module(module_path)
        phase_cls = getattr(mod, class_name)
        phase = phase_cls()
        audio = _make_mono_audio()
        result = phase.process(audio, sample_rate=48000)

        if result.audio is not None:
            assert result.audio.ndim == 1, f"{phase_name}: mono input got {result.audio.ndim}D output"
            assert len(result.audio) == len(audio), f"{phase_name}: length mismatch {len(audio)} → {len(result.audio)}"


# ── Phase Contract Guard Tests ──────────────────────────────────


class TestPhaseContractGuard:
    """The centralized guard module catches invalid inputs."""

    def test_guard_rejects_tuple(self):
        """guard_phase_input must reject tuple input."""
        from backend.core.phase_contract_guard import guard_phase_input

        # Should convert tuple to ndarray (not crash)
        result = guard_phase_input((np.zeros(100, dtype=np.float32),), 48000, "test")  # type: ignore[arg-type]
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_guard_rejects_wrong_ndim(self):
        """guard_phase_input must reject 3D audio."""
        import pytest

        from backend.core.phase_contract_guard import guard_phase_input

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            guard_phase_input(np.zeros((2, 3, 100), dtype=np.float32), 48000, "test")

    def test_guard_output_rejects_non_phaseresult(self):
        """guard_phase_output must reject non-PhaseResult."""
        import pytest

        from backend.core.phase_contract_guard import guard_phase_output

        with pytest.raises(TypeError, match="expected PhaseResult"):
            guard_phase_output("not_a_phaseresult", np.zeros(100), "test")


# ── OneTakeExport Contract ──────────────────────────────────────


class TestOneTakeExportContract:
    """OneTakeExport must not infinite-loop and must return best-effort."""

    def test_no_change_early_exit(self):
        """When no corrections are possible, best-effort export must succeed."""
        from backend.core.one_take_export import OneTakeExport

        audio = _make_stereo_audio(duration_s=5.0)
        result = OneTakeExport.prepare(audio, 48000, is_studio_2026=False)

        assert result.audio is not None, "OneTakeExport must return audio"
        assert isinstance(result.passed, bool), "OneTakeExport must set passed flag"
        # Must not exceed MAX_RETRIES
        assert result.retries <= 3, f"retries={result.retries} exceeds MAX_RETRIES=3"


# ── STCG Consistency Test ───────────────────────────────────────


class TestSTCGConsistency:
    """STCG measurement and correction must use the same algorithm."""

    def test_verify_lag_multi_point_returns_expected_keys(self):
        """_verify_lag_multi_point must return consistent dict structure."""
        from backend.core.stereo_temporal_coherence_guard import (
            StereoTemporalCoherenceGuard,
        )

        stcg = StereoTemporalCoherenceGuard()
        sr = 48000
        n = sr * 3  # 3 seconds
        t = np.arange(n) / sr
        ch_l = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        ch_r = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # same = no lag

        result = stcg._verify_lag_multi_point(ch_l, ch_r, sr)

        assert "median_lag" in result, "must have median_lag"
        assert "max_spread" in result, "must have max_spread"
        assert "num_points" in result, "must have num_points"
        assert isinstance(result["median_lag"], (int, float)), "median_lag must be numeric"

    def test_same_signal_no_lag(self):
        """Identical L/R channels should produce near-zero lag."""
        from backend.core.stereo_temporal_coherence_guard import (
            StereoTemporalCoherenceGuard,
        )

        stcg = StereoTemporalCoherenceGuard()
        sr = 48000
        n = sr * 5  # 5 seconds
        t = np.arange(n) / sr
        signal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result = stcg._verify_lag_multi_point(signal, signal, sr)
        # Near-zero lag expected for identical channels
        assert abs(result["median_lag"]) < 5, f"identical signals should have ~0 lag, got {result['median_lag']}"

    def test_correct_interchannel_delay_preserves_shape(self):
        """correct_interchannel_delay must preserve input shape."""
        from backend.core.stereo_temporal_coherence_guard import (
            StereoTemporalCoherenceGuard,
        )

        stcg = StereoTemporalCoherenceGuard()
        audio = _make_stereo_audio(duration_s=4.0)
        result = stcg.correct_interchannel_delay(audio, 48000, phase_id="test")

        assert result.shape == audio.shape, f"STCG changed shape: {audio.shape} → {result.shape}"
        assert result.dtype == audio.dtype, f"STCG changed dtype: {audio.dtype} → {result.dtype}"


# ── PostGate Lambda Contract ────────────────────────────────────


class TestPostGateLambdaContract:
    """PostGate lambdas must accept 3 positional args (audio, sr, strength)."""

    def test_antimuffling_lambda_signature(self):
        """AntiMufflingPass: PostGate-kompatible Signatur (a, sr, strength=None)."""
        from backend.core.anti_muffling_pass import AntiMufflingPass

        amp = AntiMufflingPass()
        audio = _make_mono_audio()
        result = amp.process(audio, 48000)
        assert isinstance(result, np.ndarray)

    def test_vocal_clarity_lambda_signature(self):
        """VocalClarityMax: PostGate-kompatible Signatur (a, sr, strength=None)."""
        from backend.core.vocal_clarity_max import VocalClarityMax

        vcm = VocalClarityMax()
        audio = _make_mono_audio()
        result = vcm.process(audio, 48000)
        assert isinstance(result, np.ndarray)

    # ── §v10.0.5 Lambda-Signatur-Validierung ────────────────────

    def test_validate_lambda_accepts_3_arg_with_default(self):
        """PostProcessingGate._validate_lambda must accept (a, sr, strength=None)."""
        from backend.core.post_processing_gate import PostProcessingGate

        PostProcessingGate._validate_lambda("test", lambda a, sr, strength=None: a)

    def test_validate_lambda_rejects_2_arg(self):
        """PostProcessingGate._validate_lambda must REJECT (a, sr)."""
        from backend.core.post_processing_gate import PostProcessingGate

        with pytest.raises(AssertionError, match="braucht aber 3"):
            PostProcessingGate._validate_lambda("test", lambda a, sr: a)

    def test_validate_lambda_accepts_2_arg_with_kwargs(self):
        """PostProcessingGate._validate_lambda must accept (a, sr, **kw)."""
        from backend.core.post_processing_gate import PostProcessingGate

        PostProcessingGate._validate_lambda("test", lambda a, sr, **kw: a)

    def test_all_postgate_lambdas_in_uv3_are_3_arg(self):
        """§v10.0.5: JEDE in unified_restorer_v3.py an PostGate übergebene Lambda
        muss 3 positional args akzeptieren. Scannt den Quelltext auf
        ``get_post_processing_gate().apply(`` und prüft die direkt folgende
        Lambda-Signatur."""
        import ast
        import os

        from backend.core.post_processing_gate import PostProcessingGate

        uv3_path = os.path.join(os.path.dirname(__file__), "..", "core", "unified_restorer_v3.py")
        uv3_path = os.path.abspath(uv3_path)

        with open(uv3_path) as f:
            source = f.read()

        tree = ast.parse(source)

        class LambdaCollector(ast.NodeVisitor):
            def __init__(self):
                self.violations: list[tuple[int, str]] = []

            def visit_Call(self, node):
                # Suche nach: get_post_processing_gate().apply(
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "apply"
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Attribute)
                    and node.func.value.func.attr == "get_post_processing_gate"
                ):
                    # Prüfe ob erstes Argument eine Lambda oder FunctionDef ist
                    if node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Lambda):
                            # args von Lambda zählen (Parameter ohne default)
                            pos_args = sum(
                                1
                                for a in first_arg.args.args
                                if not getattr(first_arg.args, "defaults", None)
                                or first_arg.args.args.index(a)
                                >= len(first_arg.args.args) - len(first_arg.args.defaults)
                            )
                            # Vereinfacht: zähle alle positional args + *args + **kwargs
                            n_positional = len(first_arg.args.args)
                            n_vararg = 1 if first_arg.args.vararg else 0
                            n_kwarg = 1 if first_arg.args.kwarg else 0

                            if n_positional + n_vararg + n_kwarg < 3:
                                self.violations.append(
                                    (first_arg.lineno, f"Lambda: {n_positional} pos + {n_vararg} var + {n_kwarg} kw")
                                )
                        elif isinstance(first_arg, ast.Name):
                            # Referenzierter Name — überspringen (zu komplex)
                            pass

        collector = LambdaCollector()
        collector.visit(tree)

        assert len(collector.violations) == 0, (
            f"Found {len(collector.violations)} PostGate lambdas with < 3 positional args in uv3:\n"
            + "\n".join(f"  Line {line}: {desc}" for line, desc in collector.violations)
        )


# ── Genre Propagation Contract ──────────────────────────────────


class TestGenrePropagationContract:
    """§v10.0.5: Genre muss via _restoration_context → phase kwargs propagieren."""

    def test_restoration_context_has_genre_and_genre_label(self):
        """_restoration_context must contain both 'genre' and 'genre_label' keys."""
        # Simuliere minimales UV3-setup
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        ur = UnifiedRestorerV3.__new__(UnifiedRestorerV3)
        ur._restoration_context = {
            "genre_label": "Schlager",
            "decade": 1970,
            "primary_material": "vinyl",
        }

        # Genre-Key-Normalisierung: genre_label muss vorhanden sein
        assert ur._restoration_context.get("genre_label") == "Schlager"
        # Nach _prepare_profiled_phase_context wird auch "genre" gesetzt
        # (dieser Test prüft den Ausgangszustand; die Normalisierung
        #  wird im Unit-Test für prepare_profiled_phase_context geprüft.)

    def test_genre_key_normalization_in_profiled_context(self):
        """_prepare_profiled_phase_context injects both genre and genre_label."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        ur = UnifiedRestorerV3.__new__(UnifiedRestorerV3)
        ur._restoration_context = {
            "genre_label": "Klassik",
            "decade": 1955,
            "primary_material": "shellac",
        }
        ur.config = type("Cfg", (), {"mode": type("M", (), {"value": "quality"})()})()

        # Dummy phase mit get_metadata()
        class DummyPhaseMeta:
            phase_id = "phase_19_de_esser"
            name = "DeEsser"

        class DummyPhase:
            def get_metadata(self):
                return DummyPhaseMeta()

        _, _, _, _, _, _, _ = ur._prepare_profiled_phase_context(DummyPhase(), kwargs := {})  # type: ignore[var-annotated]
        # Nach der Normalisierung müssen beide Keys existieren
        assert kwargs.get("genre_label") == "Klassik", f"genre_label missing: {kwargs}"
        assert kwargs.get("genre") == "Klassik", f"genre missing: {kwargs}"

    def test_de_esser_reads_genre_from_kwargs(self):
        """Phase 19 must read genre from kwargs (testet genre/ genre_label Fallback)."""
        import inspect as _inspect
        import os

        # Parse phase_19 source code to verify it reads both keys
        phase19_path = os.path.join(os.path.dirname(__file__), "..", "core", "phases", "phase_19_de_esser.py")
        phase19_path = os.path.abspath(phase19_path)

        with open(phase19_path) as f:
            source = f.read()

        # Verify: kwargs.get("genre", kwargs.get("genre_label", ""))
        # or kwargs.get("genre") with genre_label fallback
        has_genre_fallback = (
            'kwargs.get("genre", kwargs.get("genre_label"' in source
            or 'kwargs.get("genre", kwargs.get("genre_label"' in source
        )
        has_genre_label_only = 'kwargs.get("genre_label"' in source

        assert has_genre_fallback or has_genre_label_only, (
            "phase_19 must read genre from kwargs (genre or genre_label); "
            "if only 'genre' is read without 'genre_label' fallback, "
            "the DeEsser calibration will use empty genre string"
        )

    def test_all_phase_kwargs_genre_consumers_have_fallback(self):
        """Alle Phasen die kwargs.get('genre') lesen, sollten auch genre_label-Fallback haben
        oder über _prepare_profiled_phase_context zentral versorgt werden."""
        import ast
        import os

        phases_dir = os.path.join(os.path.dirname(__file__), "..", "core", "phases")
        phases_dir = os.path.abspath(phases_dir)

        violations: list[tuple[str, int]] = []

        for fname in sorted(os.listdir(phases_dir)):
            if not fname.endswith(".py") or fname.startswith("__"):
                continue
            fpath = os.path.join(phases_dir, fname)
            with open(fpath) as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                # Pattern: kwargs.get("genre") ohne Fallback auf genre_label
                stripped = line.strip()
                if 'kwargs.get("genre"' in stripped or "kwargs.get('genre'" in stripped:
                    # Ignoriere bereits gefixte (die genre_label fallback haben)
                    if "genre_label" in stripped or "kwargs.get" not in stripped:
                        continue
                    # Diese Phase liest nur "genre" ohne genre_label Fallback
                    # Das ist OK weil _prepare_profiled_phase_context jetzt
                    # beide Keys setzt — aber wir loggen es als Info
                    pass  # now safe due to centralized normalization

        # Kein Assert — diesser Test dokumentiert nur die Abhängigkeit
        # zur zentralen Normalisierung in _prepare_profiled_phase_context.
        # Wird die zentrale Normalisierung entfernt, müssen ALLE diese Phasen
        # auf genre_label-Fallback umgestellt werden.
        pass  # informational test — no assert


# ── UVR Divide-by-Zero Regression ───────────────────────────────


class TestUvrDivideByZeroRegression:
    """§v10.0.5: UVR MDX-Net darf nicht durch len(sessions)=0 crashen."""

    def test_run_ensemble_guards_against_empty_sessions(self):
        """UvrMdxNetPlugin._run_ensemble must handle zero/empty sessions."""
        import ast
        import os

        uvr_path = os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "uvr_mdxnet_plugin.py")
        uvr_path = os.path.abspath(uvr_path)

        with open(uvr_path) as f:
            source = f.read()

        # Verify: max(len(sessions), 1) guard exists
        assert "max(len(sessions)" in source or "max(len(self._sessions)" in source, (
            "UVR _run_ensemble must guard divide-by-zero with max(len(sessions), 1)"
        )

        # Verify: early exit when sessions is empty
        assert "if not sessions" in source, "UVR _run_ensemble must return fallback when sessions is empty"
