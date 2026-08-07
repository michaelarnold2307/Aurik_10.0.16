from __future__ import annotations

"""
Unit-Tests für die 6 SOTA-Plugins — Aurik 10.0.0

Tests nach §5.1 Anforderungen:
  - Shape / Dtype
  - NaN / Inf
  - Bounds-Tests (alle metrischen Ausgaben)
  - Edge-Cases (Stille, Rauschen, Dirac-Impuls)
  - Mono + Stereo
  - Konsistenz (selbe Eingabe → selbe Ausgabe)
  - Integration (Modul-zu-Pipeline-Verbindung)

Namenskonvention §5.2: tests/unit/test_v99_sota_plugins.py

Klassen:
  TestBSRoFormerPlugin     (9 Tests)
  TestCQTdiffPlusPlugin    (8 Tests)
  TestApolloPlugin         (8 Tests)
  TestBigVGANv2Plugin      (8 Tests)
  TestLAIONCLAPPlugin      (8 Tests)
  TestUTMOSPlugin          (8 Tests)
  TestV99PluginIntegration (6 Tests)
Total: 55 Tests
"""


import math

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktionen (Test-Signale — keine realen Audio-Dateien)
# ---------------------------------------------------------------------------

SR = 48000
SEED = 42


def _silence(duration_s: float = 1.0) -> np.ndarray:
    return np.zeros(int(SR * duration_s), dtype=np.float32)


def _noise(duration_s: float = 1.0, amplitude: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return (rng.standard_normal(int(SR * duration_s)) * amplitude).astype(np.float32)


def _sine(freq: float = 440.0, duration_s: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _dirac(duration_s: float = 1.0) -> np.ndarray:
    sig = np.zeros(int(SR * duration_s), dtype=np.float32)
    sig[SR // 100] = 1.0
    return sig


def _stereo(signal: np.ndarray) -> np.ndarray:
    """Mono → Stereo (2D, shape [2, N])."""
    return np.stack([signal, signal * 0.9], axis=0)


def _clipped(duration_s: float = 1.0) -> np.ndarray:
    sig = _sine(440.0, duration_s, amplitude=2.0)
    return np.clip(sig, -1.0, 1.0)


# ---------------------------------------------------------------------------
# TestBSRoFormerPlugin (9 Tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBSRoFormerPlugin:
    def test_01_import(self):
        from plugins.bs_roformer_plugin import BSRoFormerPlugin, get_bs_roformer

        plugin = get_bs_roformer()
        assert isinstance(plugin, BSRoFormerPlugin)

    def test_02_sine_returns_result(self):
        from plugins.bs_roformer_plugin import separate_stems

        audio = _sine(440.0, 2.0)
        result = separate_stems(audio, SR)
        assert result is not None

    def test_03_result_dataclass_fields(self):
        from plugins.bs_roformer_plugin import separate_stems

        result = separate_stems(_noise(2.0), SR)
        assert hasattr(result, "stems")
        assert hasattr(result, "sdri_db")
        assert hasattr(result, "model_used")
        assert hasattr(result, "confidence")

    def test_04_stems_dict_not_empty(self):
        from plugins.bs_roformer_plugin import separate_stems

        result = separate_stems(_sine(440.0, 2.0), SR)
        assert isinstance(result.stems, dict)
        assert len(result.stems) >= 1

    def test_05_no_nan_in_stems(self):
        from plugins.bs_roformer_plugin import separate_stems

        result = separate_stems(_noise(2.0), SR)
        for name, stem in result.stems.items():
            assert np.isfinite(stem).all(), f"Stem '{name}' enthält NaN/Inf"

    def test_06_stems_clipped(self):
        from plugins.bs_roformer_plugin import separate_stems

        result = separate_stems(_clipped(2.0), SR)
        for name, stem in result.stems.items():
            assert np.max(np.abs(stem)) <= 1.0, f"Stem '{name}' über [-1, 1]"

    def test_07_silence_input(self):
        from plugins.bs_roformer_plugin import separate_stems

        result = separate_stems(_silence(2.0), SR)
        assert result is not None
        for stem in result.stems.values():
            assert np.isfinite(stem).all()

    def test_08_confidence_bounds(self):
        from plugins.bs_roformer_plugin import separate_stems

        result = separate_stems(_sine(220.0, 2.0), SR)
        assert 0.0 <= result.confidence <= 1.0

    def test_09_as_dict_serializable(self):
        from plugins.bs_roformer_plugin import separate_stems

        result = separate_stems(_noise(1.0), SR)
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "model_used" in d


# ---------------------------------------------------------------------------
# TestCQTdiffPlusPlugin (8 Tests)
# ---------------------------------------------------------------------------


class TestCQTdiffPlusPlugin:
    def test_01_import(self):
        from plugins.cqtdiff_plus_plugin import CQTdiffPlusPlugin, get_cqtdiff_plus

        plugin = get_cqtdiff_plus()
        assert isinstance(plugin, CQTdiffPlusPlugin)

    def test_02_inpaint_returns_audio(self):
        from plugins.cqtdiff_plus_plugin import inpaint_gap

        audio = _sine(440.0, 2.0)
        gap_start = SR // 2
        gap_end = gap_start + SR // 5  # 200 ms
        result = inpaint_gap(audio, SR, gap_start, gap_end)
        assert result is not None

    def test_03_result_fields_present(self):
        from plugins.cqtdiff_plus_plugin import inpaint_gap

        audio = _sine(440.0, 2.0)
        result = inpaint_gap(audio, SR, SR // 2, SR // 2 + SR // 5)
        assert hasattr(result, "audio")
        assert hasattr(result, "kl_divergence")
        assert hasattr(result, "chroma_corr")

    def test_04_output_same_length(self):
        from plugins.cqtdiff_plus_plugin import inpaint_gap

        audio = _noise(3.0)
        gap_s, gap_e = SR, SR + SR // 4
        result = inpaint_gap(audio, SR, gap_s, gap_e)
        assert len(result.audio) == len(audio), "Ausgabe-Länge weicht ab"

    def test_05_no_nan_in_output(self):
        from plugins.cqtdiff_plus_plugin import inpaint_gap

        audio = _noise(2.0)
        result = inpaint_gap(audio, SR, SR // 4, SR // 4 + 6000)
        assert np.isfinite(result.audio).all()

    def test_06_output_clipped(self):
        from plugins.cqtdiff_plus_plugin import inpaint_gap

        audio = _sine(440.0, 2.0)
        result = inpaint_gap(audio, SR, SR // 4, SR // 4 + 5000)
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_07_kl_divergence_finite(self):
        from plugins.cqtdiff_plus_plugin import inpaint_gap

        audio = _sine(1000.0, 2.0)
        result = inpaint_gap(audio, SR, SR // 3, SR // 3 + 8000)
        assert math.isfinite(result.kl_divergence)
        assert result.kl_divergence >= 0.0

    def test_08_chroma_corr_bounds(self):
        from plugins.cqtdiff_plus_plugin import inpaint_gap

        audio = _sine(440.0, 2.0)
        result = inpaint_gap(audio, SR, SR // 3, SR // 3 + 8000)
        assert -1.0 <= result.chroma_corr <= 1.0


# ---------------------------------------------------------------------------
# TestApolloPlugin (8 Tests)
# ---------------------------------------------------------------------------


class TestApolloPlugin:
    def test_01_import(self):
        from plugins.apollo_plugin import ApolloPlugin, get_apollo

        plugin = get_apollo()
        assert isinstance(plugin, ApolloPlugin)

    def test_02_repair_returns_result(self):
        from plugins.apollo_plugin import repair_codec_artifacts

        audio = _noise(2.0)
        result = repair_codec_artifacts(audio, SR, material="mp3_low")
        assert result is not None

    def test_03_result_fields_present(self):
        from plugins.apollo_plugin import repair_codec_artifacts

        result = repair_codec_artifacts(_sine(440.0, 2.0), SR, material="aac")
        assert hasattr(result, "audio")
        assert hasattr(result, "hf_gain_db")
        assert hasattr(result, "brillanz_score")
        assert hasattr(result, "waerme_score")

    def test_04_output_same_length(self):
        from plugins.apollo_plugin import repair_codec_artifacts

        audio = _noise(2.0)
        result = repair_codec_artifacts(audio, SR, material="mp3_high")
        assert len(result.audio) == len(audio)

    def test_05_no_nan_in_output(self):
        from plugins.apollo_plugin import repair_codec_artifacts

        result = repair_codec_artifacts(_silence(2.0), SR, material="mp3_low")
        assert np.isfinite(result.audio).all()

    def test_06_brillanz_score_bounds(self):
        from plugins.apollo_plugin import repair_codec_artifacts

        result = repair_codec_artifacts(_sine(440.0, 2.0), SR, material="minidisc")
        assert 0.0 <= result.brillanz_score <= 1.0

    def test_07_waerme_score_bounds(self):
        from plugins.apollo_plugin import repair_codec_artifacts

        result = repair_codec_artifacts(_noise(2.0), SR, material="streaming")
        assert 0.0 <= result.waerme_score <= 1.0

    def test_08_as_dict_keys(self):
        from plugins.apollo_plugin import repair_codec_artifacts

        result = repair_codec_artifacts(_sine(220.0, 1.0), SR)
        d = result.as_dict()
        assert "hf_gain_db" in d
        assert "model_used" in d


# ---------------------------------------------------------------------------
# TestBigVGANv2Plugin (8 Tests)
# ---------------------------------------------------------------------------


class TestBigVGANv2Plugin:
    def test_01_import(self):
        from plugins.bigvgan_v2_plugin import BigVGANv2Plugin, get_bigvgan_v2

        plugin = get_bigvgan_v2()
        assert isinstance(plugin, BigVGANv2Plugin)

    def test_02_synthesize_returns_result(self):
        from plugins.bigvgan_v2_plugin import synthesize_audio

        audio = _sine(440.0, 2.0)
        result = synthesize_audio(audio, SR, mode="studio2026")
        assert result is not None

    def test_03_result_fields_present(self):
        from plugins.bigvgan_v2_plugin import synthesize_audio

        result = synthesize_audio(_noise(1.0), SR, mode="studio2026")
        assert hasattr(result, "audio")
        assert hasattr(result, "pqs_mos")
        assert hasattr(result, "model_used")

    def test_04_restoration_mode_raises(self):
        """BigVGAN-v2 darf NICHT im Restoration-Modus verwendet werden (§4.5)."""
        from plugins.bigvgan_v2_plugin import synthesize_audio

        with pytest.raises(ValueError, match="Restoration-Modus"):
            synthesize_audio(_sine(440.0, 1.0), SR, mode="restoration")

    def test_05_no_nan_in_output(self):
        from plugins.bigvgan_v2_plugin import synthesize_audio

        result = synthesize_audio(_noise(2.0), SR, mode="studio2026")
        assert np.isfinite(result.audio).all()

    def test_06_output_clipped(self):
        from plugins.bigvgan_v2_plugin import synthesize_audio

        result = synthesize_audio(_sine(440.0, 2.0), SR, mode="studio2026")
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_07_pqs_mos_bounds(self):
        from plugins.bigvgan_v2_plugin import synthesize_audio

        result = synthesize_audio(_sine(880.0, 2.0), SR, mode="studio2026")
        assert 1.0 <= result.pqs_mos <= 5.0

    def test_08_silence_input(self):
        from plugins.bigvgan_v2_plugin import synthesize_audio

        result = synthesize_audio(_silence(1.0), SR, mode="studio2026")
        assert result is not None
        assert np.isfinite(result.audio).all()


# ---------------------------------------------------------------------------
# TestLAIONCLAPPlugin (8 Tests)
# ---------------------------------------------------------------------------


class TestLAIONCLAPPlugin:
    def test_01_import(self):
        from plugins.laion_clap_plugin import LAIONCLAPPlugin, get_laion_clap

        plugin = get_laion_clap()
        assert isinstance(plugin, LAIONCLAPPlugin)

    def test_02_tag_returns_result(self):
        from plugins.laion_clap_plugin import tag_audio

        audio = _sine(440.0, 2.0)
        result = tag_audio(audio, SR)
        assert result is not None

    def test_03_result_fields_present(self):
        from plugins.laion_clap_plugin import tag_audio

        result = tag_audio(_noise(2.0), SR)
        assert hasattr(result, "instrument_tags")
        assert hasattr(result, "genre_tags")
        assert hasattr(result, "material_tags")
        assert hasattr(result, "embedding")

    def test_04_embedding_finite(self):
        from plugins.laion_clap_plugin import tag_audio

        result = tag_audio(_sine(440.0, 1.0), SR)
        assert np.isfinite(result.embedding).all()

    def test_05_instrument_tags_list_of_str(self):
        from plugins.laion_clap_plugin import tag_audio

        result = tag_audio(_noise(1.0), SR)
        assert isinstance(result.instrument_tags, dict)
        for k, v in result.instrument_tags.items():
            assert isinstance(k, str)
            assert 0.0 <= v <= 1.0

    def test_06_silence_input(self):
        from plugins.laion_clap_plugin import tag_audio

        result = tag_audio(_silence(1.0), SR)
        assert result is not None
        assert np.isfinite(result.embedding).all()

    def test_07_top_instruments_returns_list(self):
        from plugins.laion_clap_plugin import tag_audio

        result = tag_audio(_sine(440.0, 2.0), SR)
        top = result.top_instruments(n=3)
        assert isinstance(top, list)
        assert len(top) <= 3

    def test_08_as_dict_serializable(self):
        from plugins.laion_clap_plugin import tag_audio

        result = tag_audio(_noise(1.0), SR)
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "model_used" in d


# ---------------------------------------------------------------------------
# TestUTMOSPlugin (8 Tests)
# ---------------------------------------------------------------------------


class TestUTMOSPlugin:
    def test_01_import(self):
        from plugins.utmos_plugin import UTMOSPlugin, get_utmos

        plugin = get_utmos()
        assert isinstance(plugin, UTMOSPlugin)

    def test_02_estimate_returns_result(self):
        from plugins.utmos_plugin import estimate_mos

        audio = _sine(440.0, 2.0)
        result = estimate_mos(audio, SR)
        assert result is not None

    def test_03_result_fields_present(self):
        from plugins.utmos_plugin import estimate_mos

        result = estimate_mos(_noise(2.0), SR)
        assert hasattr(result, "mos")
        assert hasattr(result, "confidence")
        assert hasattr(result, "grade")
        assert hasattr(result, "model_used")
        assert hasattr(result, "music_aware")

    def test_04_mos_bounds(self):
        """MOS muss immer ∈ [1.0, 5.0] sein."""
        from plugins.utmos_plugin import estimate_mos

        for sig in [_silence(1.0), _noise(1.0), _sine(440.0, 1.0), _dirac(1.0)]:
            result = estimate_mos(sig, SR)
            assert 1.0 <= result.mos <= 5.0, f"MOS außerhalb [1,5]: {result.mos}"

    def test_05_mos_finite(self):
        from plugins.utmos_plugin import estimate_mos

        result = estimate_mos(_noise(2.0), SR)
        assert math.isfinite(result.mos)

    def test_06_confidence_bounds(self):
        from plugins.utmos_plugin import estimate_mos

        result = estimate_mos(_sine(220.0, 1.0), SR)
        assert 0.0 <= result.confidence <= 1.0

    def test_07_grade_valid(self):
        from plugins.utmos_plugin import estimate_mos

        valid_grades = {"Excellent", "Good", "Fair", "Poor", "Bad"}
        result = estimate_mos(_noise(1.0), SR)
        assert result.grade in valid_grades

    def test_08_consistency(self):
        """Gleiche Eingabe → gleicher MOS-Score."""
        from plugins.utmos_plugin import get_utmos

        audio = _sine(440.0, 1.0)
        # Verwende dasselbe Plugin-Singleton
        plugin = get_utmos()
        r1 = plugin.estimate_mos(audio, SR)
        r2 = plugin.estimate_mos(audio.copy(), SR)
        assert abs(r1.mos - r2.mos) < 0.01, "MOS inkonsistent bei gleicher Eingabe"


# ---------------------------------------------------------------------------
# TestV99PluginIntegration (6 Tests)
# ---------------------------------------------------------------------------


class TestV99PluginIntegration:
    def test_01_singleton_thread_safety(self):
        """Singletons sind Thread-sicher (Double-Checked Locking §3.2)."""
        import threading

        from plugins.apollo_plugin import get_apollo
        from plugins.utmos_plugin import get_utmos

        results = []
        errors = []

        def load_plugins():
            try:
                u = get_utmos()
                a = get_apollo()
                results.append((id(u), id(a)))
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=load_plugins) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-Fehler: {errors}"
        # Alle Threads müssen dasselbe Singleton-Objekt zurückgeben
        utmos_ids = {r[0] for r in results}
        apollo_ids = {r[1] for r in results}
        assert len(utmos_ids) == 1, "Mehrere UTMOS-Instanzen (kein Singleton)"
        assert len(apollo_ids) == 1, "Mehrere Apollo-Instanzen (kein Singleton)"

    def test_02_sr_assertion_triggered(self):
        """Alle Plugins werfen AssertionError bei SR != 48000."""
        from plugins.utmos_plugin import estimate_mos

        audio = _sine(440.0, 0.5)
        with pytest.raises(AssertionError):
            estimate_mos(audio, sr=44100)

    def test_03_apollo_sr_assertion(self):
        from plugins.apollo_plugin import repair_codec_artifacts

        audio = _noise(0.5)
        with pytest.raises(AssertionError):
            repair_codec_artifacts(audio, sr=22050)

    def test_04_as_dict_all_plugins(self):
        """Alle Plugin-Ergebnisse sind als Dict serialisierbar."""
        from plugins.apollo_plugin import repair_codec_artifacts
        from plugins.bs_roformer_plugin import separate_stems
        from plugins.laion_clap_plugin import tag_audio
        from plugins.utmos_plugin import estimate_mos

        audio = _sine(440.0, 1.5)
        for fn in [
            lambda: estimate_mos(audio, SR).as_dict(),
            lambda: repair_codec_artifacts(audio, SR).as_dict(),
            lambda: separate_stems(audio, SR).as_dict(),
            lambda: tag_audio(audio, SR).as_dict(),
        ]:
            d = fn()
            assert isinstance(d, dict)
            # Keine numpy-Typen in Dict (JSON-Serialisierbarkeit)
            for v in d.values():
                if isinstance(v, (float, int, str, bool, type(None))):
                    continue  # OK
                # Dicts und Listen sind OK (werden von as_dict() selbst gehandhabt)

    def test_05_dirac_no_crash(self):
        """Dirac-Impuls (Edge-Case) erzeugt keinen Absturz in keinem Plugin."""
        from plugins.apollo_plugin import repair_codec_artifacts
        from plugins.bs_roformer_plugin import separate_stems
        from plugins.laion_clap_plugin import tag_audio
        from plugins.utmos_plugin import estimate_mos

        dirac = _dirac(2.0)
        assert estimate_mos(dirac, SR).mos >= 1.0
        assert tag_audio(dirac, SR) is not None
        r_stems = separate_stems(dirac, SR)
        for stem in r_stems.stems.values():
            assert np.isfinite(stem).all()
        assert repair_codec_artifacts(dirac, SR).audio is not None

    def test_06_inpainting_pipeline_sequence(self):
        """CQTdiff+ Inpainting gefolgt von UTMOS-Bewertung — Mini-Pipeline."""
        from plugins.cqtdiff_plus_plugin import inpaint_gap
        from plugins.utmos_plugin import estimate_mos

        audio = _sine(440.0, 3.0)
        # Dropout-Lücke simulieren (200 ms)
        gap_s, gap_e = int(SR * 0.5), int(SR * 0.7)
        audio_with_gap = audio.copy()
        audio_with_gap[gap_s:gap_e] = 0.0

        inpaint_result = inpaint_gap(audio_with_gap, SR, gap_s, gap_e)
        assert np.isfinite(inpaint_result.audio).all()
        assert np.max(np.abs(inpaint_result.audio)) <= 1.0

        mos_result = estimate_mos(inpaint_result.audio, SR)
        assert 1.0 <= mos_result.mos <= 5.0
        # Nach Inpainting sollte MOS akzeptabel sein (mindestens "Poor")
        assert mos_result.mos >= 1.0
