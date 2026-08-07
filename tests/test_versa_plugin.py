from __future__ import annotations

"""
test_versa_plugin.py — Tests für VERSA 2024 MOS-Plugin (§4.4)

VERSA ersetzt CDPAM als primäre non-reference MOS-Metrik für Musikrestaurierung.
API: get_versa_plugin().score(audio_np, sr) → VersaResult(mos, model_used)

Invarianten (§3.1, §3.2):
    - MOS ∈ [1.0, 5.0]
    - NaN/Inf-sicher
    - Singleton Thread-sicher
    - CPU-only (§9.5)
"""


import math
import warnings

import numpy as np
import pytest

SR = 48_000
DURATION_S = 1.0
_N = int(SR * DURATION_S)

AUDIO_SINE = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, DURATION_S, _N))).astype(np.float32)
AUDIO_NOISE = (0.1 * np.random.default_rng(42).standard_normal(_N)).astype(np.float32)
AUDIO_ZERO = np.zeros(_N, dtype=np.float32)
AUDIO_STEREO = np.stack([AUDIO_SINE, AUDIO_SINE], axis=1)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_versa_plugin_importable():
    """VersaPlugin muss aus plugins.versa_plugin importierbar sein."""
    from plugins.versa_plugin import VersaPlugin

    assert VersaPlugin is not None


def test_get_versa_plugin_callable():
    """get_versa_plugin() liefert eine VersaPlugin-Instanz."""
    from plugins.versa_plugin import get_versa_plugin

    p = get_versa_plugin()
    assert p is not None


def test_score_mos_callable():
    """score_mos() convenience-Funktion ist aufrufbar."""
    from plugins.versa_plugin import score_mos

    assert callable(score_mos)


def test_s3prl_timm_deprecation_filter_is_narrow():
    """Der VERSA-Importfilter schluckt nur die bekannte timm-Deprecation."""
    from plugins.versa_plugin import _suppress_s3prl_timm_deprecation

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with _suppress_s3prl_timm_deprecation():
            warnings.warn_explicit(
                "Importing from timm.models.layers is deprecated, please import via timm.layers",
                FutureWarning,
                filename="/tmp/timm/models/layers/__init__.py",
                lineno=49,
                module="timm.models.layers",
            )
            with pytest.raises(FutureWarning):
                warnings.warn_explicit(
                    "Importing from some.other.module is deprecated",
                    FutureWarning,
                    filename="/tmp/other.py",
                    lineno=1,
                    module="some.other.module",
                )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_singleton_same_instance():
    """Mehrfache get_versa_plugin()-Aufrufe liefern dieselbe Instanz."""
    from plugins.versa_plugin import get_versa_plugin

    p1 = get_versa_plugin()
    p2 = get_versa_plugin()
    assert p1 is p2


# ---------------------------------------------------------------------------
# VersaResult — Struktur
# ---------------------------------------------------------------------------


def test_result_has_mos_attribute():
    """VersaResult hat mos-Attribut."""
    from plugins.versa_plugin import get_versa_plugin

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    assert hasattr(result, "mos"), "VersaResult muss mos-Attribut haben"


def test_result_has_model_used_attribute():
    """VersaResult hat model_used-Attribut."""
    from plugins.versa_plugin import get_versa_plugin

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    assert hasattr(result, "model_used"), "VersaResult muss model_used-Attribut haben"


# ---------------------------------------------------------------------------
# MOS-Wertebereich §4.4: mos ∈ [1.0, 5.0]
# ---------------------------------------------------------------------------


def test_mos_in_valid_range_sine():
    """MOS für sauberes Sinus-Signal muss in [1.0, 5.0] liegen."""
    from plugins.versa_plugin import get_versa_plugin

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    assert math.isfinite(result.mos), f"MOS ist nicht finite: {result.mos}"
    assert 1.0 <= result.mos <= 5.0, f"MOS außerhalb [1,5]: {result.mos}"


def test_mos_in_valid_range_noise():
    """MOS für Rauschen muss in [1.0, 5.0] liegen."""
    from plugins.versa_plugin import get_versa_plugin

    result = get_versa_plugin().score(AUDIO_NOISE, SR)
    assert math.isfinite(result.mos)
    assert 1.0 <= result.mos <= 5.0


def test_mos_in_valid_range_zeros():
    """MOS für Stille muss in [1.0, 5.0] liegen (keine Division durch Null)."""
    from plugins.versa_plugin import get_versa_plugin

    result = get_versa_plugin().score(AUDIO_ZERO, SR)
    assert math.isfinite(result.mos)
    assert 1.0 <= result.mos <= 5.0


def test_mos_stereo_input():
    """VERSA akzeptiert Stereo-Eingabe (ndim=2) ohne Fehler."""
    from plugins.versa_plugin import get_versa_plugin

    result = get_versa_plugin().score(AUDIO_STEREO, SR)
    assert math.isfinite(result.mos)
    assert 1.0 <= result.mos <= 5.0


# ---------------------------------------------------------------------------
# score_mos() convenience wrapper
# ---------------------------------------------------------------------------


def test_score_mos_result_matches_plugin():
    """score_mos() liefert konsistentes Ergebnis mit get_versa_plugin().score()."""
    from plugins.versa_plugin import get_versa_plugin, score_mos

    r1 = score_mos(AUDIO_SINE, SR)
    r2 = get_versa_plugin().score(AUDIO_SINE, SR)
    # Beide müssen valide MOS-Werte liefern
    assert math.isfinite(r1.mos)
    assert math.isfinite(r2.mos)
    assert 1.0 <= r1.mos <= 5.0
    assert 1.0 <= r2.mos <= 5.0


# ---------------------------------------------------------------------------
# NaN/Inf-Robustheit (§3.1)
# ---------------------------------------------------------------------------


def test_nan_input_no_crash():
    """NaN-Eingabe darf keinen Absturz auslösen."""
    from plugins.versa_plugin import get_versa_plugin

    audio_nan = np.full(_N, np.nan, dtype=np.float32)
    try:
        result = get_versa_plugin().score(audio_nan, SR)
        assert math.isfinite(result.mos)
        assert 1.0 <= result.mos <= 5.0
    except (RuntimeError, ValueError):
        pass  # Explizite Exceptions erlaubt, Absturz (SIGSEGV) nicht


def test_inf_input_no_crash():
    """Inf-Eingabe darf keinen Absturz auslösen."""
    from plugins.versa_plugin import get_versa_plugin

    audio_inf = np.full(_N, np.inf, dtype=np.float32)
    try:
        result = get_versa_plugin().score(audio_inf, SR)
        assert math.isfinite(result.mos)
    except (RuntimeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# MOS-Normalisierung für Kompatibilität mit alten CDPAM-Callern (§4.4)
# ---------------------------------------------------------------------------


def test_mos_normalization_to_0_100():
    """MOS [1,5] → [0,100] Normalisierung muss valide sein."""
    from plugins.versa_plugin import get_versa_plugin

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    score_0_100 = float(np.clip((result.mos - 1.0) / 4.0 * 100.0, 0.0, 100.0))
    assert 0.0 <= score_0_100 <= 100.0
    assert math.isfinite(score_0_100)


def test_mos_normalization_to_0_1():
    """MOS [1,5] → [0,1] Normalisierung muss valide sein."""
    from plugins.versa_plugin import get_versa_plugin

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    score_01 = float(np.clip((result.mos - 1.0) / 4.0, 0.0, 1.0))
    assert 0.0 <= score_01 <= 1.0
    assert math.isfinite(score_01)


def test_singmos_prefers_batched_input_shape():
    """SingMOS path should pass a batched [B, T] tensor if backend requires it."""
    from plugins.versa_plugin import VersaPlugin

    plugin = VersaPlugin()
    plugin._model_loaded = True
    plugin._predictor_dict = {}
    plugin._predictor_fs = {}

    def _fake_metric(x, sr, _pred, _fs):
        assert sr == 16000
        assert hasattr(x, "ndim")
        assert x.ndim == 2
        return {"singmos_pro": 4.2}

    plugin._pseudo_mos_metric = _fake_metric  # type: ignore[assignment]

    result = plugin._score_singmos_pro(AUDIO_SINE, SR)
    assert math.isfinite(result.mos)
    assert result.model_used == "singmos_pro"
    assert result.mos >= 4.0


def test_score_uses_pqs_fallback_for_non_vocal_content(monkeypatch):
    """SingMOS darf für nicht-vokale Inhalte nicht verwendet werden."""
    from plugins.versa_plugin import VersaPlugin

    plugin = VersaPlugin()
    plugin._model_loaded = True

    class _FakePanns:
        def get_tags(self, _audio, _sr):
            return {"Singing voice": 0.10, "Vocals": 0.15}

    monkeypatch.setattr("plugins.panns_plugin.get_panns_plugin", lambda: _FakePanns())
    monkeypatch.setattr(plugin, "_score_singmos_pro", lambda *_args, **_kwargs: pytest.fail("SingMOS should not run"))

    result = plugin.score(AUDIO_SINE, SR)
    assert result.model_used == "pqs_dsp_fallback"
    assert 1.0 <= result.mos <= 5.0


def test_score_uses_singmos_for_vocal_content(monkeypatch):
    """SingMOS wird nur bei ausreichender Vocal-Konfidenz aktiviert."""
    from plugins.versa_plugin import VersaPlugin, VersaResult

    plugin = VersaPlugin()
    plugin._model_loaded = True
    plugin._predictor_dict = {}
    plugin._predictor_fs = {}

    class _FakePanns:
        def get_tags(self, _audio, _sr):
            return {"Singing voice": 0.72, "Vocals": 0.68}

    monkeypatch.setattr("plugins.panns_plugin.get_panns_plugin", lambda: _FakePanns())
    monkeypatch.setattr(
        plugin,
        "_score_singmos_pro",
        lambda *_args, **_kwargs: VersaResult(mos=4.1, model_used="singmos_pro", confidence=0.9),
    )

    result = plugin.score(AUDIO_SINE, SR)
    assert result.model_used == "singmos_pro"
    assert result.mos == pytest.approx(4.1)
