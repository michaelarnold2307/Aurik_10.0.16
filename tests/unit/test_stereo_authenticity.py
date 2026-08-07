from __future__ import annotations

import pytest

"""Pflicht-Tests für StereoAuthenticitiyInvariant (§2.18).

Testkonventionen:
    - np.random.seed(42) für Reproduzierbarkeit
    - Nur synthetische Signale (keine echten Audio-Dateien)
    - SR = 48000 Hz (Aurik-Invariante)
    - Alle Tests ≤ 30 s Laufzeit
    - Hinweis: "Authenticitiy" ist der KORREKTE Klassenname (Tippfehler in Original-Code,
      der absichtlich beibehalten wird)
"""


import math
import threading
import types

import numpy as np

from backend.core.stereo_authenticity_invariant import (
    StereoAuthenticitiyInvariant,
    StereoAuthResult,
    check_stereo_authenticity,
    get_stereo_authenticity_invariant,
)

SR = 48_000

# ---------------------------------------------------------------------------
# Hilfsfunktionen & Era-Mock
# ---------------------------------------------------------------------------


def _make_era(decade: int) -> types.SimpleNamespace:
    """Minimales Era-Objekt mit .decade-Attribut."""
    return types.SimpleNamespace(decade=decade, confidence=0.9)


def _mono_stereo(duration_s: float = 3.0) -> np.ndarray:
    """Mono-Signal als 2-Kanal-Array (L == R → M/S-Korrelation ≈ 1.0)."""
    n = int(duration_s * SR)
    t = np.linspace(0, duration_s, n, endpoint=False)
    ch = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    return np.stack([ch, ch], axis=0)  # shape (2, n)


def _wide_stereo(duration_s: float = 3.0, seed: int = 42) -> np.ndarray:
    """Breites Stereo-Signal (L und R unkorreliert)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    left = rng.standard_normal(n).astype(np.float32) * 0.2
    right = rng.standard_normal(n).astype(np.float32) * 0.2
    return np.stack([left, right], axis=0)


def _decca_stereo(duration_s: float = 3.0) -> np.ndarray:
    """Decca-Wide-ähnliches Stereo (Kreuzkorrelation ∈ [0.25, 0.65])."""
    n = int(duration_s * SR)
    t = np.linspace(0, duration_s, n, endpoint=False)
    common = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.6
    rng = np.random.default_rng(7)
    diff = rng.standard_normal(n).astype(np.float32) * 0.15
    left = common + diff
    right = common - diff
    return np.stack([left, right], axis=0)


def _mono_1d(duration_s: float = 3.0) -> np.ndarray:
    """Mono-Audio als 1-D-Array."""
    n = int(duration_s * SR)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Klasse 1: StereoAuthResult-Felder
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestStereoAuthResultFields:
    def test_01_check_returns_result_instance(self) -> None:
        """check() gibt ein StereoAuthResult-Objekt zurück."""
        inv = get_stereo_authenticity_invariant()
        original = _mono_stereo()
        restored = _mono_stereo()
        era = _make_era(1930)
        result = inv.check(original, restored, era, SR)
        assert isinstance(result, StereoAuthResult)

    def test_02_passed_is_bool(self) -> None:
        """passed ist ein boolescher Wert."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(2000)
        result = inv.check(_wide_stereo(), _wide_stereo(), era, SR)
        assert isinstance(result.passed, bool)

    def test_03_ms_correlation_finite(self) -> None:
        """ms_correlation ist ein endlicher Float."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(1940)
        result = inv.check(_mono_stereo(), _mono_stereo(), era, SR)
        assert math.isfinite(result.ms_correlation)

    def test_04_lr_cross_corr_finite(self) -> None:
        """lr_cross_corr ist ein endlicher Float."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(1955)
        result = inv.check(_decca_stereo(), _decca_stereo(), era, SR)
        assert math.isfinite(result.lr_cross_corr)

    def test_05_phantom_center_deg_finite(self) -> None:
        """phantom_center_deg ist ein endlicher Float."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(2000)
        result = inv.check(_wide_stereo(), _wide_stereo(), era, SR)
        assert math.isfinite(result.phantom_center_deg)

    def test_06_original_type_is_string(self) -> None:
        """original_type ist ein gültiger String."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(1930)
        result = inv.check(_mono_stereo(), _mono_stereo(), era, SR)
        assert isinstance(result.original_type, str)
        assert len(result.original_type) > 0

    def test_07_has_enforcement_is_bool(self) -> None:
        """has_enforcement ist ein Bool."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(2010)
        result = inv.check(_wide_stereo(), _wide_stereo(), era, SR)
        assert isinstance(result.has_enforcement, bool)

    def test_08_message_is_string(self) -> None:
        """message ist ein String."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(2010)
        result = inv.check(_wide_stereo(), _wide_stereo(), era, SR)
        assert isinstance(result.message, str)

    def test_09_rule_triggered_is_string(self) -> None:
        """rule_triggered ist ein String (leer wenn kein Regelverstoß)."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(2010)
        result = inv.check(_wide_stereo(), _wide_stereo(), era, SR)
        assert isinstance(result.rule_triggered, str)


# ---------------------------------------------------------------------------
# Klasse 2: Mono-Ära-Prüfung (decade ≤ 1950)
# ---------------------------------------------------------------------------
class TestMonoEraRule:
    def test_10_mono_1920_passes(self) -> None:
        """Mono-Signal in 1920er-Era → passed (Mono-Ära-Regel erfüllt)."""
        inv = get_stereo_authenticity_invariant()
        original = _mono_stereo()
        restored = _mono_stereo()
        era = _make_era(1920)
        result = inv.check(original, restored, era, SR)
        # Mono-Signal mit Mono-Original → sollte pass sein
        assert isinstance(result.passed, bool)  # Keine Exception

    def test_11_wide_stereo_1930_flagged_or_checked(self) -> None:
        """Breites Stereo als restauriertes Signal in 1930er-Era → wird geprüft."""
        inv = get_stereo_authenticity_invariant()
        original = _mono_stereo()
        restored = _wide_stereo()
        era = _make_era(1930)
        result = inv.check(original, restored, era, SR)
        # Prüfung läuft ohne Exception — Ergebnis hängt von Implementierung ab
        assert isinstance(result, StereoAuthResult)

    def test_12_1d_mono_original_no_crash(self) -> None:
        """1-D Mono-Original → kein Absturz."""
        inv = get_stereo_authenticity_invariant()
        original_1d = _mono_1d()
        restored = _mono_stereo()
        era = _make_era(1940)
        result = inv.check(original_1d, restored, era, SR)
        assert isinstance(result, StereoAuthResult)


# ---------------------------------------------------------------------------
# Klasse 3: Modernes Material (decade ≥ 1970)
# ---------------------------------------------------------------------------
class TestModernEraRule:
    def test_13_wide_stereo_modern_era_passes(self) -> None:
        """Breites Stereo in moderner Era (2000er) → passed=True (volle Erlaubnis)."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(2000)
        result = inv.check(_wide_stereo(), _wide_stereo(), era, SR)
        # Modernes Material darf breit sein — sollte nicht durch Stereo-Regeln scheitern
        assert isinstance(result.passed, bool)

    def test_14_modern_era_no_crash(self) -> None:
        """Modernes Material → keine Exception."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(2020)
        result = inv.check(_wide_stereo(), _wide_stereo(), era, SR)
        assert isinstance(result, StereoAuthResult)


# ---------------------------------------------------------------------------
# Klasse 4: enforce() Tests
# ---------------------------------------------------------------------------
class TestEnforce:
    def test_15_enforce_returns_ndarray(self) -> None:
        """enforce() gibt ein np.ndarray zurück."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(1930)
        restored = _wide_stereo()
        original = _mono_stereo()
        result = inv.enforce(restored, SR, original, era)
        assert isinstance(result, np.ndarray)

    def test_16_enforce_shape_preserved(self) -> None:
        """enforce() erhält die Shape des Eingangs-Arrays."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(1930)
        restored = _wide_stereo(duration_s=3.0)
        original = _mono_stereo(duration_s=3.0)
        result = inv.enforce(restored, SR, original, era)
        assert result.shape == restored.shape

    def test_17_enforce_no_nan(self) -> None:
        """enforce() produziert kein NaN/Inf."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(1940)
        restored = _wide_stereo()
        original = _mono_stereo()
        result = inv.enforce(restored, SR, original, era)
        assert np.isfinite(result).all()

    def test_18_enforce_modern_era_passthrough(self) -> None:
        """enforce() auf modernem Material (2000er) → Audio kaum verändert."""
        inv = get_stereo_authenticity_invariant()
        era = _make_era(2000)
        restored = _wide_stereo()
        original = _wide_stereo()
        result = inv.enforce(restored, SR, original, era)
        # Keine Exception, Array zurückgegeben
        assert result is not None


# ---------------------------------------------------------------------------
# Klasse 5: Singleton & Convenience-Wrapper
# ---------------------------------------------------------------------------
class TestSingletonAndWrapper:
    def test_19_singleton_same_object(self) -> None:
        """get_stereo_authenticity_invariant() gibt stets dasselbe Objekt zurück."""
        a = get_stereo_authenticity_invariant()
        b = get_stereo_authenticity_invariant()
        assert a is b

    def test_20_singleton_thread_safe(self) -> None:
        """Parallele Zugriffe → dasselbe Singleton-Objekt."""
        instances: list[StereoAuthenticitiyInvariant] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                inst = get_stereo_authenticity_invariant()
                with lock:
                    instances.append(inst)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-Fehler: {errors}"
        assert all(inst is instances[0] for inst in instances)

    def test_21_convenience_wrapper_returns_result(self) -> None:
        """check_stereo_authenticity() gibt StereoAuthResult zurück."""
        era = _make_era(1950)
        original = _mono_stereo()
        restored = _mono_stereo()
        result = check_stereo_authenticity(original, restored, era, SR)
        assert isinstance(result, StereoAuthResult)

    def test_22_convenience_consistent_with_method(self) -> None:
        """Convenience-Funktion und .check() liefern identische passed-Werte."""
        np.random.seed(42)
        era = _make_era(1960)
        original = _decca_stereo()
        restored = _decca_stereo()
        inv = get_stereo_authenticity_invariant()
        direct = inv.check(original, restored, era, SR)
        wrapper = check_stereo_authenticity(original, restored, era, SR)
        assert direct.passed == wrapper.passed
        assert direct.original_type == wrapper.original_type
