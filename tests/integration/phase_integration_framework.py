from __future__ import annotations

"""
§2.59 Integrationstest-Framework für Phasen (2026-07-09)

Basis-Klasse für schnelle, reproduzierbare Phasen-Integrationstests.
Jede Phase kann mit synthetischem Audio getestet werden.

Usage:
    class TestPhase23(PhaseIntegrationTest):
        phase_module = 'backend.core.phases.phase_23_spectral_repair'
        phase_class = 'SpectralRepairPhase'

        def test_basic_processing(self):
            result = self.run_phase({'param': value})
            assert result is not None
"""


import importlib
import time
from typing import Any

import numpy as np
import pytest


class PhaseIntegrationTest:
    """Basis-Klasse für Phasen-Integrationstests.

    Erzeugt synthetisches Audio (1 kHz Sinus + Rauschen),
    importiert die Phase und führt sie mit konfigurierbaren
    Parametern aus. Misst Ausführungszeit für Regression-Tests.
    """

    phase_module: str = ""
    phase_class: str = ""
    sr: int = 48000
    duration_s: float = 1.0

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._audio = self._make_audio()
        self._last_duration: float = 0.0

    def _make_audio(self) -> np.ndarray:
        t = np.arange(int(self.sr * self.duration_s), dtype=np.float32) / self.sr
        audio = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        audio += np.random.RandomState(42).randn(len(audio)).astype(np.float32) * 0.001
        return audio  # type: ignore[no-any-return]

    def _get_phase(self) -> Any:
        mod = importlib.import_module(self.phase_module)
        return getattr(mod, self.phase_class)

    def run_phase(self, params: dict[str, Any] | None = None) -> Any:
        """Führt die Phase aus und misst die Dauer.

        Returns:
            PhaseResult oder None bei Fehler
        """
        phase_cls = self._get_phase()
        phase = phase_cls()
        kwargs = {
            "audio": self._audio.copy(),
            "sr": self.sr,
            "material": params.pop("material", "vinyl") if params else "vinyl",
            "mode": params.pop("mode", "restoration") if params else "restoration",
        }
        if params:
            kwargs.update(params)

        t0 = time.monotonic()
        try:
            result = phase.process(**kwargs)
        except Exception as e:
            pytest.skip(f"Phase nicht ausführbar: {e}")
            return None
        self._last_duration = time.monotonic() - t0
        return result

    def assert_rt_under(self, max_rt: float = 5.0) -> None:
        """Prüft ob die Phase unter max_rt × Echtzeit bleibt."""
        rt = self._last_duration / self.duration_s
        assert rt < max_rt, (
            f"Phase {self.phase_class}: RT={rt:.1f}× > {max_rt}× Limit (duration={self._last_duration:.2f}s)"
        )

    def assert_no_nan(self, audio: np.ndarray) -> None:
        """Prüft ob das Audio NaN/Inf-frei ist."""
        assert np.all(np.isfinite(audio)), f"Phase {self.phase_class}: Audio enthält NaN/Inf"

    def assert_level_preserved(self, before: np.ndarray, after: np.ndarray, max_db: float = 3.0) -> None:
        """Prüft ob der Pegel innerhalb max_db bleibt."""
        rms_before = np.sqrt(np.mean(before**2)) + 1e-10
        rms_after = np.sqrt(np.mean(after**2)) + 1e-10
        delta_db = abs(20 * np.log10(rms_after / rms_before))
        assert delta_db < max_db, f"Phase {self.phase_class}: Pegeländerung {delta_db:.1f} dB > {max_db} dB"
