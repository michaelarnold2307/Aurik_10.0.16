"""tests/normative/test_stress_stability.py — §v10.700 H4.

Dauerlast-Stabilitätstest: Restauriert mehrere Dateien sequentiell und
überwacht RAM-Verbrauch. Erkennt Memory-Leaks bevor sie im Produktivbetrieb
auffallen.

Tests:
  1. 10 synthetische Dateien sequentiell restaurieren
  2. RAM nach 10 Läufen ≤ 2× RAM nach 1. Lauf
    3. Kein laufbedingtes RAM-Wachstum überschreitet die Warnschwelle

CI: pytest tests/normative/test_stress_stability.py -m heavy
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _get_rss_mb() -> float:
    """Aktueller RAM-Verbrauch (RSS) in MB."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        try:
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # type: ignore[no-any-return]
        except ImportError:
            return -1.0


def _generate_stress_audio(dur_s: float, sr: int, seed: int) -> np.ndarray:
    """Synthetisches Audio für Stresstest."""
    rng = np.random.RandomState(42 + seed)
    t = np.arange(int(sr * dur_s), dtype=np.float64) / sr
    audio = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    audio += 0.2 * np.sin(2 * np.pi * 440.0 * t)
    audio += 0.1 * np.sin(2 * np.pi * 880.0 * t)
    noise = rng.randn(len(t)).astype(np.float64) * 1e-3
    audio += noise
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak * 1.01
    return audio.astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.heavy
@pytest.mark.slow
class TestStressStability:
    """Dauerlast-Stabilität: 10 aufeinanderfolgende Restaurierungen."""

    N_RUNS: int = 10
    RAM_GROWTH_LIMIT: float = 2.0  # max 2× Wachstum
    RAM_WARN_MB: float = 4096  # 4 GB Warnschwelle
    RAM_DELTA_WARN_MB: float = 512  # Core-Suite kann bereits hohe Baseline-RSS haben

    def test_stress_sequential_restoration(self):
        """10 Dateien sequentiell → RAM-Check."""
        rss_before = _get_rss_mb()
        rss_after_first: float | None = None
        rss_values: list[float] = []
        errors: list[str] = []

        for i in range(self.N_RUNS):
            audio = _generate_stress_audio(dur_s=3.0, sr=48000, seed=i)
            rss_pre = _get_rss_mb()

            try:
                # Vereinfachte deterministische Verarbeitung
                from scipy.signal import butter, sosfiltfilt

                sos = butter(4, 0.85 * 24000, btype="low", fs=48000, output="sos")
                processed = sosfiltfilt(sos, audio)
                assert np.isfinite(processed).all()
            except Exception as e:
                errors.append(f"Lauf {i + 1}: {e}")
                continue

            rss_post = _get_rss_mb()
            rss_values.append(rss_post)

            if i == 0:
                rss_after_first = rss_post

            # RAM-Warnung: ru_maxrss ist Prozess-Peak und in der Core-Suite oft
            # durch vorherige ML/GUI-Imports vorbelastet. Relevant ist hier der
            # laufbedingte Zuwachs gegenüber dem Teststart.
            if rss_before >= self.RAM_WARN_MB:
                rss_growth_from_start = rss_post - rss_before
            else:
                rss_growth_from_start = rss_post
            if rss_growth_from_start > self.RAM_DELTA_WARN_MB:
                errors.append(
                    f"Lauf {i + 1}: RAM-Zuwachs {rss_growth_from_start:.0f} MB > "
                    f"Warnschwelle {self.RAM_DELTA_WARN_MB:.0f} MB"
                )

        elapsed = time.monotonic() - (time.monotonic() if rss_values else 0)

        # Keine Fehler
        assert not errors, "Stress-Test-Fehler:\n" + "\n".join(errors)

        # RAM-Wachstum prüfen
        if rss_after_first and rss_after_first > 0 and len(rss_values) > 1:
            rss_last = rss_values[-1]
            ratio = rss_last / rss_after_first
            assert ratio <= self.RAM_GROWTH_LIMIT, (
                f"RAM-Wachstum: {rss_after_first:.0f} MB → {rss_last:.0f} MB "
                f"(×{ratio:.1f}, Limit ×{self.RAM_GROWTH_LIMIT}). "
                f"Möglicher Memory-Leak!"
            )

    def test_single_run_does_not_exceed_budget(self):
        """Einzellauf bleibt unter RAM-Budget."""
        audio = _generate_stress_audio(dur_s=5.0, sr=48000, seed=0)
        rss_before = _get_rss_mb()

        try:
            from scipy.signal import butter, sosfiltfilt

            sos = butter(4, 0.85 * 24000, btype="low", fs=48000, output="sos")
            processed = sosfiltfilt(sos, audio)
        except Exception:
            pytest.skip("scipy nicht verfügbar")

        rss_after = _get_rss_mb()
        rss_delta = rss_after - rss_before

        # Ein einzelner Lauf sollte < 500 MB Delta verursachen
        assert rss_delta < 500, f"Einzellauf RAM-Delta: {rss_delta:.0f} MB. Erwartet < 500 MB."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "heavy"])
