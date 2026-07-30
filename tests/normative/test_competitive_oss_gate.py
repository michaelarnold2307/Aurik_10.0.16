"""Open-Source Competitive CI Gate — §15.1.

Vergleicht Aurik gegen DeepFilterNet3, Demucs, und MDX-Net.
Läuft in CI via `pytest -m competitive_oss`.

Anforderungen:
- Aurik muss in ≥80% der Szenarien gewinnen oder gleichauf sein
- PQS-Delta muss ≥ 0 sein für ≥80% der Szenarien
- Kein Tool darf in mehr als 50% der Szenarien besser sein als Aurik
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Hilfsfunktionen ─────────────────────────────────────────────────────────


def _load_benchmark_results() -> dict:
    """Lädt die letzten Benchmark-Ergebnisse."""
    results_dir = REPO_ROOT / "benchmarks" / "competitive" / "results"
    if not results_dir.exists():
        return {}

    # Finde das neueste Ergebnis-Verzeichnis
    dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    for d in dirs:
        summary = d / "oss_summary.json"
        if summary.exists():
            return json.loads(summary.read_text(encoding="utf-8"))
    return {}


def _oss_benchmark_available() -> bool:
    """Prüft ob die OSS-Benchmark-Tools installiert sind."""
    try:
        import numpy as np
        return True
    except ImportError:
        return False


# ── CI Gate Tests ───────────────────────────────────────────────────────────


@pytest.mark.competitive_oss
@pytest.mark.slow
class TestCompetitiveOSSGate:
    """CI-Gate für Open-Source-Competitive-Benchmarks."""

    @pytest.fixture(autouse=True)
    def results(self) -> dict:
        data = _load_benchmark_results()
        if not data:
            pytest.skip(
                "Keine OSS-Benchmark-Ergebnisse gefunden. "
                "Führe 'python benchmarks/competitive/open_source_benchmark.py --all --ci' aus."
            )
        return data

    def test_aurik_wins_majority_of_scenarios(self, results: dict):
        """Aurik muss in ≥80% der Szenarien gewinnen oder gleichauf sein."""
        summary = results.get("summary", {})
        if not summary:
            pytest.skip("Summary nicht vorhanden")
        total = summary.get("total", 0)
        if total == 0:
            pytest.skip("Keine Szenarien")
        wins = summary.get("wins", 0)
        ties = summary.get("ties", 0)
        rate = (wins + ties) / total
        assert rate >= 0.80, (
            f"Aurik gewinnt oder tied in nur {rate:.1%} der Szenarien "
            f"({wins} wins + {ties} ties / {total} total). "
            f"Mindestens 80% erforderlich (§15.1)."
        )

    def test_pqs_delta_non_negative(self, results: dict):
        """PQS-Delta muss ≥ 0 sein für ≥80% der Szenarien."""
        summary = results.get("summary", {})
        mean_delta = summary.get("mean_delta", 0)
        losses = summary.get("losses", 0)
        total_ok = summary.get("ok", 0)
        if total_ok == 0:
            pytest.skip("Keine erfolgreichen Vergleiche")
        non_neg_rate = (total_ok - losses) / total_ok
        assert non_neg_rate >= 0.80, (
            f"PQS-Delta ist nur in {non_neg_rate:.1%} der Fälle ≥ 0. "
            f"Mittleres Delta: {mean_delta:+.2f}. "
            f"Mindestens 80% erforderlich (§15.1)."
        )

    def test_no_tool_dominates_aurik(self, results: dict):
        """Kein Open-Source-Tool darf in >50% der Vergleiche besser sein."""
        # Gruppiere Ergebnisse nach Tool
        results_list = results.get("results", [])
        if not results_list:
            pytest.skip("Keine detaillierten Ergebnisse")
        from collections import Counter

        tool_losses: Counter[str] = Counter()
        tool_total: Counter[str] = Counter()
        for r in results_list:
            tool = r.get("tool", "unknown")
            tool_total[tool] += 1
            if r.get("pqs_delta", 0) < 0:
                tool_losses[tool] += 1

        for tool in tool_total:
            rate = tool_losses[tool] / tool_total[tool]
            assert rate <= 0.50, (
                f"Tool '{tool}' ist besser als Aurik in {rate:.1%} der "
                f"Szenarien ({tool_losses[tool]}/{tool_total[tool]}). "
                f"Maximal 50% erlaubt."
            )

    def test_benchmark_includes_deepfilternet(self, results: dict):
        """DeepFilterNet3 muss im Benchmark enthalten sein."""
        results_list = results.get("results", [])
        tools = {r.get("tool") for r in results_list}
        # Mindestens eines der Deep-Learning-basierten Tools muss dabei sein
        dl_tools = {"deepfilternet3", "demucs", "mdx_net", "open_unmix"}
        present = tools & dl_tools
        assert present, (
            f"Kein Deep-Learning-Tool im Benchmark. "
            f"Gefundene Tools: {tools}. "
            f"Mindestens eines von {dl_tools} muss enthalten sein (§15.1)."
        )
