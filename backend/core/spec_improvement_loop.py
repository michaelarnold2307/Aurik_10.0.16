"""
spec_improvement_loop.py — v10 Continuous Spec Improvement
===========================================================

Kernprinzip: Specs definieren das MINIMUM. Code definiert das MAXIMUM.
Wenn Code die Specs uebertrifft, muessen die Specs angehoben werden.

Drei Phasen:
  PHASE 1 — Spec-Code Comparator:
    Misst, ob die Code-Implementierung BESSER ist als die Specs fordern.
    "Compliant" = Code erreicht Spec-Ziel.
    "Exceeds"   = Code uebertrifft Spec-Ziel signifikant (>5% besser).

  PHASE 2 — Improvement Detector:
    Wenn Code ueber mehrere Runs konsistent besser ist als die Specs,
    wird ein Spec-Upgrade vorgeschlagen. Schwellwert: 3 konsekutive
    Laueufe mit >5% Uebererfuellung.

  PHASE 3 — Spec Auditor:
    Nach jedem Spec-Upgrade (oder manuell ausgeloest): Prueft ALLE
    Spec-Dateien auf:
      - Interne Konsistenz (keine widerspruechlichen Schwellwerte)
      - Cross-Spec-Kompatibilitaet (Spec 01 widerspricht nicht Spec 04)
      - Gap-Erkennung (fehlende Spec fuer existierende Funktionalitaet)
      - Regressions-Schutz (Spec-Aenderung verschlechtert nichts)

Integration:
  - Wird nach jedem Pipeline-Lauf vom Watchdog aufgerufen
  - Ergebnisse gehen in metadata["pipeline_guard"]["spec_improvement"]
  - Pre-Commit-Hook prueft Spec-Konsistenz
  - KI-Agenten nutzen die Verbesserungsvorschlaege

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Konstanten
# ═══════════════════════════════════════════════════════════════════════════════

# Wie viel besser muss Code sein, um als "exceeds" zu gelten?
EXCEEDS_THRESHOLD: float = 0.05  # 5% ueber Spec-Ziel

# Wie viele konsekutive Laueufe mit Uebererfuellung fuer Spec-Upgrade?
CONSECUTIVE_EXCEEDS_FOR_UPGRADE: int = 3

# Minimale Verbesserung fuer Upgrade-Vorschlag
MIN_UPGRADE_DELTA: float = 0.02  # 2% absoluter Anstieg

# Spec-Dateien die geprueft werden
SPEC_DIR: str = ".github/specs"
SPEC_FILES: list[str] = [
    "01_musical_goals.md",
    "02_pipeline_architecture.md",
    "04_dsp_standards.md",
    "05_material_system.md",
    "06_phases_system.md",
    "10_bug_gap_strategy.md",
    "13_human_ear_quality.md",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Datenstrukturen
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SpecMetric:
    """Eine einzelne spezifizierte Metrik mit Soll- und Ist-Wert."""

    name: str
    spec_target: float  # Was die Spec fordert
    code_achieved: float  # Was der Code tatsaechlich erreicht
    unit: str = ""  # z.B. "dB", "score", "%"
    direction: str = "higher"  # "higher" = mehr ist besser, "lower" = weniger ist besser
    exceeds: bool = False  # Code > Spec (um EXCEEDS_THRESHOLD)
    exceeds_by: float = 0.0  # Absoluter Delta
    source: str = ""  # Wo wurde gemessen (z.B. "pipeline_guard.post_flight")


@dataclass
class SpecComparisonResult:
    """Ergebnis eines Spec-vs-Code-Vergleichs."""

    timestamp: float = field(default_factory=time.time)
    material: str = "unknown"
    metrics: list[SpecMetric] = field(default_factory=list)
    compliant_count: int = 0  # Code erreicht Spec
    exceeds_count: int = 0  # Code uebertrifft Spec
    below_count: int = 0  # Code unter Spec
    overall_grade: str = ""  # "A" (ueberall exceeds), "B" (compliant+), "C" (compliant), "F" (below)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ImprovementProposal:
    """Vorschlag zur Spec-Anhebung basierend auf Code-Ueberlegenheit."""

    metric_name: str
    current_spec: float
    proposed_spec: float
    evidence_runs: int  # Anzahl konsekutiver Laueufe mit Uebererfuellung
    avg_achieved: float  # Durchschnittlicher erreichter Wert
    confidence: float  # 0-1, wie sicher ist die Verbesserung
    recommendation: str


@dataclass
class SpecAuditResult:
    """Ergebnis eines vollstaendigen Spec-Audits."""

    files_checked: int
    contradictions: list[tuple[str, str, str]]  # (file1, file2, description)
    gaps: list[str]  # Fehlende Specs
    regressions: list[str]  # Verschlechterungen
    inconsistencies: list[str]  # Interne Widersprueche
    improvement_proposals: list[ImprovementProposal]
    overall_health: str  # "healthy", "needs_attention", "critical"
    recommendations: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Spec-Code Comparator
# ═══════════════════════════════════════════════════════════════════════════════


def compare_spec_vs_code(
    achieved_metrics: dict[str, float],
    material: str = "unknown",
    *,
    spec_targets: dict[str, float] | None = None,
) -> SpecComparisonResult:
    """Vergleicht tatsaechlich erreichte Metriken mit Spec-Vorgaben.

    Args:
        achieved_metrics: dict von Metrik-Name → erreichter Wert
        material: Material-Typ (fuer material-adaptive Schwellen)
        spec_targets: Optionale override fuer Spec-Ziele (sonst aus Constitution)

    Returns:
        SpecComparisonResult mit Exceeds/Compliant/Below-Analyse
    """
    if spec_targets is None:
        from backend.core.spec_constitution import get_constitution

        const = get_constitution()
        spec_targets = const.get_musical_goal_thresholds(material)

    result = SpecComparisonResult(material=material)
    compliant = 0
    exceeds = 0
    below = 0

    for name, achieved in achieved_metrics.items():
        if name not in spec_targets:
            continue  # Kein Spec-Ziel definiert → Gap!

        target = spec_targets[name]
        delta = achieved - target
        delta_pct = delta / max(abs(target), 0.01)

        is_exceed = delta_pct > EXCEEDS_THRESHOLD
        is_compliant = delta >= 0 and not is_exceed
        is_below = delta < 0

        metric = SpecMetric(
            name=name,
            spec_target=target,
            code_achieved=achieved,
            unit="score",
            direction="higher",
            exceeds=is_exceed,
            exceeds_by=delta if is_exceed else 0.0,
            source="pipeline_run",
        )
        result.metrics.append(metric)

        if is_exceed:
            exceeds += 1
            result.recommendations.append(
                f"UPGRADE {name}: Code erreicht {achieved:.3f}, "
                f"Spec fordert nur {target:.3f} (Δ=+{delta:+.3f}, +{delta_pct:.0%})"
            )
        elif is_compliant:
            compliant += 1
        elif is_below:
            below += 1
            result.recommendations.append(
                f"FIX {name}: Code erreicht nur {achieved:.3f}, Spec fordert {target:.3f} (Δ={delta:+.3f})"
            )

    result.compliant_count = compliant
    result.exceeds_count = exceeds
    result.below_count = below

    total = compliant + exceeds + below
    if total == 0:
        result.overall_grade = "N/A"
    elif below == 0 and exceeds == total:
        result.overall_grade = "A"  # Alles ueber Spec
    elif below == 0 and exceeds > 0:
        result.overall_grade = "B"  # Compliant+ mit Exceeds
    elif below == 0:
        result.overall_grade = "C"  # Genau compliant
    else:
        result.overall_grade = "F"  # Unter Spec

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Improvement Detector
# ═══════════════════════════════════════════════════════════════════════════════


class ImprovementDetector:
    """Erkennt konsistente Code-Ueberlegenheit und schlaegt Spec-Upgrades vor.

    Speichert Metriken ueber mehrere Runs und loest Upgrade-Vorschlag aus,
    wenn CONSECUTIVE_EXCEEDS_FOR_UPGRADE konsekutive Laueufe eine Metrik
    uebererfuellen.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, list[float]] = {}  # metric_name → [values over runs]
        self._run_count: int = 0

    def record_run(self, comparison: SpecComparisonResult) -> list[ImprovementProposal]:
        """Zeichnet einen Lauf auf und gibt Upgrade-Vorschlaege zurueck."""
        proposals: list[ImprovementProposal] = []

        with self._lock:
            self._run_count += 1
            for metric in comparison.metrics:
                if metric.name not in self._history:
                    self._history[metric.name] = []
                self._history[metric.name].append(metric.code_achieved)

                # Nur die letzten N Runs behalten
                if len(self._history[metric.name]) > CONSECUTIVE_EXCEEDS_FOR_UPGRADE * 3:
                    self._history[metric.name] = self._history[metric.name][-CONSECUTIVE_EXCEEDS_FOR_UPGRADE:]

            # Pruefe auf konsekutive Uebererfuellung
            for metric in comparison.metrics:
                if not metric.exceeds:
                    continue

                recent = self._history.get(metric.name, [])[-CONSECUTIVE_EXCEEDS_FOR_UPGRADE:]
                if len(recent) < CONSECUTIVE_EXCEEDS_FOR_UPGRADE:
                    continue

                # Alle letzten N Runs muessen ueber dem Spec-Ziel liegen
                all_exceed = all(v > metric.spec_target * (1.0 + EXCEEDS_THRESHOLD) for v in recent)
                if not all_exceed:
                    continue

                avg_achieved = float(np.mean(recent))
                proposed = avg_achieved * 0.95  # 5% Sicherheitsmarge unter Durchschnitt
                proposed = max(proposed, metric.spec_target + MIN_UPGRADE_DELTA)

                if proposed <= metric.spec_target + MIN_UPGRADE_DELTA:
                    continue

                proposals.append(
                    ImprovementProposal(
                        metric_name=metric.name,
                        current_spec=metric.spec_target,
                        proposed_spec=round(proposed, 4),
                        evidence_runs=len(recent),
                        avg_achieved=round(avg_achieved, 4),
                        confidence=min(1.0, 0.5 + len(recent) * 0.1),
                        recommendation=(
                            f"Spec {metric.name}: {metric.spec_target:.3f} → {proposed:.3f} "
                            f"(Code erreicht konstant ~{avg_achieved:.3f} ueber {len(recent)} Runs)"
                        ),
                    )
                )

        return proposals

    @property
    def run_count(self) -> int:
        return self._run_count

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._run_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Spec Auditor
# ═══════════════════════════════════════════════════════════════════════════════


def audit_all_specs(
    repo_root: str | Path = ".",
    *,
    improvement_proposals: list[ImprovementProposal] | None = None,
) -> SpecAuditResult:
    """Prueft ALLE Spec-Dateien auf Konsistenz, Widersprueche, Gaps, Regression.

    Args:
        repo_root: Pfad zum Repo-Root
        improvement_proposals: Optionale Upgrade-Vorschlaege aus Phase 2

    Returns:
        SpecAuditResult mit allen gefundenen Problemen
    """
    root = Path(repo_root)
    spec_dir = root / SPEC_DIR
    result = SpecAuditResult(
        files_checked=0,
        contradictions=[],
        gaps=[],
        regressions=[],
        inconsistencies=[],
        improvement_proposals=improvement_proposals or [],
        overall_health="healthy",
    )

    # ── 1. Pruefe ob alle erwarteten Spec-Dateien existieren ────────────
    for spec_file in SPEC_FILES:
        full_path = spec_dir / spec_file
        if not full_path.exists():
            result.gaps.append(f"Fehlende Spec-Datei: {spec_file}")
        else:
            result.files_checked += 1

    if result.files_checked == 0:
        result.overall_health = "critical"
        result.gaps.append("Keine Spec-Dateien gefunden!")
        return result

    # ── 2. Parse Spec-Dateien auf strukturierte Metriken ─────────────────
    spec_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for spec_file in SPEC_FILES:
        full_path = spec_dir / spec_file
        if not full_path.exists():
            continue
        try:
            content = full_path.read_text(encoding="utf-8")
            metrics = _extract_metrics_from_spec(content)
            if metrics:
                spec_metrics[spec_file] = metrics
        except Exception as e:
            logger.debug("Spec Auditor: konnte %s nicht lesen: %s", spec_file, e)

    # ── 3. Cross-Spec-Widerspruchs-Check ─────────────────────────────────
    result.contradictions = _find_cross_spec_contradictions(spec_metrics)

    # ── 4. Interne Inkonsistenz-Checks pro Spec ──────────────────────────
    for spec_file in SPEC_FILES:
        full_path = spec_dir / spec_file
        if not full_path.exists():
            continue
        try:
            content = full_path.read_text(encoding="utf-8")
            incs = _check_spec_internal_consistency(content, spec_file)
            result.inconsistencies.extend(incs)
        except Exception as _e:
            logger.debug("spec_improvement_loop: unkritisch exception: %s", _e)

    # ── 5. Gap-Erkennung: Code-Features ohne Spec-Abdeckung ──────────────
    result.gaps.extend(_detect_spec_gaps(root))

    # ── 6. Regressions-Check: Wuerden Improvement-Proposals etwas verschlechtern? ──
    for proposal in improvement_proposals or []:
        regression = _check_proposal_regression(proposal, spec_metrics)
        if regression:
            result.regressions.append(regression)

    # ── 7. Health-Assessment ─────────────────────────────────────────────
    issues = len(result.contradictions) + len(result.gaps) + len(result.regressions) + len(result.inconsistencies)
    if issues == 0 and len(result.improvement_proposals) > 0:
        result.overall_health = "healthy_with_upgrades"
    elif issues == 0:
        result.overall_health = "healthy"
    elif issues < 5:
        result.overall_health = "needs_attention"
    else:
        result.overall_health = "critical"

    return result


# Token-Filter: Diese Woerter sind keine echten Aurik-Metriken
_NON_METRIC_TOKENS: frozenset[str] = frozenset(
    {
        "float",
        "int",
        "str",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "q",
        "n",
        "d",
        "z",
        "x",
        "y",
        "i",
        "j",
        "k",
        "sr",
        "hz",
        "db",
        "ms",
        "s",
        "khz",
        "delta",
        "threshold",
        "score",
        "scale",
        "ratio",
        "fraction",
        "strength",
        "severity",
        "conf",
        "confidence",
        "wet_mix",
        "g_floor",
        "blended",
        "axis",
        "iqr",
        "holes",
        "regression",
        "shift",
        "sprung",
        "scalar",
        "scale_factor",
        "ceiling_avg",
        "recovery_ratio",
        "restored_compat",
        "af_penalty",
        "schwellwert",
        "korrelation",
        "pearson",
        "dateilänge",
        "dateien",
        "tonal_center",
        "spectral_novelty",
        "hpi",
        "w_crepe",
        "f_bump",
        "wet_mult",
        "gate",
        "infrasonic_rms",
        "crackle_density",
        "rotation_strength",
        "panns_singing_confidence",
        "panns_singing",
        "in",
        "bei",
        "mit",
        "nach",
        "fehler",
        "floor",
        "segment",
        "window_ms",
        "drift",
    }
)


def _extract_metrics_from_spec(content: str) -> dict[str, dict[str, Any]]:
    """Extrahiert strukturierte Metriken aus Spec-Markdown.

    Erkennt:
      - Tabellen mit Schwellwerten
      - `>= 0.XX` -Formate
      - Goal-Definitionen mit Zahlenwerten

    Filtert Code-Variablennamen heraus (keine echten Metriken).
    """
    # Code-Variablennamen, die fälschlich als Metriken erkannt werden
    _CODE_TOKENS = {
        "float",
        "int",
        "str",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "q",
        "n",
        "d",
        "z",
        "x",
        "y",
        "i",
        "j",
        "k",
        "sr",
        "hz",
        "db",
        "ms",
        "s",
        "khz",
        "delta",
        "threshold",
        "score",
        "scale",
        "ratio",
        "fraction",
        "strength",
        "severity",
        "conf",
        "confidence",
        "wet_mix",
        "g_floor",
        "blended",
        "axis",
        "iqr",
        "holes",
        "regression",
        "shift",
        "sprung",
        "scalar",
        "scale_factor",
        "ceiling_avg",
        "recovery_ratio",
        "restored_compat",
        "af_penalty",
        "schwellwert",
        "korrelation",
        "pearson",
        "dateilänge",
        "dateien",
        "tonal_center",
        "spectral_novelty",
        "hpi",
        "w_crepe",
        "f_bump",
        "wet_mult",
        "gate",
        "infrasonic_rms",
        "crackle_density",
        "rotation_strength",
        "panns_singing_confidence",
        "panns_singing",
        # German words / Greek letters / common code tokens:
        "in",
        "bei",
        "mit",
        "nach",
        "fehler",
        "floor",
        "segment",
        "window_ms",
        "drift",
        "δ",
        "β",
    }
    metrics: dict[str, dict[str, Any]] = {}
    pattern = r"`?(\w+)`?\s*[≥>=]\s*([0-9]+\.?[0-9]*)"
    for match in re.finditer(pattern, content):
        name = match.group(1).lower().replace("`", "")
        if name in _NON_METRIC_TOKENS:
            continue
        try:
            value = float(match.group(2))
            metrics[name] = {"threshold": value, "source": "spec_extraction"}
        except ValueError:
            continue

    return metrics


def _find_cross_spec_contradictions(spec_metrics: dict[str, dict[str, dict[str, Any]]]) -> list[tuple[str, str, str]]:
    """Findet Widersprueche zwischen Spec-Dateien (gleiche Metrik, unterschiedliche Werte)."""
    contradictions: list[tuple[str, str, str]] = []
    all_metrics: dict[str, dict[str, tuple[str, float]]] = {}

    for spec_file, metrics in spec_metrics.items():
        for name, data in metrics.items():
            if name not in all_metrics:
                all_metrics[name] = {}
            all_metrics[name][spec_file] = (spec_file, data["threshold"])

    for name, sources in all_metrics.items():
        if len(sources) < 2:
            continue
        values = list(sources.values())
        unique_vals = {v[1] for v in values}
        if len(unique_vals) > 1:
            contradictions.append(
                (
                    list(sources.keys())[0],
                    list(sources.keys())[1],
                    f"Metrik '{name}' hat unterschiedliche Schwellwerte: {unique_vals}",
                )
            )

    return contradictions


def _check_spec_internal_consistency(content: str, filename: str) -> list[str]:
    """Prueft eine Spec-Datei auf interne Widersprueche."""
    issues: list[str] = []

    # Pruefe auf doppelt definierte Werte
    pattern = r"`?(\w+)`?\s*[≥>=]\s*([0-9]+\.?[0-9]*)"
    seen: dict[str, list[str]] = {}
    for match in re.finditer(pattern, content):
        name = match.group(1).lower()
        if name in _NON_METRIC_TOKENS:
            continue
        line_no = content[: match.start()].count("\n") + 1
        if name not in seen:
            seen[name] = []
        seen[name].append(f"{match.group(2)} (line {line_no})")

    for name, occurrences in seen.items():
        unique = {o.split(" ")[0] for o in occurrences}
        if len(unique) > 1:
            issues.append(f"INTERNAL CONTRADICTION in {filename}: '{name}' definiert als {unique}")

    return issues


def _detect_spec_gaps(repo_root: Path) -> list[str]:
    """Erkennt Code-Features ohne Spec-Abdeckung durch Vergleich mit SpecConstitution."""
    from backend.core.spec_constitution import get_constitution

    gaps: list[str] = []
    const = get_constitution()

    # Pruefe: Alle 15 Musical Goals haben Spec-Abdeckung
    # (das ist per Definition so, aber wir dokumentieren es)
    if const.goal_count < 15:
        gaps.append(f"Nur {const.goal_count}/15 Musical Goals in Constitution definiert")

    # Pruefe: Alle VERBOTEN-Regeln aus VERBOTEN.md sind in Constitution
    verboten_path = repo_root / ".github" / "VERBOTEN.md"
    if verboten_path.exists():
        try:
            content = verboten_path.read_text(encoding="utf-8")
            v_pattern = r"\|\s*(V\d+)\s*\|"
            verboten_ids = set(re.findall(v_pattern, content))
            constitution_ids = {fp.id for fp in const.get_forbidden_patterns()}
            missing = verboten_ids - constitution_ids
            if missing:
                gaps.append(f"VERBOTEN-Regeln nicht in Constitution: {sorted(missing)}")
        except Exception as _e:
            logger.debug("spec_improvement_loop: unkritisch exception: %s", _e)

    return gaps


def _check_proposal_regression(
    proposal: ImprovementProposal,
    spec_metrics: dict[str, dict[str, dict[str, Any]]],
) -> str | None:
    """Prueft ob ein Spec-Upgrade-Vorschlag andere Metriken verschlechtern wuerde."""
    # Pruefe: Wuerde proposed_spec eine andere verwandte Metrik unerreichbar machen?
    if proposal.proposed_spec > 0.99:
        return (
            f"Regression-Risiko: {proposal.metric_name} Upgrade auf {proposal.proposed_spec} "
            f"ist zu nahe an 1.0 — unmenschlich hohe Anforderung."
        )
    if proposal.proposed_spec > proposal.current_spec * 1.5:
        return (
            f"Regression-Risiko: {proposal.metric_name} Sprung von {proposal.current_spec} "
            f"auf {proposal.proposed_spec} ist zu gross (>50% Anstieg). "
            f"Inkrementelles Upgrade empfohlen."
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Vollstaendiger Self-Improvement-Loop
# ═══════════════════════════════════════════════════════════════════════════════


class SpecImprovementLoop:
    """Der vollstaendige Self-Improvement-Zyklus.

    Fuehrt nach jedem Pipeline-Lauf alle drei Phasen aus:
      1. Spec-vs-Code-Vergleich
      2. Improvement Detection (mit History ueber mehrere Runs)
      3. Spec-Audit (bei Bedarf)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._detector = ImprovementDetector()
        self._last_audit: SpecAuditResult | None = None
        self._audit_every_n_runs: int = 10

    def process_run(
        self,
        achieved_metrics: dict[str, float],
        material: str = "unknown",
        *,
        force_audit: bool = False,
    ) -> dict[str, Any]:
        """Verarbeitet einen Pipeline-Lauf durch den kompletten Improvement-Zyklus.

        Args:
            achieved_metrics: dict von Name → Wert aus dem Pipeline-Lauf
            material: Material-Typ
            force_audit: Erzwingt Spec-Audit unabhaengig von Run-Count

        Returns:
            Dict mit comparison, proposals, audit (wenn durchgefuehrt)
        """
        # Phase 1: Compare
        comparison = compare_spec_vs_code(achieved_metrics, material)

        # Phase 2: Detect improvements
        proposals = self._detector.record_run(comparison)

        # Phase 3: Audit specs (periodisch oder bei neuen Proposals)
        audit_result = None
        with self._lock:
            do_audit = force_audit or len(proposals) > 0 or self._detector.run_count % self._audit_every_n_runs == 0
        if do_audit:
            audit_result = audit_all_specs(improvement_proposals=proposals)
            with self._lock:
                self._last_audit = audit_result

        return {
            "comparison": {
                "grade": comparison.overall_grade,
                "compliant": comparison.compliant_count,
                "exceeds": comparison.exceeds_count,
                "below": comparison.below_count,
                "recommendations": comparison.recommendations[:5],
            },
            "improvement_proposals": [
                {
                    "metric": p.metric_name,
                    "current_spec": p.current_spec,
                    "proposed_spec": p.proposed_spec,
                    "evidence_runs": p.evidence_runs,
                    "confidence": p.confidence,
                    "recommendation": p.recommendation,
                }
                for p in proposals
            ],
            "audit": None
            if audit_result is None
            else {
                "health": audit_result.overall_health,
                "contradictions": len(audit_result.contradictions),
                "gaps": audit_result.gaps,
                "regressions": audit_result.regressions,
                "inconsistencies": audit_result.inconsistencies,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_loop: SpecImprovementLoop | None = None
_loop_lock = threading.Lock()


def get_improvement_loop() -> SpecImprovementLoop:
    """Thread-sicherer Singleton-Accessor."""
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = SpecImprovementLoop()
    return _loop
