import pytest

"""
§2.55 PMGG-CIG-Synchronisations-Invariante — CI-Regression-Test
=================================================================
Normative Invariante (instructions_version 7.1, v10.0.0):

    CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[phase] ⊇ PMGG.PHASE_GOAL_EXCLUSIONS[phase] ∩ P1P2
    PMGG.PHASE_GOAL_EXCLUSIONS[phase] ⊇ CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[phase] ∩ P1P2

Beide Richtungen müssen für alle Phasen erfüllt sein.
Ein Mismatch bedeutet: eine neue Phase wurde implementiert und nur eine der beiden Tabellen
aktualisiert — das erzeugt spurious CIG-Rollbacks und pipeline-weite Stärke-Kaskaden.
"""

P1_P2_GOALS = frozenset(
    {
        "natuerlichkeit",
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
    }
)


def _cig_for_phase(phase_id: str, cig_dict: dict) -> frozenset:
    """Prefix-basierter CIG-Lookup (identische Logik wie _resolve_phase_specific_drift_exclusions)."""
    for prefix, excl in cig_dict.items():
        if phase_id == prefix:
            return excl  # type: ignore[no-any-return]
        if phase_id.startswith(prefix) and (len(phase_id) == len(prefix) or phase_id[len(prefix)] in "_-"):
            return excl  # type: ignore[no-any-return]
    return frozenset()


def _pmgg_for_prefix(cig_prefix: str, pmgg_dict: dict) -> frozenset:
    """Findet den PMGG-Eintrag zum CIG-Präfix (prefix-aware, P1/P2 gefiltert)."""
    for pid, pgoals in pmgg_dict.items():
        if pid == cig_prefix:
            return frozenset(g for g in pgoals if g in P1_P2_GOALS)
        if pid.startswith(cig_prefix) and (len(pid) == len(cig_prefix) or pid[len(cig_prefix)] in "_-"):
            return frozenset(g for g in pgoals if g in P1_P2_GOALS)
    return frozenset()


@pytest.mark.unit
def test_pmgg_to_cig_no_missing_p1p2_goals():
    """Jedes P1/P2-Goal das PMGG für eine Phase ausschließt MUSS auch in CIG ausgeschlossen sein.

    Mechanismus: PMGG erlaubt Phase bei voller Stärke → CIG akkumuliert den Delta trotzdem
    → spurious Rollback an späterer Phase → Defekte bleiben unrepariert (§2.55).
    """
    from backend.core.cumulative_interaction_guard import _PHASE_SPECIFIC_DRIFT_EXCLUSIONS
    from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

    missing: list[tuple[str, list[str]]] = []
    for phase_id, goals in PHASE_GOAL_EXCLUSIONS.items():
        p12_pmgg = frozenset(g for g in goals if g in P1_P2_GOALS)
        if not p12_pmgg:
            continue
        p12_cig = _cig_for_phase(phase_id, _PHASE_SPECIFIC_DRIFT_EXCLUSIONS) & P1_P2_GOALS
        gap = p12_pmgg - p12_cig
        if gap:
            missing.append((phase_id, sorted(gap)))

    assert not missing, (
        f"§2.55 VERLETZT: {len(missing)} PMGG→CIG-Mismatches (PMGG schließt P1/P2-Goal aus, CIG nicht):\n"
        + "\n".join(f"  {pid}: PMGG excl. {goals} aber CIG fehlt diese Goals" for pid, goals in missing)
        + "\n\nFix: _PHASE_SPECIFIC_DRIFT_EXCLUSIONS in cumulative_interaction_guard.py ergänzen."
    )


def test_cig_to_pmgg_no_missing_p1p2_goals():
    """Jedes P1/P2-Goal das CIG nicht als Drift zählt MUSS auch in PMGG ausgeschlossen sein.

    Mechanismus: CIG lässt Phase ohne Drift-Strafe passieren → PMGG blockiert sie trotzdem
    → Phase wird auf minimale Stärke gedrückt → Defekt bleibt unrepariert (§2.55).
    """
    from backend.core.cumulative_interaction_guard import _PHASE_SPECIFIC_DRIFT_EXCLUSIONS
    from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

    missing: list[tuple[str, list[str]]] = []
    for cig_prefix, cig_excl in _PHASE_SPECIFIC_DRIFT_EXCLUSIONS.items():
        p12_cig = frozenset(g for g in cig_excl if g in P1_P2_GOALS)
        if not p12_cig:
            continue
        p12_pmgg = _pmgg_for_prefix(cig_prefix, PHASE_GOAL_EXCLUSIONS)
        gap = p12_cig - p12_pmgg
        if gap:
            missing.append((cig_prefix, sorted(gap)))

    assert not missing, (
        f"§2.55 VERLETZT: {len(missing)} CIG→PMGG-Mismatches (CIG schließt P1/P2-Goal aus, PMGG nicht):\n"
        + "\n".join(f"  {pfx}: CIG excl. {goals} aber PMGG fehlt diese Goals" for pfx, goals in missing)
        + "\n\nFix: PHASE_GOAL_EXCLUSIONS in per_phase_musical_goals_gate.py ergänzen."
    )


def test_pmgg_cig_sync_both_directions():
    """Kombinierter Smoke-Test: bidirektionale Synchronisation in einem Assert."""
    from backend.core.cumulative_interaction_guard import _PHASE_SPECIFIC_DRIFT_EXCLUSIONS
    from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

    # PMGG→CIG
    pmgg_to_cig_gaps = []
    for phase_id, goals in PHASE_GOAL_EXCLUSIONS.items():
        p12 = frozenset(g for g in goals if g in P1_P2_GOALS)
        if not p12:
            continue
        gap = p12 - (_cig_for_phase(phase_id, _PHASE_SPECIFIC_DRIFT_EXCLUSIONS) & P1_P2_GOALS)
        if gap:
            pmgg_to_cig_gaps.append((phase_id, sorted(gap)))

    # CIG→PMGG
    cig_to_pmgg_gaps = []
    for cig_prefix, cig_excl in _PHASE_SPECIFIC_DRIFT_EXCLUSIONS.items():
        p12 = frozenset(g for g in cig_excl if g in P1_P2_GOALS)
        if not p12:
            continue
        gap = p12 - _pmgg_for_prefix(cig_prefix, PHASE_GOAL_EXCLUSIONS)
        if gap:
            cig_to_pmgg_gaps.append((cig_prefix, sorted(gap)))

    total = len(pmgg_to_cig_gaps) + len(cig_to_pmgg_gaps)
    assert total == 0, (
        f"§2.55 PMGG-CIG-Synchronisation: {total} Mismatches "
        f"({len(pmgg_to_cig_gaps)} PMGG→CIG, {len(cig_to_pmgg_gaps)} CIG→PMGG). "
        "Beim Hinzufügen einer neuen Phase immer BEIDE Tabellen synchron aktualisieren."
    )
