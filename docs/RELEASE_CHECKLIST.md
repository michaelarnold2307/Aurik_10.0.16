# Aurik Release-Checkliste — §v10.700

> **Zweck:** Jeder Release (major/minor/patch) muss diese Checkliste durchlaufen.
> **Verantwortlich:** Release Maintainer
> **Letztes Update:** August 2026

## Vor dem Release

### 1. Code-Qualität
- [ ] `make quality` besteht (fmt + lint + typecheck + security)
- [ ] `make compliance` ohne Errors
- [ ] `make test` (unit tests) grün
- [ ] `make test-spec-gate-core` grün (ohne Langläufer)
- [ ] 0 neue pylint Errors seit letztem Release

### 2. Spezifikation
- [ ] Alle Specs in `.github/specs/` mit Status "✅ Implementiert" sind tatsächlich implementiert
- [ ] `scripts/spec_drift_check.py` zeigt keine Drift
- [ ] Keine Spec-Änderung ohne Evidenzblock im PR

### 3. Normative Gates
- [ ] `test_no_production_stubs.py` — keine Platzhalter
- [ ] `test_full_pipeline_determinism.py` — deterministisch
- [ ] `test_release_must_0m_0q_contract.py` — 0 TODO/FIXME/HACK/Q in Release-Pfaden
- [ ] `test_stability_invariants.py` — alle Invarianten erfüllt
- [ ] `test_no_stale_test_markers.py` — keine veralteten Marker

### 4. Performance
- [ ] `scripts/ci_benchmark_gate.py` — keine Regression
- [ ] `test_performance_budget_ci_gate.py` — Budgets eingehalten
- [ ] Speicher: <8 GB RAM im Normalbetrieb
- [ ] Latenz: <30s für 3-Minuten-Song (Vinyl, restoration mode)

### 5. Installer
- [ ] `make installer-all` läuft durch
- [ ] `pytest tests/normative/test_installer_smoke.py` grün
- [ ] Installer startet auf allen Zielplattformen
- [ ] Signatur-Prüfung dokumentiert (`docs/VERIFY_SIGNATURE.md`)

### 6. Dokumentation
- [ ] `docs/PROJECT_STATUS.md` aktualisiert (Version, Datum, Status)
- [ ] CHANGELOG aktualisiert
- [ ] Neue Features dokumentiert
- [ ] Breaking Changes markiert

### 7. SBOM
- [ ] `python scripts/generate_sbom.py` aktuell
- [ ] Abhängigkeiten auf bekannte CVEs geprüft

## Nach dem Release

### 8. Tag & Signatur
- [ ] Git-Tag gesetzt: `git tag -s vX.Y.Z -m "Aurik vX.Y.Z"`
- [ ] Tag gepusht: `git push origin vX.Y.Z`
- [ ] Release-Notes auf GitHub veröffentlicht

### 9. Smoke-Test (produktiv)
- [ ] Installer von Release-Seite heruntergeladen
- [ ] Signatur verifiziert
- [ ] Quick-Restore-Test mit Beispieldatei

### 10. Monitoring
- [ ] Nightly-Quality-Gate läuft
- [ ] Crash-Reporting aktiv
- [ ] Keine unerwarteten Fehler im Log
