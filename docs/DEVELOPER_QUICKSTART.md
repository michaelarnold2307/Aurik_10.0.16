# Aurik Developer Quickstart — §v10.700

> Für neue Entwickler, die an der Aurik-Codebase arbeiten wollen.

## Voraussetzungen

- Python 3.10+
- Git
- 16 GB RAM empfohlen (Modelle brauchen ~6 GB)
- Linux empfohlen (Ubuntu 22.04+); macOS/Windows möglich

## Quick Setup (~10 min)

```bash
# 1. Repository klonen
git clone <aurik-repo-url>
cd Aurik_Standalone

# 2. Virtual Environment
python3.10 -m venv .venv_aurik
source .venv_aurik/bin/activate

# 3. Abhängigkeiten
pip install -r requirements/requirements_dev.txt
pip install -e .

# 4. Pre-Commit-Hooks
pre-commit install
pre-commit install --hook-type commit-msg

# 5. Quick Smoke Test
make test-smoke
```

## Wichtige Makefile-Targets

| Befehl | Dauer | Zweck |
|---|---|---|
| `make test-smoke` | ~5s | Schnellster Health-Check (Imports, Numerik) |
| `make test` | ~30s | Unit-Tests |
| `make test-spec-gate-core` | ~5min | Integration + Normative (ohne Langläufer) |
| `make compliance` | ~2s | VERBOTEN-Regeln prüfen |
| `make fmt` | ~3min | Auto-Formatierung |
| `make lint` | ~1min | Linting |
| `make typecheck` | ~2min | Mypy-Typprüfung |
| `make quality` | ~10min | Vollständiger Quality-Check |

## CI-Pipeline

Bei jedem Push/PR laufen automatisch:

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci-lite.yml` | Push main, PR | PR Evidence Gate, Determinism, Normative Guard, Quick Smoke |
| `solo-release-gate.yml` | Push main | Spec Evidence, Quick Smoke, Normative Tests, Compliance |
| `ci-cross-platform.yml` | Push main | Ubuntu + Windows Tests |
| `nightly-quality.yml` | Täglich 2:00 UTC | Spec Drift, AMRB, Release Coverage |

## Code-Struktur

```
backend/core/          # Kern-Pipeline (Restorer, Phases, Defects)
backend/core/phases/   # Einzelne DSP-Phasen
dsp/                   # DSP-Bibliothek (Filter, Analysen)
plugins/               # Plugin-System
denker/                # KI/ML-Intelligenz (optional, heavy)
Aurik10/               # GUI (PyQt5)
tests/unit/            # Unit-Tests
tests/normative/       # Normative CI-Gates (~70 Tests)
tests/integration/     # Integrationstests
scripts/               # Utility-Skripte (100+)
docs/                  # Dokumentation
.github/specs/         # Spezifikationen (~25 Specs)
```

## Vor dem ersten Commit

```bash
# Diese Checks MÜSSEN grün sein:
make test-smoke       # 5s
make compliance       # 2s
make test             # 30s
```

## Typische Workflows

### Bug fixen
1. Test schreiben, der den Bug reproduziert
2. Fix im Code
3. `make test` — Regression prüfen
4. Commit mit `fix: Beschreibung`

### Neue Phase hinzufügen
1. Phase in `backend/core/phases/` implementieren
2. In `phase_effect_catalog.py` registrieren
3. Normativen Test in `tests/normative/` schreiben
4. Spec in `.github/specs/` mit Evidenzblock

### Release vorbereiten
1. `make quality` — alles grün
2. `make test-spec-gate-core` — alle Gates grün
3. `docs/RELEASE_CHECKLIST.md` abarbeiten
4. Tag setzen: `git tag -s vX.Y.Z`

## Troubleshooting

**`make test-smoke` schlägt fehl:**
→ Virtual Environment aktiv? `source .venv_aurik/bin/activate`
→ Abhängigkeiten installiert? `pip install -r requirements/requirements_dev.txt`

**Pre-Commit-Hooks blocken Commit:**
→ `pre-commit run --all-files` zeigt alle Issues
→ `make fmt` fixt die meisten automatisch

**Mypy-Fehler:**
→ `make typecheck` für vollen Report
→ Report in `reports/mypy_report.txt`
