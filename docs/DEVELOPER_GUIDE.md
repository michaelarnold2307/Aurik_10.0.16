# Aurik 10 — Developer Guide

> Stand: 10.0.20 | Setup, Workflow, Testing, Contributing

## Quick Start

```bash
git clone <repo>
cd Aurik_Standalone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements_dev.txt
```

## Projektstruktur

```
Aurik10/         GUI (PyQt5)
backend/         Backend (Pipeline, DSP, ML)
  api/           Bridge-API (§V4: nur hier importieren)
  core/          Pipeline, Phasen, Qualität
    phases/      69 Phasen via PhaseInterface
    dsp/         Signalverarbeitung
    ml/          ONNX-Session-Management
denker/          Strategie (vom Bridge-Verbot ausgenommen)
plugins/         ML-Plugins (ONNX-Modelle)
tests/           Unit + Integration + Normative Gates
scripts/         Build, CI, Kalibrierung, Analyse
docs/            Dokumentation
```

## Entwicklungsworkflow

```bash
# Pre-Commit
pre-commit run --all-files

# Tests
pytest tests/unit -q --timeout=30

# Type-Check
mypy --config-file pyproject.toml .

# Lint
ruff check backend/ Aurik10/

# Format
black backend/ Aurik10/ --line-length 120
```

## Neue Phase erstellen

1. `backend/core/phases/phase_XX_name.py` erstellen
2. Von `PhaseInterface` ableiten
3. `process(audio, sample_rate, **kwargs)` implementieren
4. NaN/Inf-Schutz: `np.nan_to_num()` auf Ausgabe
5. Test: `tests/unit/test_phase_XX_name.py`
6. In `unified_restorer_v3.py` registrieren

## Bridge-API erweitern

Neue Funktion in `backend/api/bridge.py`:
```python
def get_my_module():
    """Gibt MyModule-Singleton zurück."""
    from backend.core.my_module import get_my_module as _fn
    return _fn()
```

## Spec-Referenzen

- GEBOTE: `.github/copilot-instructions.md` §I
- VERBOTE: `.github/copilot-instructions.md` §II
- Specs: `.github/specs/`
- Governance: `.github/GOVERNANCE.md`

## Version

Version in 3 Dateien synchron halten:
- `pyproject.toml` (kanonisch)
- `README.md`
- `CHANGELOG.md`

Prüfen: `python scripts/check_version_consistency.py`
