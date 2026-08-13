# `denker/` — Kognitive Orchestrierungsschicht von Aurik 10.0.0

> **Spec-Referenz**: §2.1 Kernmodule · §2.2 Kanonische Pipeline · §3.2 Singleton-Pattern ·
> §6.6 Tonträgerketten-Erkennung · §9.5 Performance-Budget

### Namenskonvention

Der `denker/`-Layer verwendet **bewusst deutsche Namen** (`restauriere()`, `_konvertiere()`, `denke()`) —
er ist Auriks „Denkschicht", die kognitive Entscheidungen trifft. Die `backend/core/`-Engine
verwendet **englische Namen** (`optimize_naturalness()`, `reprocess_studio2026()`). Diese Trennung
ist kein Zufall, sondern Design: Denker = Deutsch (kognitive Semantik), Engine = Englisch
(technische Semantik).

---

## Übersicht

Das `denker/`-Paket ist die **kognitive Orchestrierungsschicht** von Aurik 10.0.0. Es bietet
eine einheitliche High-Level-API über alle §2.1-Kernmodule und schützt die Produktions-Pipeline
vor direkten Abhängigkeiten an Core-Internals.

```text
┌─────────────────────────────────────────────────────────┐
│     denker/                                             │
│     ┌──────────────┐                                    │
│     │ AurikDenker  │ ← Haupt-Orchestrator               │
│     └─────┬────────┘                                    │
│           │koordiniert 10 Stufen                     │
│   ┌───────┼──────────────────────────────────┐          │
│   ▼       ▼        ▼        ▼        ▼       ▼          │
│ Toni   Kette   Defekt   Rep.  Exzell.  Strat.  ...      │
└──┼────────┼────────┼────────┼────────┼───────┼──────────┘
   ▼        ▼        ▼        ▼        ▼       ▼
 Medium  Medium  Defect-  scipy  Excel-  Perf-
 Detect. Detect. Scanner         Optim.  Guard
```

### Warum eine Orchestrierungsschicht?

Die §2.1-Kernmodule (UnifiedRestorerV3, DefectScanner, ExcellenceOptimizer usw.) sind
bewusst als isolierte, fokussierte Komponenten implementiert. Die Orchestrierung ihrer
Zusammenarbeit in der richtigen Reihenfolge (§2.2 kanonische Pipeline) liegt in der
Verantwortung des `denker/`-Pakets.

**Vorteile gegenüber direktem Core-Aufruf:**

- Ein einziger Entry-Point (`restauriere()`) statt mehrerer Core-Aufrufe
- RT-Budget-Überwachung (`_3X_RT_LIMIT = 8.0` s/s, §9.5) an zentraler Stelle
- NaN/Inf-Guard (§3.1) vor Weitergabe an Folge-Module
- Alle Ergebnisse als `@dataclass` mit `as_dict()` (§3.6)

---

## Modulübersicht

### `aurik_denker.py` — Haupt-Orchestrator

```python
from denker import AurikDenker, get_aurik_denker, restauriere

# Convenience (empfohlen):
ergebnis = restauriere(audio, sr=48_000)

# Direkt:
denker = get_aurik_denker()
ergebnis = denker.restauriere(audio, sr=48_000)
```

Koordiniert alle 10 Stufen in der richtigen Reihenfolge:

1. TontraegerDenker → Materialerkennung
2. TontraegerketteDenker → Kettenerkennung (§6.6)
3. DefektDenker → Defektanalyse
4. MusikalischerGlobalplan → Cross-Phase-Reasoning (§Dach)
5. StrategieDenker → Phasenstrategie (8×RT-Budget)
6. ReparaturDenker → Direktreparatur (Preprocessing vor UV3)
7. RekonstruktionsDenker → Lückeninterpolation (Preprocessing vor UV3)
8. RestaurierDenker → Vollrestaurierung (UnifiedRestorerV3)
9. ExzellenzDenker → Musical-Goals-Check + Optimierung
10. VERSA MOS-Gate → Finales Qualitätsurteil

**`AurikErgebnis` (17 Felder):**

| Feld | Typ | Bedeutung |
| --- | --- | --- |
| `audio` | `np.ndarray` | Restauriertes Signal |
| `material_type` | `str` | Erkannter Träger |
| `chain_string` | `str` | z.B. `"tape→mp3_low"` |
| `defect_scores` | `dict` | DefektType → Score |
| `musical_goals` | `dict[str, float]` | 14 Musical Goals |
| `excellence_score` | `float` | Gesamt-Exzellenz ∈ [0,1] |
| `goals_passed` | `int` | Ziele ≥ Schwellwert |
| `goals_total` | `int` | Gesamt-Ziele geprüft |
| `rt_factor` | `float` | Verarbeitungszeit / Audiodauer |
| `warnings` | `list[str]` | Warnungen (DE) |
| `processing_note` | `str` | Kurznotiz (DE) |
| `chain_info` | `KettenErgebnis \| None` | Kettenerkennung-Detail |
| `defect_info` | `DefektErgebnis \| None` | Defekt-Detail |
| `strategy_info` | `StrategieErgebnis \| None` | Strategie-Detail |
| `restoration_info` | `RestaurierErgebnis \| None` | Restaurier-Detail |
| `reconstruction_info` | `RekonstruktionsErgebnis \| None` | Rekonstruktion-Detail |
| `excellence_info` | `ExzellenzErgebnis \| None` | Exzellenz-Detail |

---

### `tontraeger_denker.py` — Trägermedien-Erkennung

```python
from denker import TontraegerDenker, get_tontraeger_denker

denker = get_tontraeger_denker()
ergebnis = denker.erkenne(audio, sr=48_000)
# ergebnis.material_type: str  (z.B. "tape", "vinyl", "shellac")
# ergebnis.confidence: float
```

Wraps: `forensics.medium_detector.MediumDetector`
Spec: §2.1 MediumDetector · §6.7 Transfer-Chain-Erkennung

---

### `tontraegerkette_denker.py` — Tonträgerketten-Erkennung (§6.6)

```python
from denker import TontraegerketteDenker, get_tontraegerkette_denker

denker = get_tontraegerkette_denker()
ergebnis = denker.erkenne_kette(audio, sr=48_000)
# ergebnis.chain_string:       "tape→mp3_low"
# ergebnis.is_multi_generation: True
# ergebnis.generation_count:    2
# ergebnis.combined_phases:     ["phase_03_denoise", "phase_23_spectral_repair"]
```

**§6.6-Pflichtimplementierung** (`bindend ab v10.0.0`): Erkennt mehrstufige
Degradationspfade (z.B. Kassette→MP3-Komprimierung) und kombiniert die Phasen
beider Materialien automatisch.

Spec: §6.6 · §6.7.1 Spektralfingerabdruck · §6.7.2 MaterialType-Ableitung

---

### `defekt_denker.py` — Defektanalyse

```python
from denker import DefektDenker, get_defekt_denker

denker = get_defekt_denker()
ergebnis = denker.analysiere(audio, sr=48_000, material="tape")
# ergebnis.defect_scores: dict[str, float]
# ergebnis.primary_defect: str
# ergebnis.confidence: float
```

Wraps: `core.defect_scanner.DefectScanner` + `core.causal_defect_reasoner.CausalDefectReasoner`
Spec: §2.4 CausalDefectReasoner · §6.3 DefectType-Katalog (24 Defekte)

---

### `reparatur_denker.py` — Direktreparatur (self-contained)

```python
from denker import ReparaturDenker, get_reparatur_denker

denker = get_reparatur_denker()
ergebnis = denker.repariere(audio, sr=48_000, defects={"clicks": 0.8})
# ergebnis.audio: np.ndarray  # repariertes Signal
# ergebnis.repairs_applied: list[str]
```

**Besonderheit**: Einziger Denker ohne externe Core-Abhängigkeit — direkte
scipy-Implementierung. Geeignet als Fallback wenn Core-Module nicht verfügbar.
Spec: §3.1 Numerische Robustheit · §3.7 Type-Annotations

---

### `strategie_denker.py` — Phasenstrategie

```python
from denker import StrategieDenker, get_strategie_denker

denker = get_strategie_denker()
ergebnis = denker.plane(material="tape", defects={"hiss": 0.7}, audio_duration_s=120.0)
# ergebnis.selected_phases: list[str]
# ergebnis.budget_ok: bool
# ergebnis.estimated_rt_factor: float
```

Wraps: `core.performance_guard.PerformanceGuard`
Spec: §9.5 Performance-Budget · `_3X_RT_LIMIT = 8.0`

---

### `restaurier_denker.py` — Vollrestaurierung

```python
from denker import RestaurierDenker, get_restaurier_denker

denker = get_restaurier_denker()
ergebnis = denker.restauriere(audio, sr=48_000, material="tape")
# ergebnis.audio: np.ndarray
# ergebnis.rt_factor: float  # INVARIANT: ≤ 3.0 × Audiodauer (enforce_3x_rt=True)
# ergebnis.phases_executed: list[str]
```

Wraps: `core.unified_restorer_v3.UnifiedRestorerV3`
Spec: §2.2 Kanonische Pipeline

**Invariante**: `enforce_3x_rt=True` — Überschreitung des 3×-RT-Limits erzeugt
Warnung und bricht Optimierungsschleife ab (kein Hard-Crash).

---

### `rekonstruktions_denker.py` — Lückenfüllung / Inpainting

```python
from denker import RekonstruktionsDenker, get_rekonstruktions_denker

denker = get_rekonstruktions_denker()
ergebnis = denker.rekonstruiere(audio, sr=48_000, gap_start_ms=1200.0, gap_end_ms=1350.0)
# ergebnis.audio: np.ndarray
# ergebnis.method_used: str  ("GapReconstructor" | "lineare_interpolation")
```

Wraps: `core.gap_reconstructor.GapReconstructor` (Fallback: lineare Interpolation)
Spec: §2.12 Musikalische Phrasenkontextfenster · §4.5 Dropout-Inpainting

---

### `exzellenz_denker.py` — Musical-Goals-Check + Optimierung

```python
from denker import ExzellenzDenker, get_exzellenz_denker

denker = get_exzellenz_denker()
ergebnis = denker.pruefe_und_optimiere(audio, sr=48_000, material="tape")
# ergebnis.excellence_score: float  ∈ [0, 1]
# ergebnis.musical_goals: dict[str, float]  # 14 Ziele
# ergebnis.goals_passed: int
# ergebnis.goals_total: int
# ergebnis.audio: np.ndarray  # ggf. optimiertes Signal
```

Wraps: `core.vocal_ai_enhancement` + `core.excellence_optimizer.ExcellenceOptimizer`
         + `backend.core.musical_goals.MusicalGoalsChecker`
Spec: §1.2 Die 14 Musical Goals · §2.5 GPParameterOptimizer · §2.29 PMGG

---

## Singleton-Pattern (§3.2)

Alle Denker implementieren das vorgeschriebene Thread-sichere Singleton-Muster:

```python
import threading
from typing import Optional

_instance: Optional[MeinDenker] = None
_lock = threading.Lock()

def get_mein_denker() -> MeinDenker:
    """Thread-sicherer Singleton (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MeinDenker()
    return _instance
```

---

## Schnellstart

```python
import numpy as np
from denker import restauriere

# Synthetisches Testsignal
sr = 48_000
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, 3 * sr)).astype(np.float32)

# Einzeiliger Aufruf — orchestriert die gesamte Aurik-Pipeline
ergebnis = restauriere(audio, sr=sr)

print(f"Material:     {ergebnis.material_type}")
print(f"Kette:        {ergebnis.chain_string}")
print(f"RT-Faktor:    {ergebnis.rt_factor:.2f}×")
print(f"Exzellenz:    {ergebnis.excellence_score:.2f}")
print(f"Goals:        {ergebnis.goals_passed}/{ergebnis.goals_total}")
print(f"Warnungen:    {ergebnis.warnings}")
```

---

## Integrations-Anforderungen

### PyInstaller (`aurik_90.spec`)

`denker/` und alle Unter-Module müssen in `hiddenimports` aufgenommen werden:

```python
hiddenimports = [
    # ... bestehende Einträge ...
    'denker',
    'denker.aurik_denker',
    'denker.defekt_denker',
    'denker.exzellenz_denker',
    'denker.rekonstruktions_denker',
    'denker.reparatur_denker',
    'denker.restaurier_denker',
    'denker.strategie_denker',
    'denker.tontraeger_denker',
    'denker.tontraegerkette_denker',
]
```

### Produktions-Verdrahtung (empfohlener Einstiegspunkt)

Denker sollte über die API-Schicht (`backend/api/`) aufgerufen werden, nicht direkt
aus dem Frontend:

```python
# backend/api/rest/api.py — empfohlene Integration
from denker import restauriere, AurikErgebnis

@app.post("/restore")
async def restore_audio(audio_bytes: bytes, sr: int = 48_000) -> dict:
    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    ergebnis: AurikErgebnis = restauriere(audio, sr=sr)
    return ergebnis.as_dict()
```

---

## Test-Suite

Vollständige 1:1-Testabdeckung unter `tests/unit/test_denker/`:

```text
tests/unit/test_denker/
├── __init__.py
├── test_aurik_denker.py        (295 Zeilen — Hauptorchestrator)
├── test_defekt_denker.py
├── test_denker_init.py         (denker/__init__.py + restauriere())
├── test_exzellenz_denker.py
├── test_rekonstruktions_denker.py
├── test_reparatur_denker.py
├── test_restaurier_denker.py
├── test_strategie_denker.py
├── test_tontraeger_denker.py
└── test_tontraegerkette_denker.py
```

Tests ausführen:

```shell
.venv_aurik/bin/python -m pytest tests/unit/test_denker/ -v --tb=short
```

---

## Qualitäts-Invarianten

Alle Denker garantieren:

- ✅ **NaN/Inf-frei** (§3.1): `np.nan_to_num()` + `np.clip(-1.0, 1.0)` vor Rückgabe
- ✅ **RT-Budget** (§9.5): `_3X_RT_LIMIT = 8.0` s/s — kein Timeout, nur Warnung
- ✅ **Dataclass-Ergebnisse** (§3.6): alle Ergebnisse als `@dataclass` mit `as_dict()`
- ✅ **Type-Annotations** (§3.7): alle öffentlichen Methoden vollständig annotiert
- ✅ **Logging** (§3.5): `logger = logging.getLogger(__name__)`, kein `print()`
- ✅ **Graceful Degradation** (§3.4): `try/except ImportError` für optionale Core-Module

---

## Architektur-Position

```text
Frontend (PyQt5)
    │  (Qt-Signals/Slots ODER)
    ▼
API-Schicht (backend/api/)
    │  POST /restore
    ▼
┌─────────────────────────┐
│  denker.restauriere()   │  ← HIER: Entry-Point
│  AurikDenker.dendke()   │
└───┬─────────────────────┘
    │  orchestriert
    ▼
core/unified_restorer_v3.py    (RestaurierDenker → §2.2 Pipeline)
core/defect_scanner.py          (DefektDenker)
core/excellence_optimizer.py    (ExzellenzDenker)
forensics/medium_detector.py    (TontraegerDenker, TontraegerketteDenker §6.6)
core/gap_reconstructor.py       (RekonstruktionsDenker)
core/performance_guard.py       (StrategieDenker)
scipy (direkt)                  (ReparaturDenker — self-contained Fallback)
```

---

## Versionsinfo

Spec-Version: Aurik 10.0.0 — Stand: Februar 2026


## §v10.900: SOTA-Ketten-Integration (2026-08)

Die Denker sind das zentrale Räderwerk der SOTA-Reparatur-Kette:

```
DefektDenker.analysiere()
    → DefectConsensusPipeline (30 Module → 1 Manifest)
    → DefektErgebnis._consensus_manifest

RestaurierDenker.restauriere()
    → RepairPlanner.plan(manifest) → RepairPlan
    → _uv3_kwargs["repair_plan"] = plan

UnifiedRestorerV3.restore(repair_plan=...)
    → selected_phases = Plan-Reihenfolge
    → CoordinatedRepair führt mit Guards aus
```

**Neue Pipelines (siehe .github/specs/v10.900_sota_pipeline_architecture.md):**
- Defect Consensus: 5 aktive Detektoren, 5 kontextabhängige (sichtbar geloggt)
- Repair Planner: 6 Prioritätsstufen, Inpainting immer zuletzt
- Guards: 4-Schicht (TruePeak-relativ, Pumping, Formant, Spektral) + UTMOS-Loop
- Bug-Register: 11 Bugs dokumentiert + Regressionstest (tests/unit/test_bug_register_regression.py)
