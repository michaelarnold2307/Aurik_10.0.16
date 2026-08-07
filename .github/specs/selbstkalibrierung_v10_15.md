# §v10.15 Selbstkalibrierungs-Architektur (Self-Calibration Architecture)

## Prinzip

> Aurik verhält sich wie ein echter Toningenieur: Jeder einzelne Eingriff
> wird kalibriert, verifiziert und nur bei nachgewiesener Verbesserung
> übernommen. Kein Parameter ist hartcodiert, wenn er auch aus dem Signal
> ableitbar ist.

## Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────┐
│                    AUDIO-DATENFLUSS                       │
├──────────┬──────────┬────────────────┬──────────────────┤
│ PRE      │ PHASE 01 │  PHASE 02 … 64 │ POST-PROCESSING  │
│ (Import) │  … 64    │  (PMGG-Loop)   │  (PostGate)      │
├──────────┼──────────┼────────────────┼──────────────────┤
│ STCG     │ PMGG     │  PMGG          │  PostGate        │
│ ✓ v10.14 │ ✓        │  ✓             │  ★ v10.15        │
│ Polarity │          │                │                  │
│ ✓ TODO   │          │                │                  │
└──────────┴──────────┴────────────────┴──────────────────┘
```

## Drei Zonen der Selbstkalibrierung

### Zone A: Pre-Pipeline (vor Phase 01)

| Komponente         | Status    | Maßnahme                                  |
|--------------------|-----------|-------------------------------------------|
| STCG (L/R-Delay)   | ✅ v10.14 | Universeller 20ms-Guard                   |
| Polarity Inversion | ★ TODO   | Verify via L/R-Korrelation Δ vor/nach     |

### Zone B: Main Pipeline (Phasen 01–64)

| Komponente         | Status    | Maßnahme                                  |
|--------------------|-----------|-------------------------------------------|
| Alle 64 Phasen     | ✅ PMGG   | 15-Ziele-Schnellcheck + 5 Retry-Stufen    |
| PhaseSkipper       | ✅        | Defekt-basiert                            |
| Preflight-Risk     | ✅        | Risiko-basiert                            |

**PMGG-Regeln (unverändert):**

1. 5 s Stichprobe aus Audio-Mitte VOR der Phase messen (15 Ziele)
2. Phase ausführen
3. Dieselben 15 Ziele NACH der Phase messen
4. Δ < −Schwellwert → Retry mit strength × 0.65, 0.50, 0.35, 0.20, 0.10
5. Nach 5 Retries immer noch Regression → HPE-Check
6. HPE-Delta < −0.02 → Phase verwerfen, Pre-Phase-Audio wiederherstellen

### Zone C: Post-Processing (nach Phase 64)

> ★ NEU in v10.15: Alle Post-Processing-Komponenten werden durch
> `PostProcessingGate` gewrappt — analog zu PMGG, aber mit reduziertem
> Ziel-Set (5 statt 15 Ziele, da es sich um finale Politur handelt).

| Komponente              | Typ          | Kalibrierung                      |
|--------------------------|--------------|-----------------------------------|
| VocalScratchRepair       | Detektion    | PostGate: Brillanz+Natürlichkeit  |
| TapeHeadArtifactRepair   | Detektion    | PostGate: Transparenz+Wärme       |
| AntiMufflingPass         | Spektral     | PostGate: Brillanz+Wärme          |
| SmartTapeRepair          | Detektion    | PostGate: Transparenz             |
| ArtifactEchoRemoval      | Detektion    | PostGate: SpatialDepth            |
| SibilanceMaxRepair       | Detektion    | PostGate: Natürlichkeit            |
| VocalClarityMax          | Enhancement  | PostGate: Artikulation+Brillanz   |
| SpecializedDefectRepair  | Detektion    | PostGate: Transparenz             |
| **HumanizationPass**     | Enhancement  | **Adaptive Stärke** + PostGate    |
| PerceptualExportOptimizer| Enhancement  | PostGate: Gesamtqualität           |
| DirectDefectRepair       | Detektion    | PostGate: Transparenz             |
| HarmonicPreservationGuard| Guard        | PostGate: Wärme+TimbAuth           |
| HarmonicLatticeAnalyzer  | Guard        | PostGate: TonalCenter              |

**PostGate-Regeln (v10.0.5 hardened):**

1. 5 s Stichprobe VOR der Komponente messen (5 Ziele, DSP-only, ≤ 80 ms)
2. Komponente ausführen
3. 5 Ziele NACH der Komponente messen
4. Δ < −REGRESSION_THRESHOLD → Komponente überspringen
5. KEINE Retry-Schleife (Post-Processing ist nicht iterativ kalibrierbar)
6. Ausnahme: HumanizationPass hat adaptive Stärke (0.05–0.25)
7. **§v10.0.5 Lambda-Signatur-Guard**: `component_fn` MUSS 3 positional args
   `(audio, sr, strength=None)` akzeptieren. `PostProcessingGate._validate_lambda()`
   prüft dies via `inspect.signature` beim ersten `apply()`-Aufruf. 2-arg-Lambdas
   wie `lambda a, sr: ...` lösen sofort `AssertionError` aus — kein kryptischer
   Runtime-TypeError mehr.

### Zone D: STCG (quer durch alle Zonen)

| Aufrufer                  | Status    |
|---------------------------|-----------|
| pre_pipeline              | ✅ 20ms   |
| post_pipeline             | ✅ 20ms   |
| phase_12_pre_chunking     | ✅ 20ms   |
| phase_12_wow_flutter_fix  | ✅ 20ms   |
| phase_24                  | ✅ 20ms   |
| phase_31                  | ✅ 20ms   |
| import_pipeline           | ✅ 20ms   |
| stereo_drift_final        | ✅ 20ms   |
| post_export               | ✅ 20ms   |

## HumanizationPass: Adaptive Stärke

**Vorher (v10.14):** `strength=0.15` hartcodiert
**Nachher (v10.15):** `strength = calibrate(audio, sr)` → ∈ [0.05, 0.25]

Kalibrierung:

- Analysiert die spektrale Dichte und den Dynamikumfang
- Bei bereits „lebendigem" Material (hohe Mikrodynamik) → niedrige Stärke
- Bei „sterilem" Material (flache Dynamik) → höhere Stärke
- PostGate verifiziert: Wärme + Natürlichkeit nach Humanization

## Ausnahmen — keine Selbstkalibrierung

Folgende Komponenten haben KEINE Kalibrierung, weil sie:

- binär sind (entweder sie erkennen einen Defekt und reparieren, oder nicht)
- keine tunable Parameter haben
- immer konservativ arbeiten

| Komponente            | Grund für Ausnahme                        |
|-----------------------|-------------------------------------------|
| PhaseOutputGuard      | Safety-only (NaN/Inf/Clip), kein Parameter |
| DC-Offset-Korrektur   | Deterministisch (Hochpass 10 Hz)          |
| Sample-Rate-Konvert.  | Mathematisch exakt, kein Spielraum         |
| Silence-Maske         | Binär (On/Off), keine Kalibrierung nötig   |

## Pre-Commit / Watchdog

Keine Änderungen nötig:

- `_GLOBAL_MAX_MS` ist bereits in `stereo_temporal_coherence_guard.py` definiert
- PostGate wird als neue Klasse in `backend/core/post_processing_gate.py` implementiert
- Keine bestehenden Watchdog-Regeln betroffen

## Implementierungs-Reihenfolge

1. ✅ STCG Universal Guard (v10.14, bereits umgesetzt)
2. ✅ Null-Lag-Integrationstest (tests/normative/test_null_lag_pass_through.py)
3. ✅ PostProcessingGate — Klasse + verify_quick (5 Ziele) + Lambda-Signatur-Guard (§v10.0.5)
4. 🔲 HumanizationPass — adaptive Stärke + PostGate-Integration
5. ✅ Post-Processing-Chain — alle Komponenten durch PostGate wrappen
6. 🔲 Polarity Inversion — verify via Korrelations-Δ
7. ✅ Tests aktualisiert (24 Contract-Tests in backend/tests/test_phase_contracts.py)

## §v10.0.5 Genre-Key-Normalisierung

**Problem:** `_restoration_context` speichert Genre unter Key `genre_label`,
aber Phasen lesen teils `kwargs.get("genre")` (Phase 19, 16, 54) oder
`kwargs.get("genre_label")`. Inkonsistenz führt zu leeren Genre-Strings
in der DeEsser-Kalibrierung.

**Lösung (zentral, kein Per-Phase-Patch):**

- `_restoration_context` enthält nun beide Keys `genre` UND `genre_label` (Alias)
- `_prepare_profiled_phase_context` normalisiert beide Keys via `setdefault()`,
  sodass ALLE Phasen unabhängig vom gelesenen Key den Genre-Wert erhalten
- Contract-Test `TestGenrePropagationContract` verifiziert die Propagation

## §v10.0.5 Vollständige Änderungsliste

### Bugfixes

| Bug | Datei | Fix |
|-----|-------|-----|
| Leerer Genre-String in DeEsser | `phase_19_de_esser.py:915` | `genre_label` Fallback + zentrale Normalisierung |
| Genre-Key-Mismatch | `unified_restorer_v3.py:27375-27381` | `genre`↔`genre_label` bidirektional via `setdefault()` |
| HPG/ListeningEQ Lambda-TypeError | `unified_restorer_v3.py:12420,13950` | `lambda a,sr,strength=None:` (3 args) |
| PostGate-Lambda-Signatur | `post_processing_gate.py:122-125,235-269` | `_validate_lambda()` via `inspect.signature` |
| Tuple-`ndim`-Crash in Phasen | `unified_restorer_v3.py:31797-31802` | `_normalize_phase_result`: erstes `np.ndarray` im Tuple suchen |
| Tuple-`ndim` Post-Phase Guard | `unified_restorer_v3.py:30604-30611,30656-30664` | `isinstance(result.audio, np.ndarray)` vor `.ndim` |

### Genre-Profile

| Änderung | Datei |
|----------|-------|
| `AMBIENT_RESTORATION_PROFILE` (12 Parameter) | `genre_classifier.py:2264-2280` |
| `WORLD_RESTORATION_PROFILE` (8 Parameter) | `genre_classifier.py:2282-2294` |
| `oper` +5 Parameter vervollständigt | `genre_classifier.py:2325-2330` |
| `schlager` +`compression_ratio_cap` | `genre_classifier.py:2147-2148` |
| 19/19 Profile in `GENRE_RESTORATION_PROFILES` + `get_restoration_profile()` | `genre_classifier.py` |

### Perceptual-Quality-Modifier

| Änderung | Datei |
|----------|-------|
| `_GENRE_QUALITY_MODIFIER`: 12 Genres ergänzt | `perceptual_quality_council.py:92-115` |

### Psychoakustik

| Änderung | Datei |
|----------|-------|
| `compute_adaptive_phon()` via LUFS | `fletcher_munson_curves.py:461-502` |
| `PsychoacousticConfig.fletcher_adaptive_phon` | `psychoacoustic_core.py:131-133` |
| Adaptive Phon in `apply_loudness_compensation` | `psychoacoustic_core.py:292-305` |

### Quality-Gates

| Änderung | Datei |
|----------|-------|
| Instrumental-Gate: `panns_orchestral ≥ 0.35` + `natreblichkeit < 0.70` → phase_65 | `unified_restorer_v3.py:17388-17409` |
| Exciter-Freigabe: `brillanz < 0.60 ∧ waerme ≥ 0.70` → phase_21 erlaubt | `unified_restorer_v3.py:33331-33348` |
| DeEsser-Skip: `HF < 0.005` → Phase komplett skippen | `phase_19_de_esser.py:758-778` |

### Performance

| Änderung | Datei |
|----------|-------|
| Wall-Time-Budget ×2 (2700→5400 etc.) | `unified_restorer_v3.py:33640-33656` |
| Overhead 1800→3600, Per-Sek 15→25 | `unified_restorer_v3.py:33667-33671` |
| RT-Limit 32→48× | `performance_guard.py:121-125` |
| Min-Effective-Strength-Guard (strength < 0.12 → Skip) | `unified_restorer_v3.py:28574-28585` |
| Skip-Guards (DR-Exp, Tape-Hiss, Transient, Bass) | `phase_26/29/36/37` |

### Goosebumps

| Änderung | Datei |
|----------|-------|
| `_GENRE_WEIGHTS` + `_GENRE_THRESHOLDS` (11 Profile) | `goosebumps_factor.py:69-100` |
| Genre-adaptives Scoring in `compute_goosebumps()` | `goosebumps_factor.py:108-134` |

### OneTakeExport

| Änderung | Datei |
|----------|-------|
| Iterative 2-Pass (`iterative=True`) | `one_take_export.py:60,98-118` |

### Export-Quality (Hidden Features aktiviert)

| Änderung | Datei |
|----------|-------|
| `fletcher_adaptive_phon=True` (Default aktiviert) | `psychoacoustic_core.py:133` |
| `apply_final_polish` in Export-Chain (Era-EQ + Dither) | `unified_restorer_v3.py:12357-12383` |

### Tests

| Änderung | Datei |
|----------|-------|
| 24 Contract-Tests (PostGate, Genre, UVR) | `backend/tests/test_phase_contracts.py` |
| 13 Genre-Universalitätstests | `tests/unit/test_genre_classifier.py` |
