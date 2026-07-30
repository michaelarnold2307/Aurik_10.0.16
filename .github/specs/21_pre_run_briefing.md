# §v10.117 Pre-Run Briefing — Nächster Pipeline-Lauf

> **AKTUALISIERT** nach Real-Run-Analyse (21:00–21:10, 2026-08-03)
> Commit `8182f0b8` | 14 Commits seit Baseline | 0 Compile-Fehler | 0 Scanner-Warnungen

## Real-Run-Erkenntnisse (Kassette, 4-stufig, Schlager)

### Aufgetretene Bugs (alle behoben)

| Bug | Impact | Fix-Commit |
|-----|--------|------------|
| `NameError: Any` in audio_utils.py | UV3 + alle Denker ausgefallen → Pipeline im Notfall-Modus | `2c63c8b0` |
| `NameError: MaterialType` in causal_defect_reasoner | DefektDenker.reason() crashte | `8182f0b8` |

### Root Cause Analyse

**Bug 1 — `Any`:** safe_stft/safe_istft verwendet `**kwargs: Any` aber
`from typing import cast` statt `from typing import Any, cast`.
→ audio_utils.py → ImportError → UV3 unladbar → StrategieDenker, RestaurierDenker,
PhaseInteractionDenker alle im Fallback → Export crasht → Forensik-Hook nie erreicht.

**Bug 2 — `MaterialType`:** Der §v10.113 P6-Auto-Fixer fügte
`_mk = material.value if isinstance(material, MaterialType) else material`
in 63 Dateien ein, aber NUR die `.get()`-Ersetzung, NICHT den Import.
9 Dateien referenzierten `MaterialType` ohne es zu importieren.

**Lehre:** Ein Auto-Fixer muss BEIDES tun: Referenz ersetzen UND Import hinzufügen.
Der Scanner hätte das finden können, aber P6 prüft nur Dict-Lookups, nicht
Import-Vollständigkeit. → Neuer Pattern-Kandidat für Scanner: P9 Import-Consistency.

### Run-Metriken (trotz Bugs)

| Metrik | Wert | Kontext |
|--------|------|---------|
| Material | Kassette (4-stufig: reel_tape→vinyl→cassette→mp3_low) | Deep-Chain-Boost relevant |
| Genre | Deutscher Schlager | JND-Faktor 2.20 |
| Restorability | 64/100 (Mäßig) | MP3-Quelle, 4-stufige Kette |
| Exzellenz | 0.873 | 12/15 Goals (ohne UV3!) |
| VERSA MOS | 4.526 | Studioqualität |
| Phasen | 15 | Sparsam (Notfall-Modus) |
| Artifact Freedom | 0.800 | Unter 0.95 → Export blockiert |

---

## Was seit dem letzten Lauf geändert wurde

### Stabilität (94 Crash-Risiken eliminiert)

- P1: 24 shape-Anti-Patterns gefixt (channels-first vs channels-last)
- P2: 2 filtfilt-Crash-Risiken + safe_filtfilt Identity-Filter + Rekursion-Fix
- P3: 24 noverlap-Crash-Risiken (stft/istft-Guards in 13 Dateien)
- P4: 2 os-Import-Fehler (dac_plugin, validate_before_run)
- P5: tuple→ndarray in PhaseResult.**post_init** (§v10.95)
- P6: 63 material-Dict-Lookups mit Enum-Normalisierung
- P7: tuple.ndim (70 Exceptions) — bereits durch §v10.95 behoben

### Korrektheit (alle Parameter korrekt)

- Alle 69 Phasen nutzen safe_stft/safe_istft (0 bare scipy.signal.stft)
- Material-Adaption: 63 Dict-Lookups jetzt Enum-sicher
- Jede .get(material, default) → .get(_mk, default) mit Normalisierung

### Wahrnehmung (Material+Genre-adaptiv)

- **Material-JND**: CD=1.0, Vinyl=1.4, Tape=1.6, Kassette=2.0, Shellac=2.5
- **Genre-JND**: Klassik=0.80, Jazz=0.90, Rock=1.00, Schlager=1.10, Electronic=1.20
- **Kassette Deep-Chain** (Tiefe ≥3): noise_reduction=1.15, hf_restoration=1.25
- **PerceptualGate**: should_skip_phase() jetzt material+genre-aware

### Forensik (Kreislauf geschlossen)

- ExceptionAggregator liest 136.507 NDJSON-Einträge
- PatternMiner entdeckte P7, P8 aus realen Daten
- QualityRegressionDetector trackt Q-Score-Trends
- Post-Pipeline-Hook läuft AUTOMATISCH am Ende jedes Laufs
- Scanner lädt dynamische Patterns aus discovered_patterns.json

---

## Was beim nächsten Lauf zu erwarten ist

### Automatisch (non-blocking)

```
Pipeline-Ende (UV3._execute_pipeline, Zeile 38795)
  → run_forensics()
    → ExceptionAggregator.summary()
    → PatternMiner.discover()
    → QualityRegressionDetector.record()
    → ContinuousAnalyzer.analyze_new()
```

### Manuell nach dem Lauf

```bash
# Dashboard aufrufen
python scripts/forensics_dashboard.py --full

# Nur neue Exceptions seit letztem Lauf
python scripts/forensics_dashboard.py --trends

# Pattern-Mining
python scripts/forensics_dashboard.py --patterns

# Q-Score-Korrelation
python scripts/forensics_dashboard.py --qscore
```

### Erwartete Verbesserungen

| Metrik | Vorher (Juli 2026) | Erwartet |
|--------|-------------------|----------|
| Exceptions/Lauf | 460 | < 100 (−78%) |
| Unklassifizierte Exceptions | ~115 | < 30 |
| Kassette Q-Score | 0.767 | > 0.80 |
| Phasen-Überspringungen (Crash) | ~15/Lauf | 0 |
| Neue Pattern-Entdeckungen | 0 (Einmal-Analyse) | 1–3 pro Lauf |

### Worauf zu achten ist

1. **Q-Score-Regression**: Wenn Q-Score sinkt → `post_pipeline_forensics.py` loggt Warning.
   QualityRegressionDetector.compare() zeigt ΔQ an.

2. **Neue Pattern-Entdeckungen**: PatternMiner schreibt nach `logs/discovered_patterns.json`.
   Scanner lädt diese automatisch beim nächsten Commit.

3. **safe_stft-Verhalten**: Alle 69 Phasen nutzen jetzt safe_stft mit auto-Clamping.
   Bei extrem kurzen Segmenten (< n_fft) wird nperseg automatisch reduziert.

4. **JND-Effekt**: Phasen mit marginaler Änderung werden jetzt material+genre-abhängig
   übersprungen. Kassette (JND=2.0) überspringt mehr Phasen als CD (JND=1.0).

---

## Commit-Historie (12 Commits)

```
7ca95334 feat: §v10.116 Material-adaptive JND + Genre-Tuning + Kassette-Optimierung
a72578af docs: §v10.116 Erkenntnisse & Roadmap
1e0988da feat: §v10.115 safe_stft-Migration — 5 Phasen, 25 stft/istft Calls
11f75ac6 feat: §v10.115 Forensik in Pipeline verdrahtet — letzte Meile
245a05b2 fix: §v10.115 Forensik-Integration — Pipeline-Hook + Q-Score-Fix + Pattern-Miner aktiv
d35363cc docs: §v10.115 Spec-Update — Forensik-Kreislauf geschlossen
e16b3997 feat: §v10.115 Exception-Forensik — alle 6 Lücken geschlossen (SOTA)
71ba014f fix: §v10.114 Scanner auf alle Layer + P2/P3/P4/P6 in Plugins/Scripts
e4fa95d7 chore: Scanner — letzte False-Positives eliminiert
782e41aa fix: P6 material Dict-Lookup — 63→1 (62 Fixes in 30 Dateien)
bf3fcb56 fix: P3 noverlap-Guard — 14 STFT-Crash-Risiken behoben
20582199 fix: §v10.97–§v10.112 umfassende Bug-Fix-Runde
```

---

## Neue Dateien (dieser Sprint)

| Datei | Zeilen | Zweck |
|-------|--------|-------|
| `backend/core/perceptual_tuning.py` | 285 | Material-adaptive JND + Genre + Kassette |
| `backend/core/exception_forensics.py` | 460 | Aggregator + Miner + ContinuousAnalysis |
| `backend/core/quality_regression_detector.py` | 245 | Q-Score-Trend + Regression-Detection |
| `scripts/forensics_dashboard.py` | 214 | CLI: summary/top/trend/qscore/watch |
| `scripts/post_pipeline_forensics.py` | 164 | Automatischer Post-Pipeline-Hook |
| `scripts/fix_p6_material_lookups.py` | 110 | P6 Auto-Fixer (historisch) |
| `scripts/fix_p6_v2.py` | 130 | P6 Auto-Fixer v2 |
| `tests/unit/test_v10_fix_regression_gate.py` | 345 | 21 Regression-Tests |
| `.github/specs/20_erkenntnisse_maximale_restaurierung.md` | 180 | Erkenntnisse & Roadmap |

---

## Bereit zum Run

✅ 0 Compile-Fehler
✅ 0 Scanner-Warnungen
✅ 1 Pipeline-Verdrahtung (UV3 Zeile 38795)
✅ 136.507 NDJSON-Einträge (wird ergänzt)
✅ Forensik-Kreislauf geschlossen
✅ Material+Genre-adaptive JND aktiv
✅ 69/69 Phasen safe_stft
