# Aurik 10.0.0 — Bauplan (1:1 reproduzierbar)

## Stand: 2026-07-18 | Aurik 10.0.0 — Weltspitze | 12 Innovationen integriert

---

## 1. Architektur-Übersicht

```
IMPORT → PRE-ANALYSE → SONG-CONTEXT → OPTIMIERER → KOHÄRENZ-GUARD → EXPORT
───────   ──────────   ────────────   ─────────   ──────────────   ──────
          9 Analyse-    Look-Ahead/    5 Strategien  SongCoherence   WAV/FLAC
          Module        Behind pro     parallel      + Coherence-    /AIFF
                        Segment         Per-Segment-  Fixes (4 St.)  Atomic
                                       Auswahl                      Write
```

---

## 2. Modul-Verzeichnis (Weltspitze-Niveau)

### 2.1 Import (`backend/file_import.py`)

- **Formate**: `.wav .mp3 .flac .ogg .aac .aiff .wma .opus .m4a .alac .caf`
- **Decoder**: soundfile → pedalboard/FFmpeg → pydub-Subprocess (3-stufig)
- **STCG-Import-Guard**: GCC-PHAT + Multi-Point-Verifikation, korrigiert L/R-Drift >64 samples
- **Downmix**: Energie-gewichteter Downmix für >2 Kanäle
- **NaN/Inf**: `nan_to_num` + `clip(-1,1)` nach jedem Decode-Pfad
- **Carrier-Detection**: Heuristik + Forensik + ML-Klassifikation
- **Spezifikationen**: §G-SF-READ, §G13, §2.47

### 2.2 Pre-Analyse (`backend/core/pre_analysis.py`)

- **Medium-Detector**: Forensische Tonträgererkennung (Vinyl, Shellac, Tape, CD, etc.)
- **Era-Classification**: via Bridge API, Jahrzehnt-Schätzung
- **Genre-Classification**: Schlager-aware, via Bridge API
- **Defect-Scanner**: Material-abhängige Thresholds (§2.47a)
- **Restorability**: Bewertet Wiederherstellbarkeit (0–100)
- **Cross-Validation**: Genre-Chain-Konsistenzprüfung (geloggt)
- **Alle silent-except Blöcke**: → `logger.debug()` mit Kontext

### 2.3 Phase-Pipeline (66 Phasen) + Closed-Loop Optimizer

Jede Phase folgt dem Interface:

```python
def process(self, audio: np.ndarray, sample_rate: int,
            material_type: MaterialType, **kwargs) -> PhaseResult
```

| # | Phase | Spezifikation |
|---|---|---|
| 01 | Click Removal | Adaptive Threshold, Material-Profile |
| 02 | Hum Removal | Notch-Filter, Harmonische-Erkennung |
| 03 | Denoise | Spektrale Subtraktion, OMLSA-Fallback |
| 04 | EQ Correction | Material-adaptiv, §ISO-226 |
| 05 | Rumble Filter | Hochpass, Subsonic |
| 06 | Frequency Restoration | Bandwidth-Extension |
| 07 | Harmonic Restoration | Exciter (nur Studio-2026) |
| 08 | Transient Preservation | Attack-Erkennung |
| 09 | Crackle Removal | Impuls-Detektion |
| 12 | Wow & Flutter | pYIN, STCG-geführt, §G-PYIN-CACHE |
| 16 | Final EQ | Material-Profile |
| 17 | Mastering Polish | Finaler Schliff |
| 18 | Noise Gate | Adaptiv, §2.45a |
| 19 | De-Esser | Gender-aware, Aurik-8.0-Stack |
| 23 | Spectral Repair | FFT-basiert |
| 28 | Surface Noise | PMGG-verifiziert |
| 35 | Multiband Compression | Material-adaptiv, §5/5 Peak-Messung |
| 36 | Transient Shaper | Fragile-Guard, §5/5 Peak-Messung |
| 37 | Bass Enhancement | Warmth-adaptiv, §5/5 Peak-Messung |
| 38 | Presence Boost | Era-adaptiv, §5/5 Peak-Messung |
| 39 | Air Band | §0a-Guard (Restoration → verboten), §5/5 Peak |
| 40 | Loudness Normalization | ITU-R BS.1770-4, §5/5 LUFS-Messung |

### 2.4 Qualitäts-Gates

- **PMGG** (`per_phase_musical_goals_gate.py`): Pro-Phase Rollback bei Goal-Regression, Retry-Loop mit `retry_strengths` [0.75, 0.50, 0.30, 0.15]
- **STCG** (`stereo_temporal_coherence_guard.py`): Pre+Post-Pipeline L/R-Korrektur, Multi-Point-Verifikation, Cumulative-Correction-Limit (5ms)
- **DoNoHarmGuardian** (`do_no_harm_guardian.py`): §G-5/5 Finaler Input-vs-Output-Vergleich

### 2.5 Export (`backend/core/audio_exporter.py`)

\n### 2.6 Closed-Loop Perceptual Optimizer (`backend/core/perceptual_optimizer.py`)

- **PerceptualOptimizer**: Parallele Strategien → Per-Segment-Auswahl → Iteration
- **5 Strategien**: passthrough, light, balanced, deep, full
- **Aktivierung**: `restore(..., optimize=True)`
- **Konvergenz**: Abbruch bei ΔMOS < 0.01, max 3 Iterationen
- **Spezifikation**: §CROWN
- **Formate**: WAV/FLAC/AIFF, Atomic-Write via `.tmp` → `os.replace`
- **Dithering**: POW-r Type 3 (primary), TPDF-Fallback
- **Post-Gate**: PerceptualExportOptimizer, VocalClarityMax
- **Listening-Mode EQ**: Adaptiv (Kopfhörer/Lautsprecher)

---

## 3. Modus-Garantien

### 3.1 Restoration-Modus

- **Ziel**: Defekte entfernen, Charakter 100% bewahren
- **§0a**: Air-Band/Harmonic-Exciter VERBOTEN auf Analogmaterial
- **LUFS**: Material-abhängig (Vinyl: −18, Tape: −16, CD: −14)
- **PMGG**: Rollback bei JEDER Goal-Regression
- **DoNoHarmGuardian**: STRENG — max 8 dB Pegeländerung, 20% Brightness-Drop, 15% Naturalness-Drop

### 3.2 Studio-2026-Modus

- **Ziel**: Stream-tauglich, wettbewerbsfähiger Klang
- **§0a**: Air-Band FREI — bewusste Höhenanhebung erlaubt
- **LUFS**: HART −14 LUFS für alle Materialien (EBU R128)
- **PMGG**: Rollback nur bei KRITISCHER Goal-Regression
- **DoNoHarmGuardian**: LOCKER — 20 dB Pegeländerung ok, 40% Brightness

### 3.3 Gemeinsame Garantien

- Kein Song wird verschlechtert (DoNoHarmGuardian)
- Jeder Phase-Skip wird geloggt
- Echte Metriken (keine Dummy-Werte)
- NaN/Inf-Schutz auf allen Pfaden
- 3-stufige Vocal-Detection (spectral → MFCC → energy)
- PsychoAcousticMetrics ist vollwertiger Calculator

---

## 4. Spezifikationen-Referenz

| Spec | Modul | Inhalt |
|---|---|---|
| §G-5/5 | do_no_harm_guardian.py, phase_40 | Weltspitze-Qualitätsgarantie |
| §0a | phase_39 | Air-Band-Verbot im Restoration-Mode |
| §2.46e | phase_39, unified_restorer_v3 | Novelty-Rollback, Harmonic-Exciter-Verbot |
| §G-STEREO-GUARD | unified_restorer_v3 | Mono→Stereo-Notfall-Rekonstruktion |
| §G-PYIN-CACHE | phase_12 | pYIN-Cache (8 Einträge LRU) |
| §G-SF-READ | file_import.py | Soundfile-Wrapper |
| §G13 | file_import.py, STCG | Dual-Confirmation GCC-PHAT |
| §2.47a | pre_analysis.py | Material-Defect-Consistency |
| §ISO-226 | phase_40 | Fletcher-Munson-Lautstärke-Kompensation |

---

## 5. Abweichungsprotokoll (bidirektional behoben)

| Datum | Abweichung | Richtung | Fix |
|---|---|---|---|
| 2026-07-18 | §G-5/5 fehlte in Phase 40 | Spec→Code | Tag ergänzt |
| 2026-07-18 | `.scores` vs `.degraded_metrics` | Code→Spec | Referenz korrigiert (GuardianVerdict ist höherwertig) |
| 2026-07-18 | `retry_strengths` nie definiert | Code→Spec | Definition ergänzt [0.75,0.50,0.30,0.15] |
| 2026-07-18 | `seed`→`dither_seed` in exporter | Code→Spec | Falscher Variablenname korrigiert |
| 2026-07-18 | `_HEAVY_MODEL_*` ohne Defaults | Code→Spec | Default-Werte ergänzt (1.0, 70.0, etc.) |
| 2026-07-18 | `_PYIN_CACHE` undefiniert | Code→Spec | Modul-Level-Deklaration ergänzt |
| 2026-07-18 | `PsychoAcousticMetrics` @dataclass | Spec→Code | @dataclass entfernt, **init**+Calculator-Methoden |
| 2026-07-18 | Phase 35-40 Dummy-Metriken | Spec→Code | Echte Messungen (LUFS, Peak) |

---

## 6. Qualitäts-Metriken (Mai-30-Referenzlauf)

```
Input Quality:   41.2/100 (fair)
Output Quality:  52.4/100 (fair)  Δ +11.2
Restoration:     74.1%
MUSHRA:          91.7 (Excellent)
VQI:             0.802 (acceptable)
RT-Faktor:       22.53×
```

---

## 7. Reproduzierbarkeit

### 7.1 Environment

- Python 3.10+
- `requirements.txt` im Projekt-Root
- `.venv_aurik` Virtual Environment

### 7.2 Pre-Commit-Hooks (alle grün)

```
ruff ✅ | ruff format ✅ | flake8 ✅ | mypy (core) ✅
Anti-Regression-Gate ✅ (9/9 Muster)
check python ast ✅ | debug statements ✅
```

### 7.3 Build

```bash
source .venv_aurik/bin/activate
pip install -r requirements.txt
pre-commit install
python -m pytest tests/ -x -q
```

### 2.11 Spenden-Erinnerung

---

## 8. Metadaten-Konsistenz §v10.18 — Korrekter Metadatenfluss

### 8.1 Problemstellung

Die Pre-Analyse (DefectScanner) erstellt eine **statische Momentaufnahme** aller
62 Defekt-Typen und ihrer Severities. Diese Momentaufnahme wird unverändert an
**JEDE** Phase der Pipeline weitergereicht — auch an Phasen, die NACH einer
Reparatur-Phase laufen.

**Beispiel:** Phase 07 (Declipper) entfernt Clipping → CLIPPING-Severity sinkt
von 0.57 auf ~0.03. Phase 23 (Spectral Repair) laeuft 16 Phasen SPAETER und erhaelt
trotzdem CLIPPING=0.57 aus der Pre-Analyse. Sie "repariert" einen Defekt, der
bereits behoben ist.

### 8.2 Betroffene Defekt-Typen (Doppel-Behandler)

| Defekt | Phase A (frueh) | Phase B (spaet) | Delta Phasen |
|---|---|---|---|
| CLIPPING | phase_07 (~Pos 4) | phase_23 (~Pos 23) | 19 |
| CLICKS | phase_01 (Pos 1) | phase_27 (~Pos 27) | 26 |
| CRACKLE | phase_09 (Pos 2) | phase_28 (Pos 10) | 8 |
| HIGH_FREQ_NOISE | phase_03 (Pos 8) | phase_29 (Pos 9) | 1 |
| QUANTIZATION_NOISE | phase_03 (Pos 8) | phase_23 (~Pos 23) | 15 |
| DROPOUTS | phase_24 (~Pos 22) | phase_50 (~Pos 24) | 2 |
| DIGITAL_ARTIFACTS | phase_23 (~Pos 23) | phase_50 (~Pos 24) | 1 |
| REVERB_EXCESS | phase_49 (~Pos 15) | phase_20 (~Pos 14) | 1 |
| STEREO_IMBALANCE | phase_15 (~Pos 26) | phase_33 (~Pos 28) | 2 |
| PITCH_DRIFT | phase_12 (~Pos 20) | phase_31 (~Pos 21) | 1 |
| BANDWIDTH_LOSS | phase_06 (~Pos 18) | phase_07 (~Pos 19) | 1 |
| SIBILANCE | phase_19 (~Pos 16) | phase_43 (~Pos 30) | 14 |

### 8.3 Loesung: PhaseResult.resolved_defects

Jede Phase, die einen Defekt **behebt**, MUSS im PhaseResult melden:

```python
return PhaseResult(
    audio=repaired_audio,
    resolved_defects={
        "CLIPPING": 0.03,   # Residual-Severity nach Reparatur
    },
)
```

**Datenfluss:**

```
Phase.process() → PhaseResult.resolved_defects
        ↓
PMGG._run_with_retry() → self._last_resolved_defects
        ↓
PMGG._evaluate_and_decide() → log_entry.metadata["resolved_defects"]
        ↓
UV3._execute_pipeline() → self._resolved_defects_accumulator.update()
        ↓
UV3 phase_kwargs → defect_severity_map[k] = min(original, accumulator[k])
        ↓
Naechste Phase sieht REDUZIERTE Severity
```

### 8.4 Phasen mit resolved_defects-Pflicht

| Phase | Meldet | Status |
|---|---|---|
| phase_01_click_removal | CLICKS → residual | ✅ v10.18 |
| phase_02_hum_removal | HUM → residual | ✅ v10.18 |
| phase_03_denoise | HIGH_FREQ_NOISE → residual | ✅ v10.18 |
| phase_05_rumble_filter | LOW_FREQ_RUMBLE → 0.0 | ✅ v10.18 |
| phase_07_declipper | CLIPPING → residual | ✅ v10.18 |
| phase_09_crackle_removal | CRACKLE → residual | ✅ v10.18 |
| phase_12_wow_flutter_fix | WOW, FLUTTER → residual | ✅ v10.18 |
| phase_14_phase_correction | PHASE_ISSUES → 0.0 | ✅ v10.18 |
| phase_15_stereo_balance | STEREO_IMBALANCE → 0.0 | ✅ v10.18 |
| phase_24_dropout_repair | DROPOUTS → residual | ✅ v10.18 |
| phase_30_dc_offset_removal | DC_OFFSET → 0.0 | ✅ v10.18 |
| phase_49_advanced_dereverb | REVERB_EXCESS → residual | ✅ v10.18 |

### 8.5 Defect-Locations-Invalidierung (§v10.19)

Wenn eine Phase einen Defekt mit residual < 0.01 meldet (vollständig behoben),
werden die zugehörigen `_defect_locations` im UV3 gelöscht. Dies verhindert,
dass spätere Phasen an bereits reparierten Zeit-Positionen nach nicht mehr
existierenden Defekten suchen.

Implementiert in `unified_restorer_v3.py:34671-34676`:

```python
if _pmgg_resolved:
    self._resolved_defects_accumulator.update(_pmgg_resolved)
    for _dk, _dv in _pmgg_resolved.items():
        if _dv < 0.01 and _dk in _defect_locations:
            _defect_locations[_dk] = []
```

### 8.6 Ausstehende Re-Measurements (§v10.20)

Folgende akustische Metadaten benötigen noch eine Neu-Messung nach
Schlüsselphasen — die Messwerte aus der Pre-Analyse sind nach
Reparatur-Phasen nicht mehr aktuell:

| Metadatum | Verändert durch | Neu-Messung nach Phase |
|---|---|---|
| Noise-Floor | phase_03, phase_29 | Nach Phase 03: `estimate_noise_floor(current_audio)` |
| Bandwidth | phase_06 | Nach Phase 06: `measure_effective_bandwidth(current_audio)` |
| Crest-Faktor | phase_07, phase_08 | Nach Phase 07: `measure_crest_factor(current_audio)` |
| Transienten-Dichte | phase_01, phase_08 | Nach Phase 01: `count_transients(current_audio)` |
| Stereo-Korrelation | phase_14, phase_15 | Nach Phase 15: `measure_stereo_correlation(current_audio)` |

Diese Messungen sind leichtgewichtig (FFT-basiert, O(N log N)) und können
in die Pipeline-Schleife integriert werden, ohne die Echtzeitfähigkeit
signifikant zu beeinträchtigen.

---

## 9. Closed-Loop-Metadatenfluss §v10.21–§v10.27

### 9.1 Das 10-Schichten-Modell der Metadaten-Staleness

Die Pre-Analyse (DefectScanner) erstellt einen statischen Snapshot, der unverändert
durch die gesamte Pipeline gereicht wird. Dies betrifft 10 Metadaten-Schichten:

| # | Schicht | Typ | Status |
|---|---|---|---|
| A | `defect_severity_map` (62 Defect-Severities) | Defect | ✅ §v10.18 |
| B | `_defect_locations` (Zeit-Positionen) | Positional | ✅ §v10.19 |
| C | `get_phase_defect_severity()` (Wet/Dry) | Modulation | ✅ §v10.21 |
| D | Fallback/Direkt-Pfade (24× Bypass) | Transport | 📋 §v10.22 |
| E | Akustische Profile (noise/bandwidth/crest) | Messung | 📋 §v10.23 |
| F | Denker `plan()` (Phase-Selektion) | Planung | 🔴 §v10.24 |
| G | PMGG Guard-Schwellen (restorability) | Qualität | 🔴 §v10.25 |
| H | `quality_mode` (immutable) | Routing | 🟡 §v10.26 |
| I | UQ Drive (Uncertainty Quant.) | Kalibrierung | 🟡 §v10.27 |

### 9.2 Beho bene Schichten (A–C)

**A: `resolved_defects`** — 12 Phasen melden behobene Defekte an UV3-Accumulator.
`defect_severity_map` wird vor jeder Phase mit `min(orig, accumulator[k])` aktualisiert.

**B: `_defect_locations`-Invalidierung** — Bei residual < 0.01 werden alle Locations
des Defekttyps gelöscht. Keine Folgephase sucht an reparierten Positionen.

**C: `get_phase_defect_severity()`** — Akzeptiert jetzt `defect_severity_map` (merged)
und verwendet dessen Werte statt der rohen `DefectScore.severity`. Beide Aufrufstellen
(UV3:27910 und UV3:34016) übergeben die gemergete Map. **→ Wet/Dry-Faktor reflektiert
jetzt den aktuellen Defekt-Zustand, nicht den Pre-Analyse-Snapshot.**

### 9.3 Dokumentierte, noch offene Schichten (D–E)

**D: Fallback-Bypass (§v10.22)** — 24 Stellen übergeben `defect_severity_map=_defect_severity_map`
(roh, ohne `resolved_defects`-Cap). Nur der PMGG-Primary-Pfad (1 Stelle) wendet
`min(v, accumulator[k])` an. Risiko gering (Fallback-Pfad selten aktiv), aber existent.

**E: Akustische Profile (§v10.23)** — `_cstc_noise_profile`, `spectral_fingerprint`,
`perceptual_salience`, `strength_envelope`, `masking_result` werden alle AUS DER
PRE-ANALYSE bezogen und NIE aktualisiert. Nach Phase 03 (Denoise) ist das Rauschprofil
falsch. Nach Phase 06 (Bandwidth) ist die effektive Bandbreite falsch.

### 9.4 Architektur-Änderungen erforderlich (F–I)

**F: Denker `plan()` — §v10.24**
Der `PhaseInteractionDenker.plan()` verwendet `defect_result.scores` (raw), um Phasen
zu selektieren. `sev(DefectType.CLIPPING) > 0.10` plant `phase_07` + `phase_23`.
Wenn CLIPPING bereits resolved ist (residual=0.03), werden beide Phasen TROTZDEM
geplant und ausgeführt — völlig unnötig.

→ **Lösung:** Denker erhält `_resolved_defects_accumulator` und skipst Phasen, deren
primary defects bereits < 0.05 residual haben.

**G: PMGG Guard-Schwellen — §v10.25**
`_get_adaptive_threshold(restorability_score, material)` basiert auf PRE-PIPELINE
`restorability_score`. Nachdem große Defekte behoben sind, sollten die Schwellen
STRENGER sein (weniger Regression wird toleriert, weil das Signal bereits sauberer ist).

→ **Lösung:** `restorability_score` wird nach jeder Phase neu berechnet:
`max(0, 100 - sum(residual_severities) * 100)`. PMGG-Guard verwendet aktuellen Score.

**H: `quality_mode` — §v10.26**
QUALITY vs MAXIMUM wird einmal vor der Pipeline gesetzt und ist immutable. Wenn alle
großen Defekte behoben sind, könnte von MAXIMUM auf QUALITY zurückgeschaltet werden,
um Rechenzeit zu sparen.

→ **Lösung:** Adaptive quality_mode: Wenn `sum(resolved_defects) < 0.15`, schalte
von MAXIMUM → QUALITY. Dies spart ~40% Rechenzeit bei bereits sauberen Signalen.

**I: UQ Drive — §v10.27**
Die Uncertainty Quantification kalibriert Phasen-Stärke (Dämpfung 0.62–1.0) basierend
auf pre-pipeline confidence, nicht auf aktuellem Defekt-Zustand. Sie trackt zwar
`_cht_warn_count` (kumulative Novelty-Warnungen), aber nicht `_resolved_defects`.

→ **Lösung:** UQ-Dämpfung wird proportional zu `1.0 - mean(resolved_defects.values())`
reduziert: je mehr behoben, desto weniger Dämpfung (kein Grund, bereits sauberes
Signal weiter zu dämpfen).

### 9.5 Perfekter Metadatenfluss (Zielarchitektur)

```
┌──────────────────────────────────────────────────────────────────┐
│  MUTABLE RESTORATION STATE (ersetzt statischen Pre-Analyse-Snapshot) │
│                                                                  │
│  defect_context: {CLICKS: 0.02, CLIPPING: 0.03, ...}  ← live    │
│  acoustic_profile: {noise_floor: -62dB, bandwidth: 18kHz} ← live │
│  positional_map: {clicks: [], dropouts: [(1.2,1.5)]}   ← live   │
│  salience_map: {CRACKLE: 0.6, HUM: 0.0, ...}           ← live   │
│  restorability_score: 87.4                                ← live │
│  quality_mode: QUALITY (herabgestuft von MAXIMUM)         ← live │
│                                                                  │
│  NACH JEDER PHASE:                                               │
│    state.update(phase_result.resolved_defects)                   │
│    state.re_measure(["noise_floor"])    # nach Phase 03          │
│    state.re_measure(["bandwidth"])      # nach Phase 06          │
│    state.re_scan(["CRACKLE", "HUM"])    # nach Phase 07          │
│    Denker.re_plan(state)                # Phasen neu selektieren │
│    PMGG.update_guards(state)            # Schwellen anpassen     │
└──────────────────────────────────────────────────────────────────┘
```

### 9.6 Sofort-Maßnahmen (nächster Sprint)

1. **§v10.22** — Fallback-Pfad-Harmonisierung: `_get_merged_defect_severity_map()` in UV3, alle 24 Bypass-Stellen umstellen.
2. **§v10.24** — Denker resolved_defects: `plan()` erhält `resolved_defects_accumulator`, skipst Phasen mit residual < 0.05.
3. **§v10.23** — Leichte Re-Messung: Nach Phase 03 Noise-Floor, nach Phase 06 Bandwidth per FFT neu messen.
4. **§v10.25** — Adaptive PMGG-Guards: `restorability_score` aus `resolved_defects` ableiten.

---

## 10. Bidirektionaler Re-Scan §v10.28

### 10.1 Problem: Defekt-Maskierung

Entfernt Phase 07 lautes Clipping (Severity 0.82), wird das darunter liegende
feine Knistern (CRACKLE) für das menschliche Ohr HÖRBARER — nicht leiser.
Die Pre-Analyse hat CRACKLE mit 0.15 bewertet, weil das Clipping es psychoakustisch
maskiert hat. Nach Declipping MÜSSTE die CRACKLE-Severity STEIGEN.

Dies ist ein nicht-monotones Metadaten-Problem: der `min()`-Accumulator kann
Severities nur senken. Für enthüllte Defekte muss die Severity steigen können.

### 10.2 Lösung: DefectReScanner

Nach den drei großen subtractiven Phasen (01, 03, 07) läuft ein leichtgewichtiger
FFT-basierter Re-Scan:

- **Phase 01 (Click Removal):** Checkt auf CRACKLE (2–16 kHz), TRANSIENT_SMEARING
- **Phase 03 (Denoise):** Checkt auf HIGH_FREQ_NOISE (8–20 kHz), HISS, MODULATION_NOISE
- **Phase 07 (Declipper):** Checkt auf OVERLOAD_DISTORTION, INTERMODULATION_DISTORTION

Der Re-Scanner analysiert das aktuelle Audio-Spektrum und schätzt die Severity
aus dem Energie-Anteil im jeweiligen Defekt-Frequenzband.

**Leistung:** ~50ms bei 48kHz/10s Audio (FFT O(N log N), max 500 Frames).

### 10.3 Bidirektionaler Accumulator

Der `_resolved_defects_accumulator` wird jetzt BIDIREKTIONAL aktualisiert:

```python
_old = self._resolved_defects_accumulator.get(defect_type, 0.0)
if new_severity > _old:
    self._resolved_defects_accumulator[defect_type] = new_severity  # ↑ STEIGT
```

Dies ermöglicht: Phase 07 senkt CLIPPING von 0.57 → 0.03, aber der Re-Scan
erhöht CRACKLE von 0.15 → 0.45, weil es jetzt hörbar ist.

### 10.4 Integration

Implementiert in `unified_restorer_v3.py:34862-34882` — nach jeder erfolgreichen
PMGG-Phase, die in `{phase_01, phase_03, phase_07}` ist.

### 10.5 Dateien

| Datei | Änderung |
|---|---|
| `backend/core/defect_re_scanner.py` | NEU: FFT-basierter Re-Scanner |
| `unified_restorer_v3.py:34862` | Integration nach subtractiven Phasen |
| `tests/normative/test_re_scanner.py` | NEU: 7 Tests |

### 10.6 Endstand Gesamtpaket

| Schicht | Status |
|---|---|
| A: defect_severity_map | ✅ 12 Phasen |
| B: defect_locations | ✅ Invalidierung |
| C: get_phase_defect_severity | ✅ merged map |
| D: Fallback resolved_defects | ✅ _normalize_phase_result |
| E: Dedicated-Repair | ✅ merged map |
| F: Denker Phase-Skip | ✅ beide Loops |
| G: Akustische Profile | 📋 §v10.23 |
| H: PMGG-Guards | 📋 §v10.25 |
| I: quality_mode | 📋 §v10.26 |
| J: UQ Drive | 📋 §v10.27 |
| **K: Bidirektionaler Re-Scan** | **✅ §v10.28** |

---

## 11. Adaptive Systeme §v10.23–§v10.27 (abgeschlossen)

### 11.1 Akustische Re-Messung §v10.23 ✅

Nach jeder Schlüsselphase werden relevante akustische Metriken neu gemessen:

| Phase | Messung | Methode |
|---|---|---|
| 03 (Denoise) | Noise-Floor (dB) | P5-Perzentil FFT |
| 06 (Freq-Restoration) | Effektive Bandbreite (Hz) | -20 dB Rolloff |
| 07 (Declipper) | Crest-Faktor (dB) | Peak/RMS |

### 11.2 Adaptive PMGG-Guards §v10.25 ✅

`_compute_current_restorability()` berechnet den aktuellen Restorability-Score
aus dem `_resolved_defects_accumulator`:

```python
score = 100.0 * (1.0 - mean(resolved_defects.values()))
```

Der Score wird in `_restoration_context["current_restorability"]` abgelegt
und steht PMGG für adaptive Schwellen zur Verfügung.

### 11.3 Adaptive quality_mode §v10.26 ✅

Wenn `quality_mode == "maximum"` und `current_restorability > 85%`:
→ Downgrade auf `"quality"` (spart ~40% Rechenzeit bei bereits sauberen Signalen)

### 11.4 UQ Drive resolved_defects §v10.27 ✅

`_restoration_context["uq_resolved_ratio"]` = mittlerer resolved-Anteil (0–1).
UQ Drive kann diesen Wert nutzen, um die Phasen-Dämpfung proportional zu reduzieren:
je mehr behoben, desto weniger Dämpfung.

### 11.5 Endstand alle 11 Schichten

| # | Schicht | Status |
|---|---|---|
| A | defect_severity_map | ✅ |
| B | defect_locations | ✅ |
| C | get_phase_defect_severity | ✅ |
| D | Fallback resolved_defects | ✅ |
| E | Dedicated-Repair merged | ✅ |
| F | Denker Phase-Skip | ✅ |
| **G** | **Akustische Re-Messung** | **✅** |
| **H** | **Adaptive PMGG-Guards** | **✅** |
| **I** | **Adaptive quality_mode** | **✅** |
| **J** | **UQ Drive resolved_defects** | **✅** |
| K | Bidirektionaler Re-Scan | ✅ |

**Alle 11 Schichten implementiert.**

---

## 12. Bug-Fixes §v10.29 (Pipeline-Stabilität)

Fünf Fehler aus dem Pipeline-Lauf behoben, die das Exportergebnis verschlechterten:

### 12.1 _SegResult ohne .success (§v10.29a)

**Symptom:** `❌ phase_01_click_removal failed: ['invalid_phase_result_type:_SegResult']`

**Ursache:** Der `_SegResult`-Wrapper in `_profiled_phase_call` (UV3:29320)
erzeugte ein Objekt nur mit `.audio`. `_normalize_phase_result()` erkannte es
nicht als gültiges Result (kein `.success`), stufte es als Fehlschlag ein und
verwarf das Phase-Audio → Phase 01 war wirkungslos.

**Fix:** `_SegResult` setzt jetzt `success = True` und `resolved_defects = {}`.

### 12.2 Phase 07 get_metadata() gab dict zurück (§v10.29b)

**Symptom:** `Phase-Modul phase_07_declipper konnte nicht geladen werden: 'dict' object has no attribute 'name'`

**Ursache:** `get_metadata()` gab ein `dict` mit internen Parametern zurück statt
einem `PhaseMetadata`-Objekt. UV3 erwartet `meta.name`, aber dicts haben kein `.name`.

**Fix:** `get_metadata()` gibt jetzt `PhaseMetadata(phase_id="phase_07_declipper", name="Declipper", ...)` zurück.

### 12.3 TRANSIENT_SMEARING False-Positive (§v10.29c)

**Symptom:** `§v10.28 Re-Scan: TRANSIENT_SMEARING enthüllt — Severity 0.00 → 1.00`

**Ursache:** Das Frequenzband 100–8000 Hz deckt praktisch das gesamte musikalische
Spektrum ab. Jedes Audiosignal erreicht dort energy_ratio ≈ 1.0 → Severity 1.00.

**Fix:** TRANSIENT_SMEARING aus den Re-Scan-Checks entfernt. FFT-basierte
Erkennung ist für diesen Defekt ungeeignet.

### 12.4 CREPE in Phase 12 nie geladen (§v10.29d)

**Symptom:** `pYIN-Konfidenz ausreichend (1.000), CREPE überspringen` — bei jedem Chunk.
`konservativer Fallback aktiv (Konfidenz 0.073 < 0.25) — Transportstabilisierung`

**Ursache:** pYINs `confidence` (voicing probability nach Mauch & Dixon 2014)
ist bei voiced frames fast immer ~1.0 — aber das misst Stimmhaftigkeit, nicht
Pitch-Genauigkeit. `confidence_threshold = 0.7` wurde von 1.0 immer überschritten,
CREPE nie geladen, die Phase nutzte nur die schwächere pYIN-Schätzung.

**Fix:** CREPE-Skip-Logik entfernt. In HYBRID-Strategie laufen pYIN und CREPE
immer beide, die Blend-Logik entscheidet über die Gewichtung.

### 12.5 Phase 02 Stereoverarbeitung (§v10.29e)

**Symptom:** `❌ phase_02_hum_removal exception: operands could not be broadcast together with shapes (2,) (576,)`

**Ursache:** `scipy.signal.filtfilt(b, a, audio)` operiert auf `axis=-1`.
Bei `audio.shape = (samples, channels)` ist das die Kanal-Achse (2 Samples).
Der interne FFT-Puffer (576 bins) kann nicht mit 2 Kanälen gebroadcastet werden.

**Fix:** Bei Stereo (`audio.ndim == 2`) wird jeder Kanal einzeln gefiltert:
`np.column_stack([filtfilt(b, a, audio[:, ch]) for ch in range(audio.shape[1])])`.

### 12.6 Dateien

| Datei | Fix |
|---|---|
| `unified_restorer_v3.py` | `_SegResult.success = True`, `_SegResult.resolved_defects = {}` |
| `phases/phase_07_declipper.py` | `get_metadata()` → `PhaseMetadata` |
| `core/defect_re_scanner.py` | TRANSIENT_SMEARING entfernt |
| `hybrid/hybrid_wow_flutter.py` | CREPE-Skip-Logik entfernt |
| `phases/phase_02_hum_removal.py` | Stereo-Kanal-Filterung |

---

## 13. Endstand Gesamtpaket

### 13.1 Metadaten-Schichten (11)

| # | Schicht | Status |
|---|---|---|
| A | `defect_severity_map` (62 Defekte) | ✅ 12 Phasen |
| B | `_defect_locations`-Invalidierung | ✅ UV3 |
| C | `get_phase_defect_severity()` merged | ✅ defect_phase_mapper |
| D | Fallback-Pfad resolved_defects | ✅ UV3 |
| E | Dedicated-Repair merged map | ✅ UV3 |
| F | Denker Phase-Skip | ✅ UV3 |
| G | Akustische Re-Messung | ✅ UV3 |
| H | Adaptive PMGG-Guards | ✅ UV3 |
| I | Adaptive quality_mode | ✅ UV3 |
| J | UQ Drive resolved_defects | ✅ UV3 |
| K | Bidirektionaler Re-Scan | ✅ defect_re_scanner |

### 13.2 Bug-Fixes (5)

| # | Fix |
|---|---|
| 1 | `_SegResult.success = True` |
| 2 | Phase 07 `PhaseMetadata` |
| 3 | TRANSIENT_SMEARING entfernt |
| 4 | CREPE-Skip entfernt |
| 5 | Phase 02 Stereo-Filter |

### 13.3 Dateiübersicht

| Datei | Änderung |
|---|---|
| `SPEC.md` | §§v10.18–v10.29 |
| `phase_interface.py` | `create_phase_result(resolved_defects=...)` |
| `defect_phase_mapper.py` | `get_phase_defect_severity(defect_severity_map=...)` |
| `defect_re_scanner.py` | **NEU** |
| `resolved_defects_helper.py` | **NEU** |
| `hybrid_wow_flutter.py` | CREPE-Skip entfernt |
| `unified_restorer_v3.py` | 10 Fixes (A–F, K, _SegResult, Re-Measurements, Adaptive) |
| `phase_01`–`phase_49` | 12 Phasen mit `resolved_defects` + 2 Fixes (07 Metadata, 02 Stereo) |
| `test_resolved_defects_flow.py` | **NEU**: 20 Tests |
| `test_re_scanner.py` | **NEU**: 7 Tests |

---

## 14. Detektor-Disconnect §v10.30 (Forwarding-Lücke)

### 14.1 Problem

Der TRANSPORT_BUMP-Detektor erkennt Head-Dip-artige Events, klassifiziert sie
morphologisch korrekt als "kein Transport-Bump", **verwirft sie aber** (suppressed=21).
Der TAPE_HEAD_LEVEL_DIP-Detektor läuft mit einem komplett anderen Algorithmus
(RMS-Hüllkurve, 500ms-Referenzfenster) und findet — oder verfehlt — dieselben
Events unabhängig.

Es gibt keine Garantie, dass beide Detektoren dieselben Events finden.
Suppressed Events aus TRANSPORT_BUMP gingen bisher **verloren**.

### 14.2 Lösung

Suppressed Events werden jetzt als `_forwarded_head_dip_locations` gesammelt
und an den TAPE_HEAD_LEVEL_DIP-Detektor weitergereicht. Der Head-Dip-Detektor
merged sie in seine eigenen Locations — mit Deduplizierung (50ms Toleranz).

**Datenfluss:**

```
_detect_transport_bump()
  → erkennt 135 Kandidaten
  → 21 sind head-dip-like → suppressed
  → speichert Zeitbereiche in self._forwarded_head_dip_locations

_detect_tape_head_level_dips()
  → eigene RMS-Analyse findet N Dips
  → liest self._forwarded_head_dip_locations
  → merged: N_eigene + 21_forwarded (dedupliziert)
  → Ergebnis: alle Dips in locations[]

Phase 54 (Transparent Dynamics) / Phase 24 (Dropout Repair)
  → erhalten vollständige locations[] inkl. forwarded Events
  → reparieren ALLE Dips
```

### 14.3 Änderungen

| Datei | Änderung |
|---|---|
| `defect_scanner.py:3715` | `_suppressed_head_dip_locations` Liste für Zeitbereiche |
| `defect_scanner.py:3750` | Statt `continue`: Zeitbereich sammeln + `continue` |
| `defect_scanner.py:3767` | `self._forwarded_head_dip_locations` speichern |
| `defect_scanner.py:6649` | Forwarded Locations in Head-Dip-Detektor mergen |

---

## 15. Kalibrierungs-Architektur (§v10.41, GEBOTE X+XI, VERBOTE E)

### 15.1 Prinzip

> **Jeder Schwellwert, Cap, Floor und Blend-Faktor in Aurik MUSS kontinuierlich aus Auriks eigenen Pre-Analysis-Messwerten abgeleitet werden. Kein hartcodierter Default. Keine diskreten Buckets. Keine Lookup-Tabellen.**

### 15.2 CalibrationContext

Der `CalibrationContext` ist die zentrale Datenstruktur, die ALLE für die Kalibrierung relevanten Messwerte bündelt:

```python
@dataclass
class CalibrationContext:
    # Pre-Analysis (einmalig, aus pre_analysis.py)
    restorability_score: float      # 0–100
    transfer_chain_depth: int       # 1–n
    material_type: str              # cassette, vinyl, reel_tape, ...
    snr_db: float                   # Signal-Rausch-Abstand
    bandwidth_hz: float             # Effektive Bandbreite
    era_decade: int                 # 1950, 1960, 1970, ...
    genre: str                      # Deutscher Schlager, Jazz, ...
    vocal_confidence: float         # 0–1

    # Runtime (aktualisiert nach jeder Phase via Rekalibrierung)
    _novelty_crit: float            # Aktueller NOVELTY_CRIT-Wert
    _echo_thresh: float             # Aktueller ECHO_THRESH-Wert
    _snapshot_generation: int       # Monoton steigend
```

### 15.3 Drei-Klassen-Kalibrierung

| Klasse | Beispiele | Ableitung |
|--------|----------|-----------|
| **Physikalische Konstanten** | −60 dBFS (digital black), −0.3 dBTP (ITU-R BS.1770) | Keine Kalibrierung nötig — Naturgesetze |
| **Material-Konstanten** | max_bandwidth_by_material, noise_floor_by_medium | Aus material_type + era_decade in calibration_matrix.py |
| **Song-Adaptive Parameter** | NOVELTY_CRIT, min_strength, ECHO_THRESH, gain_budget | **Kontinuierlich aus CalibrationContext** — NIE hartcodiert |

### 15.4 Kontinuierliche Ableitungs-Funktionen

Jede Ableitung folgt dem Muster:

```python
def compute_X(context: CalibrationContext) -> float:
    """Leitet X kontinuierlich aus dem CalibrationContext ab."""
    return continuous_function(context.restorability_score,
                                context.transfer_chain_depth,
                                context.snr_db, ...)
```

Implementierte Ableitungen:

- `compute_novelty_crit_threshold(ctx)` → §v10.41
- `compute_min_phase_strength(ctx)` → joint_calibrator
- `compute_echo_threshold(ctx)` → signal_flow_tracer
- `compute_max_retries(ctx)` → one_take_export

### 15.5 Laufzeit-Rekalibrierung (§v10.42)

Der bestehende `_mid_pipeline_calibration_step()` (33% und 66% Pipeline-Fortschritt) nutzt PMGG-gemessene Musical-Goal-Scores — **ohne zusätzliche DSP-Kosten**. Natuerlichkeit dient als Proxy für die effektive Restorability: je sauberer das Audio, desto höher die effektive Restorability, desto niedriger die NOVELTY_CRIT-Toleranz.

```python
# §v10.42 in _mid_pipeline_calibration_step():
_nat = current_scores.get("natuerlichkeit")
_eff_rs = clamp(orig_rs + max(0, nat − 0.70) × 80, orig_rs, 100)
_new_nov = 0.20 + (1 − eff_rs/100) × 0.40 + max(0, depth−1) × 0.03
set_novelty_crit_threshold(_new_nov)  # Monotonie via min(current, previous)
```

Die Monotonie-Garantie (§G86) ist in `set_novelty_crit_threshold()` implementiert: `_NOVELTY_CRIT = min(value, _NOVELTY_CRIT)`.

### 15.6 Cross-Referenzen

| Dokument | Inhalt |
|----------|--------|
| `.github/GEBOTE.md` Kategorie X | §G76–§G81: Kalibrierungs-Dispatch |
| `.github/GEBOTE.md` Kategorie XI | §G82–§G86: Laufzeit-Rekalibrierung |
| `.github/VERBOTE.md` Kategorie E | §V25–§V28: Hartcodierte Schwellwerte VERBOTEN |
| `.github/specs/09_global_calibration_matrix.md` | Detaillierte Dispatch-Implementierung |
| `.github/specs/17_sft_novelty_adaptive_calibration.md` | SFT-Adaptivität und Defekt-Audibilität |


---

## 16. Frontend↔Backend-Harmonisierung & Evaluation (§v10.990–§v10.999, 2026-08-13)

Der Kreis des Gesamtsystems schließt sich: **Vorher → Während → Nachher → Beleg**.

| §Ref | Bedeutung | Kern-Dateien |
|------|-----------|--------------|
| `§v10.990` | Zentrale UI-Palette, SOTA-Status-Panel im Hauptfenster, Bridge-SOTA-Zugänge (Model-Zoo, Consensus, Plan, Guards) | `Aurik10/ui/ui_constants.py`, `restoration_status_panel.py`, `backend/api/bridge.py`, `coordinated_repair.py` |
| `§v10.991` | closeEvent: kein modaler Dialog bei headless/unsichtbar (Layout-Gate-Hang behoben) | `modern_window.py` |
| `§v10.992` | Laienverständliche Einwilligungs-Ansicht („Gefunden … · Aurik wird …") | `bridge.get_repair_plan_consent`, `restoration_status_panel.py` |
| `§v10.993` | Crash-Report-Sichtbarkeit: Exception-Hook im GUI-Start, Fehler-Dialog beim Start, GUI-Smoke-Gate in CI | `crash_reporter.py`, `main.py`, `ci-lite.yml` |
| `§v10.994` | Model-Zoo-Aktivierung: SGMSE+ kontextaktiv, MP-SENet Norm-Kalibrierung (Peak-99 + Gain + Loudness-Guard) | `coordinated_repair.py`, `model_zoo_registry.py` |
| `§v10.995` | DAS EINE Evaluations-System: Engine, Report-Schema, CLI, Gates (Regression/Competitive/Goals), Hörvergleich-Export | `backend/core/evaluation_system.py`, `scripts/evaluate.py` |
| `§v10.996` | Konsolidierter Restaurierungs-Bericht (Plan → Ausführung → Beweis) im Ergebnis-Dialog | `bridge.get_restoration_bericht`, `results_summary.py` |
| `§v10.997` | Live-Plan-Tracker (✓/▶/○) während der Verarbeitung | `restoration_status_panel.py` |
| `§v10.998` | Kassetten-Katastrophe behoben: Null-Schwere-Fehlalarme triggern keine Phasen, Strength-Durchreichung, Energy-Collapse-Guard, Bass/Hum-Diskriminierung (Perzentil-Modulationstiefe), Block-LMS | `coordinated_repair.py`, `phase_02_hum_removal.py`, `phase_57_print_through_reduction.py` |
| `§v10.999` | Live-Mithören: voller Zwischenstand, Toggle statt Neustart, Positionsgedächtnis, Scrubber-Seek | `modern_window.py` |

### 16.1 Ehrlichkeits-Regeln (Evaluation)

1. JEDES Ergebnis wird berichtet — auch Verschlechterungen (nie gefiltert).
2. Ein Report-Schema (JSON v1.0) für objektive Läufe, Wettbewerber-Vergleiche und Hörergebnisse.
3. Gates: regression (nie schlechter), competitive (≥ RX-11-Proxy in ≥7/10), goal_achievement (Passrate ≥ 0.8).
4. Erste echte Korpus-Messung (2026-08-13): FAIL (−52 dB) → Ursache isoliert (Phase 02 auf Null-Schwere-Fehlalarm) → Fix → PASS (0.00 dB, 0 Schritte).

### 16.2 Guard-Netz (Primum non nocere, §0)

- TruePeak / Pumping / Formant / Spektral (ArtifactGuard)
- **Energy-Collapse-Guard (§v10.998)**: RMS-Ausgang < 25% des Eingangs → vollständiger Revert
- UTMOS-Loop + Do-No-Harm-Gate (UV3)
- Planner-Severity-Gate: severity < 0.05 → keine Phase

### 16.3 Cross-Referenzen

| Dokument | Inhalt |
|----------|--------|
| `.github/specs/v10.99x_stimmiges_gesamtsystem.md` | Detaillierte Spezifikation + Evidenzblock |
| `.github/workflows/ci-lite.yml` | gui-smoke-gate + evaluation-gate |
| `tests/unit/test_frontend_backend_harmony.py` | Drift-Gates Frontend↔Backend |
| `tests/unit/test_evaluation_system.py` | Evaluations-Selbsttests (Ehrlichkeit gepinnt) |
| `tests/unit/test_model_zoo_activation.py` | Zoo-Aktivierung + §v10.998-Regressionen |
| `tests/unit/test_phase02_bass_hum_discrimination.py` | Physik der Bass/Hum-Trennung |
| `scripts/diagnose_phase_degradation.py` | Schrittweise SNR-Diagnose |
