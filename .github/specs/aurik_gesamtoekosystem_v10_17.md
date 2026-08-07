# §v10.17 Aurik Gesamtökosystem — Spezifikation

## Präambel

> Aurik ist ein selbstkalibrierendes Audio-Restaurationssystem. Jeder Eingriff  
> folgt dem Zyklus: **Messen → Kalibrieren → Anwenden → Verifizieren → Entscheiden**.  
> Kein Parameter ist hartcodiert, wenn er aus dem Signal ableitbar ist.  
> Die binäre Suche (12 Iterationen, ±0.025% Präzision) ist der Standard für  
> alle kontinuierlichen Parameter.

---

## 1. Architektur-Datenfluss

```
IMPORT → PRE → 64 PHASEN (PMGG) → POST-PROCESSING (PostGate) → EXPORT
         │         │                      │                      │
    STCG 20ms  Binäre Suche         Verify/Rollback       OneTakeExport
    Polarity   16 Goals (inkl.      Strength-Parameter    TruePeak/LUFS/
               Fatigue + Anchor)    Mode-Aware            Fatigue/Stereo
               Mode-Gewichte                              Auto-Fix-Loop
```

---

## 2. Zone A: Pre-Pipeline

| Komponente | Methode | Parameter |
|---|---|---|
| STCG | Universeller 20ms-Guard | `_GLOBAL_MAX_MS = 20.0` |
| Polarity Inversion | L/R-Korrelations-Check | `corr < -0.30` → invertieren |

---

## 3. Zone B: Main Pipeline (PMGG, 64 Phasen)

### 3.1 Kalibrierungs-Algorithmus

**Binäre Intervallhalbierung** (ersetzt lineare Stärke-Reduktion):

```
1. Start: strength = initial_strength (material-kalibriert, typ. 1.0)
2. Wenn keine Regression → FERTIG (volle Stärke ist optimal)
3. Sonst: lo=0.0, hi=initial_strength
4. Für i in 1..12:
     strength = (lo + hi) / 2
     Phase mit strength ausführen
     Regression? → hi = strength  (zu stark)
     Keine?      → lo = strength  (könnte stärker sein)
     Wenn hi - lo < 0.005 → BREAK
5. Optimal = lo (höchste Stärke ohne Regression)
```

| Parameter | Wert |
|---|---|
| Iterationen | 12 |
| Grundauflösung | 1/4096 = ±0.025% |
| Effektive Präzision | ±0.005% (mit Interpolation) |
| Abbruchkriterium | Intervall < 0.5% ODER 12 Iterationen |

### 3.2 Goal-Set (16 Ziele)

```
Brillanz, Wärme, Groove, TonalCenter, Natürlichkeit, Timbre-Authentizität,
Transient-Energie, Bass-Kraft, Authentizität, Emotionalität, Transparenz,
SpatialDepth, Mikro-Dynamik, Separation-Fidelity, Artikulation,
Listening-Fatigue  ← v10.15
```

### 3.3 Mode-Gewichte

| Goal | Restoration | Studio 2026 |
|---|---|---|
| Natürlichkeit | ×1.25 | ×1.00 |
| Authentizität | ×1.20 | ×1.00 |
| TonalCenter | ×1.15 | ×1.00 |
| Timbre-Auth. | ×1.15 | ×1.00 |
| Brillanz | ×0.85 | ×1.20 |
| Bass-Kraft | ×0.90 | ×1.10 |
| SpatialDepth | ×0.90 | ×1.15 |
| Listening-Fatigue | ×1.30 | ×0.90 |

### 3.4 Regression-Threshold

| Restorability | Threshold |
|---|---|
| ≥ 80 (exzellent) | 0.012 |
| 50–79 (gut) | 0.025 |
| < 50 (beschädigt) | 0.040–0.060 |

---

## 4. Zone C: Post-Processing (PostGate)

### 4.1 PostGate-Regeln

1. 5 s Stichprobe VORHER messen (5 Ziele: brillanz, warmth, natreblichkeit, transparenz, spatial_depth)
2. Komponente mit `strength` ausführen
3. 5 Ziele NACHHER messen
4. Δ < −0.015 → Komponente überspringen (Rollback)
5. `strength`-Parameter via Binäre Suche optimierbar

### 4.2 Gewrappte Komponenten

| Komponente | Typ | PostGate-Goals |
|---|---|---|
| AntiMufflingPass | Spektral | Brillanz+Wärme |
| VocalClarityMax | Enhancement | Artikulation+Brillanz |
| PerceptualExportOptimizer | Enhancement | Alle 5 |
| HarmonicPreservationGuard | Guard | Wärme+Timbre |
| HarmonicLatticeAnalyzer | Guard | TonalCenter |
| HumanizationPass | Enhancement | Adaptiv (Material-Kalibrierung) |
| ListeningEQ | EQ | Adaptiv (Spektral-Δ) |

**Nicht gewrappt** (detektionsbasiert, idempotent):
VocalScratchRepair, TapeHeadArtifactRepair, SmartTapeRepair,
ArtifactEchoRemoval, SibilanceMaxRepair, SpecializedDefectRepair,
DirectDefectRepair

---

## 5. Zone D: Export (OneTakeExport)

### 5.1 Quality Gates

| Gate | Schwelle | Aktion bei Fail |
|---|---|---|
| True Peak | ≤ 0.0 dBTP | Brickwall-Limiter −0.3 dBTP |
| LUFS | −16 ± 2 (Restoration) / −12 ± 2 (Studio) | Gain-Korrektur ±6 dB |
| Fatigue | < 0.4 | −1 dB Shelf > 4 kHz |
| Stereo-Korrelation | > −0.3 | Warnung (kein Auto-Fix) |

### 5.2 Auto-Fix-Loop

```
1. ExportQualityGate.check()
2. PASS → Export
3. FAIL/WARN → Auto-Korrektur → Re-Check (max. 3 Retries)
4. Nach 3 Retries immer noch FAIL → Export mit Warnung
```

---

## 6. PerceptualReferenceValidator (v10.17)

### 6.1 Perceptual Similarity Score (PSS)

Misst die perzeptuelle Distanz zum ORIGINAL (Pre-Pipeline-Anker).

| Dimension | Gewicht | Methode |
|---|---|---|
| Spektrale Fidelity | 40% | Pearson-r der Bark-Hüllkurve (25 Bänder) |
| Transienten-Erhalt | 25% | Jaccard-Ähnlichkeit der Onset-Positionen |
| Stereo-Kohärenz | 20% | L/R-Korrelations-Delta |
| Energie-Erhalt | 15% | RMS-Ratio |

**Gate:** PSS ≥ 0.85 → akzeptiert. PSS < 0.85 → Phase verwerfen.

---

## 7. Präzisions-Hierarchie

| Ebene | Methode | Präzision |
|---|---|---|
| Parameter-Kalibrierung | Binäre Suche 12 Iter. | ±0.025% |
| Material-Kalibrierung | DSP-Formel (RMS/FFT) | ±2% des Bereichs |
| Quality-Gate-Messung | RMS/FFT/Peak | ±0.5 dB / ±1% |
| Perzeptuelle Validierung | PSS (4 Dim.) | ±0.01 PSS |

---

## 8. Ausnahmen (keine Kalibrierung)

| Komponente | Grund |
|---|---|
| Detektions-basierte Reparaturen | Binär (reparieren oder nicht) |
| Deterministische Transformationen | Mathematisch exakt |
| ML-Inferenz (FlashSR, Demucs, etc.) | Modell-Output nicht kontinuierlich |
| PhaseOutputGuard, DC-Offset, Resampling | Safety/Exakt |

---

## 9. Datei-Index

| Datei | Zweck | Version |
|---|---|---|
| `stereo_temporal_coherence_guard.py` | STCG 20ms Universal Guard | v10.14 |
| `per_phase_musical_goals_gate.py` | PMGG + Binäre Suche + Fatigue | v10.16 |
| `song_goal_importance.py` | Mode-Gewichte | v10.15 |
| `post_processing_gate.py` | PostGate + Singleton | v10.15 |
| `humanization_pass.py` → `klang_guards.py` | Adaptive Stärke | v10.15 |
| `listening_mode_eq.py` | Adaptiver Listening-EQ | v10.15 |
| `listening_fatigue_metric.py` | Fatigue-Metrik | v10.15 |
| `export_quality_gate.py` | Export-Qualitätsmessung | v10.15 |
| `one_take_export.py` | Auto-Fix-Export-Loop | v10.15 |
| `perceptual_reference_validator.py` | PSS gegen Original | v10.17 |
| `phase_interface.py` | PhaseResult + audio_before_snippet | v10.15 |
| `unified_restorer_v3.py` | Haupt-Pipeline-Integration | v10.17 |

---

## 10. Widerspruchsfreiheit

- Restoration Mode = `is_studio_2026=False`: Natürlichkeit first, Enhancement zurückgenommen
- Studio 2026 = `is_studio_2026=True`: Brillanz first, moderne Klangästhetik
- Beide Modi nutzen dieselbe binäre Suche, dieselben Quality Gates, denselben PSS
- PSS-Gate (0.85) ist LOCKERER als PostGate (Δ<0.015) — PSS fängt nur katastrophale Abweichungen
- STCG 20ms-Guard gilt universell — kein Widerspruch zwischen Pre/Post/Intra-Phase
- Fatigue-Metrik invers zu PMGG-Goals (0=schlecht→1=optimal via `fatigue_as_pmgg_goal`)

## 11. Stereo-Lag-Prävention (garantiert)

### 11.1 Prinzip

> Keine Phase darf einen L/R-Kanal-Versatz einführen. Das Importfile hat  
> keinen Lag — also darf Aurik keinen erzeugen.

### 11.2 Mehrschichtige Prävention

| Schicht | Mechanismus | Garantie |
|---|---|---|
| **A) STCG Universal Guard** | Alle 10 Aufrufpfade: Korrektur > 20ms wird blockiert | Keine False-Positive-Korrektur |
| **B) Per-Phase Lag-Safety** | Phase 12, 24, 29, 31: `_enforce_stereo_lag_safety()` | Lag wird NACH der Phase gemessen und korrigiert |
| **C) Chunk-Verarbeitung** | AdaptiveChunkProcessor §GEBOT-G42: Stereo-Lag-Integrität pro Chunk | L/R-Drifts innerhalb eines Chunks erkannt |
| **D) Phase 12 M/S-Fix** | Side-Kanal = selber Algorithmus wie Mid-Kanal | PSOLA erzeugt keinen differentiellen L/R-Versatz |
| **E) PMGG Fatigue-Goal** | L/R-Korrelations-Delta in Listening-Fatigue-Metrik | Phasen mit Stereo-Kollaps werden erkannt |
| **F) PSS Stereo-Kohärenz** | PerceptualReferenceValidator misst L/R-Korrelations-Delta | Phasen mit Stereo-Verlust werden verworfen |

### 11.3 STCG-Aufrufpfade (vollständig)

| Aufrufer | Datei | Zeile | Guard |
|---|---|---|---|
| pre_pipeline | unified_restorer_v3.py | 10924 | 20ms ✅ |
| post_pipeline | unified_restorer_v3.py | 12473 | 20ms ✅ |
| post_export | unified_restorer_v3.py | 13100 | 20ms ✅ |
| post_phase | unified_restorer_v3.py | 9101 | 20ms ✅ |
| phase_12_pre_chunking | phase_12_wow_flutter_fix.py | 466 | 20ms ✅ |
| phase_12_wow_flutter_fix | phase_12_wow_flutter_fix.py | 1517 | 20ms ✅ |
| phase_24 | phase_24_dropout_repair.py | 1415 | 20ms ✅ |
| phase_31 | phase_31_speed_pitch_correction.py | 799 | 20ms ✅ |
| import_pipeline | file_import.py | 618 | 20ms ✅ |
| stereo_drift_final | stereo_drift_state.py | 114 | 20ms ✅ |

### 11.4 Per-Phase Lag-Safety (Phasen mit unabhängiger L/R-Verarbeitung)

| Phase | Mechanismus | Messmethode |
|---|---|---|
| Phase 12 (Wow/Flutter) | M/S-Gleichlauf + STCG pre/post chunking | STCG |
| Phase 24 (Dropout) | `_enforce_stereo_lag_safety(audio, repaired, sr)` | GCC-PHAT, ±960 Samples |
| Phase 29 (Hiss) | `_enforce_stereo_lag_safety(audio, processed, sr)` | GCC-PHAT, ±960 Samples |
| Phase 31 (Speed/Pitch) | STCG am Ende der Verarbeitung | STCG |

### 11.5 Garantie

```
Input: L/R-Delay = 0 ± 0.5 Samples (kein Lag)
  → Pre-Pipeline STCG: misst, blockiert > 20ms
  → Phase 01–64: per-phase safety (wo nötig)
  → Post-Pipeline STCG: misst, blockiert > 20ms
  → ExportQualityGate: Stereo-Korrelation > −0.3
Output: L/R-Delay = 0 ± 1.5 Samples (kein hörbarer Lag)

Maximale L/R-Abweichung nach vollständiger Pipeline: < 1.5 Samples = 0.03 ms.
Dies liegt unter der menschlichen Wahrnehmungsschwelle für ITD (10–20 µs).
```

### 11.6 Test-Abdeckung

| Test | Datei | Was |
|---|---|---|
| Universal Guard (alle 10 phase_ids) | test_stereo_temporal_coherence_guard.py | Korrektur > 20ms blockiert |
| Zero-Lag-Through (alle 10 phase_ids) | test_null_lag_pass_through.py | Lag-freies Stereo unverändert |
| Pipeline-Call-Chain-Simulation | test_null_lag_pass_through.py | 9 STCG-Aufrufe in Serie |
| Post-Pipeline-Retry-Loop | test_null_lag_pass_through.py | G14 Retry-Loop stabil |
