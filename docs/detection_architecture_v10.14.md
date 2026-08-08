# Aurik 10.14 „Durchblick" — Erkennungsarchitektur & Fixes

## Übersicht: Die drei Root-Cause-Fixes

| # | Komponente | Root Cause | Fix | Spec-Ref |
|---|-----------|------------|-----|----------|
| 1 | **MediumDetector** | `unknown` gewinnt jede ambiguitive Eingabe mit 0.999 Posterior durch extrem breite σ-Werte | Bayesian Prior `P(unknown)=0.02` eingeführt (Cromwell's Rule) | §6.7.4 |
| 2 | **EraClassifier** | CLAP-Plausibilitätsprüfung rejects korrekte Ära-Vorhersage für digitalisierte historische Aufnahmen (Stereo/HF vom A/D-Wandler, nicht vom Original) | Digitization Gate: Stereo/HF/analog_era Violations werden deaktiviert sobald **ein** Digital-Material in der Transferkette ist | §2.47 |
| 3 | **DefectScanner** | Einzelmaterial-Stimme mit zu geringem Gewicht (0.20) — Defekt→Material-Affinitäten nicht genutzt | Per-Defekt-Material-Severity-Aggregation + Multi-Material-Stimmen im Konsens (§v10.304.14) | §v10.14 |

---

## 1. MediumDetector: Bayesian Prior Dämpfung

### Problem
Der `_bayesian_score()` berechnete Log-Likelihoods über 16 Material-Gaußverteilungen, normalisierte per Softmax — **ohne expliziten Prior**. `unknown` hatte extrem breite σ-Werte (σ=8000 Hz für Bandbreite, σ=15 dB für SNR), was bedeutete:

- Jede ambiguitive Eingabe passte besser zu `unknown` (riesiger Wertebereich) als zu spezifischen Materialien (enge σ-Werte)
- `unknown` → 0.999, alle anderen → ≈0.0
- Selbst bei starken physikalischen Indikatoren (Vinyl-Rotation, Rillenrauschen) blieb `unknown` dominant

### Lösung
Expliziter Bayesian Prior nach **Cromwell's Rule** (J. Pearl, 1988, Probabilistic Reasoning in Intelligent Systems, §2.3):

```
P(unknown) = 0.02
P(spezifisches Material) = 0.98 / 15 ≈ 0.0653
```

Implementiert als **log-prior** vor der Softmax-Normalisierung:

```python
log_prior = math.log(0.02) if mat == "unknown" else math.log(0.98 / 15)
log_likes[mat] += log_prior
```

**Effekt**: `unknown` braucht ≈1.5 nats mehr Evidenz als spezifische Materialien, um zu gewinnen. Echte Hypothesen werden bevorzugt, ohne false-positives zu erzwingen.

### Datei
- `forensics/medium_detector.py` → `_bayesian_score()` (Zeile ~2047-2063)

---

## 2. EraClassifier: Digitization Gate (§2.47)

### Problem
Die CLAP-Plausibilitätsprüfung verwarf Tier-1 CLAP-Ergebnisse wenn:

1. **Stereo-Violation**: `is_stereo ∧ clap_decade < 1960 ∧ stereo_width > 0.05`
2. **HF-Violation**: `highband_presence > 0.20 ∧ clap_decade < 1940`
3. **Analog-Era-Violation**: `clap_decade > 1989 ∧ analoge Materialien in der Chain`

Diese Checks sind für **rein analoge** Ketten korrekt — eine 1928er Aufnahme KANN nicht in Stereo sein.

**ABER**: Eine 1928er Schellack-Aufnahme die auf CD digitalisiert wurde, hat:
- Legitimes Stereo (Left/Right-Duplizierung vom A/D-Wandler)
- Volle Bandbreite bis 22 kHz
- CLAP erkennt 1928 korrekt — der Violation-Gate rejected dies fälschlich

### Lösung
**Digitization Gate**: Wenn **irgendein** digitales Material in der Transferkette ist, werden Stereo-/HF-/Analog-Era-Violations deaktiviert:

```python
_DIGITISED_MATERIALS = frozenset({
    "cd", "cd_digital", "mp3_low", "mp3_high", "aac", "streaming",
    "dat", "minidisc", "dcc", "bluray_audio", "dvd_audio", "sacd",
    "pcm_digital", "lossless_digital",
})
_is_digitized = any(material in transfer_chain)

_stereo_violation = is_stereo ∧ decade < 1960 ∧ width > 0.05 ∧ NOT _is_digitized
_hf_violation = highband > 0.20 ∧ decade < 1940 ∧ NOT _is_digitized
_analog_era_violation = NOT _is_digitized ∧ decade > 1989 ∧ analog_chain
```

**Logik**: Nur bei **rein analoger** Kette (kein Digital-Material) sind Stereo/HF physikalische Unmöglichkeiten.

### Datei
- `backend/core/era_classifier.py` → `classify()` (Zeile ~1387-1422)

---

## 3. DefectScanner: Material-Score in Konsens

### Problem
Der `resolve_material_consensus()` erhielt vom DefectScanner nur **eine** Material-Stimme:

```python
defect_result={"material": "vinyl", "score": 5.39}
```

Mit nur 0.20 Gewicht und einem einzigen Material wurde der DefectScanner im Konsens marginalisiert — selbst wenn er eindeutige Defektsignaturen für Vinyl (Crackle, Groove-Echo, Rillenrauschen) UND Cassette (Tape-Hiss, Wow/Flutter) gleichzeitig fand.

Kein Mechanismus, um die **pro-Defekt pro-Material** Affinitäten in den Konsens zu transportieren.

### Lösung

#### A) Material-Konsens erweitert (§v10.14)

`material_consensus.py` akzeptiert nun `defect_result["material_scores"]` — ein Dict mit **pro-Material aggregierter Severity**:

```python
defect_result={
    "material": "vinyl",
    "score": 5.39,
    "material_scores": {    # §v10.14 NEU
        "vinyl": 1.25,      # crackle(0.45) + riaa_error(0.35) + rumble(0.45)
        "cassette": 0.62,   # wow(0.18) + flutter(0.22) + tape_hiss(0.22)
        "reel_tape": 0.35,  # print_through(0.35)
    }
}
```

Jedes Material erhält eine Stimme proportional zu seiner normalisierten Severity. 60% des DefectScanner-Gewichts (0.20 × 0.6 = 0.12) werden über die Affinitäts-Stimmen verteilt.

#### B) Defekt-Severity-Aggregation (§v10.304.14)

In `pre_analysis.py` wird parallel zu `_defect_inferred_carriers` jetzt `_defect_carrier_scores` berechnet, die **alle** Defektscores pro Carrier aggregiert (auch unterhalb der Carrier-Inferenz-Schwelle):

```python
for defect_name, (carrier, threshold) in DEFECT_CARRIER_MAP.items():
    if score_key == defect_name:
        defect_carrier_scores[carrier] += severity  # immer aggregieren
        if severity >= threshold:  # nur für Inferenz
            defect_inferred_carriers.append(carrier)
```

### Dateien
- `backend/core/material_consensus.py` → `resolve_material_consensus()` (Zeile ~67-80)
- `backend/core/pre_analysis.py` → Deep-Transfer-Chain-Injection (Zeile ~644, ~681-683, ~834-835)

---

## Zusammenfassung der Erkennungsarchitektur

```
                         ┌──────────────────┐
     Audio Signal ──────►│  MediumDetector  │────► transfer_chain[]
                         │  (§6.7 v10.0.0) │      bayesian_scores{}
                         │  Prior: 2% unk.  │      confidence
                         └────────┬─────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
          ┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
          │ EraClassifier│ │DefectScanner│ │ GenreClassif.│
          │ (§2.14)      │ │(§2.46a)    │ │ + sem.Goals  │
          │ CLAP+DSP     │ │ 14 Defects │ │              │
          │ Digit.Gate ✓ │ │ Aff.Scores✓ │ │              │
          └───────┬──────┘ └─────┬──────┘ └──────┬───────┘
                  │               │               │
                  └───────────────┼───────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   Material-Konsens         │
                    │   (§v10.20)                │
                    │   MD:0.50 ERA:0.30 DEF:0.20│
                    │   +DefectAffinities(×0.12) │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   Deep-Transfer-Chain      │
                    │   (§2.46a)                 │
                    │   Chain-Injection + Sort   │
                    │   + Vinyl-Inference        │
                    │   + Depth-Cap              │
                    └────────────────────────────┘
```

### Gewichte

| Quelle | Gewicht | Methode |
|--------|---------|---------|
| MediumDetector | 0.50 | Physikalische Signalanalyse (autoritativ) |
| EraClassifier | 0.30 | Ära → Material-Inferenz (korrelativ) |
| DefectScanner | 0.20 | Defektmuster → Material (indirekt) |
| DefectAffinities | 0.12 | Per-Defekt Material-Severity (60% von 0.20) |

---

## Änderungs-Historie

| Version | Datum | Änderung |
|---------|-------|----------|
| v10.14.0 | 2026-08-06 | §6.7.4: Bayesian Prior für unknown (Cromwell's Rule) |
| v10.14.0 | 2026-08-06 | §2.47: Digitization Gate für EraClassifier CLAP-Plausibilität |
| v10.14.0 | 2026-08-06 | §v10.14: Defect-Material-Affinitäten im Konsens |
| v10.14.0 | 2026-08-06 | §fix: `_era_decade`, `_era_confidence`, `_defect_score` in pre_analysis definiert |
