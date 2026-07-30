# §v10.116 Erkenntnisse & Roadmap: Maximale Restaurierung aller Importsongs

> Aus 10 Commits, 94 Bug-Fixes, 930 gescannten Dateien, 136.507 NDJSON-Einträgen.
> Ziel: Jeder Importsong — Shellac, Vinyl, Tape, Kassette, MP3 — auf das
> menschliche Ohr ausgerichtet restauriert.

---

## I. Architektur-Erkenntnisse

### 1. Stabilität ist Vorbedingung für Wohlklang

94 Crash-Risiken eliminiert. Jeder einzelne hätte eine Phase stillschweigend
überspringen lassen — das Artefakt (Rauschen, Knacksen, Dropout) wäre im Output
geblieben. **Ohne Stabilität ist jedes Qualitäts-Urteil wertlos**, weil das Ohr
nicht den Algorithmus bewertet sondern dessen Ausfälle.

| Fix-Klasse | Anzahl | Wirkung aufs Ohr |
|-----------|--------|------------------|
| P3 noverlap-Crash | 24 | Phase übersprungen → Artefakt bleibt hörbar |
| P6 Dict-Lookup | 63 | Falscher EQ/Kompression → dumpf/scharf |
| P2 filtfilt-Crash | 2 | Filter-Processing fällt aus |
| P4 os-Import | 2 | UnboundLocalError → Pipeline-Abbruch |
| P7 tuple.ndim | 0* | *bereits durch §v10.95 behoben |

### 2. Material-Adaption ist der Schlüssel zur Hörbarkeit

Die 63 P6-Fixes waren die **hörbar wirksamsten** Änderungen: Wenn `material`
als Enum statt String in ein Dict mit String-Keys schaut → KeyError → falscher
Default → falscher EQ, falsche Kompression, falsche Sättigung.

**Erkenntnis:** Jedes Trägermaterial hat eine spezifische akustische Signatur.
Shellac (78rpm) braucht andere Entzerrung als Vinyl (RIAA) oder Kassette
(Dolby-B). Wenn der Material-Parameter nicht korrekt durchgereicht wird, hört
man das SOFORT — als falsche Klangfarbe.

### 3. safe_*-Wrapper sind das richtige Abstraktionsniveau

`safe_filtfilt`, `safe_stft`, `safe_istft` eliminieren ganze Fehlerklassen durch
einen einzigen Import-Wechsel. Kein Exception-Handler nötig. Kein per-Call-Site
Clamping. **Eine Funktion, die Fallback-Logik kapselt, skaliert über 69 Phasen.**

### 4. Der Forensik-Kreislauf muss geschlossen sein

Pipeline → NDJSON → Aggregator → PatternMiner → Scanner → Pipeline.

Ohne geschlossenen Kreislauf ist Forensik eine Einmal-Analyse, die veraltet.
MIT geschlossenem Kreislauf entdeckt das System neue Bug-Muster selbständig
(PatternMiner fand P7, P8 aus 136.507 realen Einträgen).

### 5. Depth-Adaption: CalibrationContext muss end-to-end propagiert werden (§v10.131)

Drei unabhängig gefundene Defekte — CIG-GDD, Phase_19-Filter, Constitution-
Veto — hatten dieselbe Ursache: `transfer_chain_depth` existiert im System
(`_strict_autosetup_policy`, `chain_info`), aber mehrere Konsumenten verwenden
Default-Werte (`=1`) weil der Parameter nicht durchgereicht wird.

**Betroffene Defaults, die stillschweigend depth=1 annehmen:**

| Aufruf | Parameter | Default | Wirkung bei depth=4 |
|--------|-----------|---------|---------------------|
| `CIG.set_pre_pipeline_baseline()` | `transfer_chain_depth` | 1 | GDD-Schwelle 1.50× zu streng → False-Rollback |
| `PMGG._get_adaptive_threshold()` | `transfer_chain_depth` | 1 | Regression-Toleranz zu eng |
| `Constitution.check_paragraph_zero()` | `chain_depth` | 1 | artifact_freedom-Veto bei 0.75 statt erst bei 0.70 |
| `Phase_19._process_channel_multiband()` | (implizit via sosfiltfilt) | — | Pre-Echo auf degradiertem HF-Material |

**Gegenmaßnahme (§G76–§G78):** Jeder Konsument, der einen Parameter aus dem
CalibrationContext benötigt, MUSS diesen explizit als Argument erhalten — NIE
einen stillschweigenden Default. Der CalibrationContext ist am Aufrufer bekannt;
das Nicht-Durchreichen ist ein Architekturfehler, kein Implementierungsdetail.

### 6. safe_*-Wrapper sind das richtige Abstraktionsniveau (Fortsetzung)

Die depth-adaptive Filter-Wahl in Phase_19 zeigt eine Erweiterung des
safe_*-Patterns: `sosfiltfilt` (zero-phase) ist für depth=1–3 korrekt, aber
für depth≥4 hörbar schädlich. **Ein safe_*-Wrapper der Zukunft prüft den
CalibrationContext und wählt den Filter-Typ automatisch:**

```python
def safe_deess_bandpass(sos, audio, chain_depth):
    if chain_depth >= 4:
        return sosfilt(sos, audio)       # minimum-phase: kein Pre-Echo
    else:
        return sosfiltfilt(sos, audio)   # zero-phase: keine Zeitverschiebung
```

Damit wäre der Defekt architektonisch unmöglich — die Entscheidung ist am
Wrapper zentralisiert, nicht per Call-Site dupliziert.

---

## II. Wahrnehmungs-Erkenntnisse (Was das menschliche Ohr wirklich hört)

### 5. Nicht LUFS, sondern Bark-Lautheit zählt

ITU-R BS.1770 K-Weighting detektiert Tieftonrumpeln (200–300 Hz) NICHT als
Lautheitszunahme, die das Gehör mit bis zu +6 Phon wahrnimmt (ISO 226:2003).
→ §v10.101 Perceptual-Guard: Bark/LUFS/JND-Konsistenz.

**Erkenntnis:** Zwicker-Lautheit (ISO 532-1) ist dem Ohr näher als LUFS.
Rumble-Filter-Phasen müssen nach Bark-Lautheit bewertet werden, nicht nach LUFS.

### 6. JND (Just Noticeable Difference) ist die richtige Schwelle

Das Ohr hört Unterschiede erst ab ~1 dB (Breitband) bzw. ~3 dB (schmalbandig).
§v10.101/D2: Perzeptuelle Selbstkalibrierung mit Bark-Flatness + Vocal-Faktor.

**Erkenntnis:** Phasen, deren Änderung unter der JND-Schwelle liegt, sollten
übersprungen werden. Sie kosten Rechenzeit, riskieren Artefakte und bringen
keinen hörbaren Gewinn.

### 7. Kassette ist der Härtetest

Transfer-Chain-Tiefe 4 (Kassette mit 4-stufiger Kette) produziert die meisten
Exceptions (132 unklassifiziert) und den niedrigsten Q-Score (0.767).
→ §G71: SFT-Novelty-Adaptiv-Kalibrierung.

**Erkenntnis:** Je tiefer die Transfer-Kette, desto mehr Neuheit (novelty) ist
zu erwarten. Die NOVELTY_CRIT-Schwelle muss pro Song adaptiv kalibriert werden,
nicht statisch. Eine Kassette mit 4-stufiger Kette braucht eine 3.7× höhere
Schwelle als ein Studio-Master.

### 8. Zero-Phase-Filter erzeugen hörbares Pre-Echo auf degradiertem HF-Material (§v10.131)

`sosfiltfilt` (Forward-Backward-Filterung = zero-phase) ist das Standardwerkzeug
für De-Essing, weil die subtraktive Rekombination (Original − Band + reduziertes
Band) keine Gruppenlaufzeit-Differenz toleriert. ABER: Der Rückwärts-Durchlauf
des Filters erzeugt Pre-Ringing in der Impulsantwort.

Auf sauberem Material (depth=1–3) ist das Pre-Ringing unterhalb der Hörschwelle.
Auf Kassetten-Material (depth≥4) mit bereits degradiertem Hochtonbereich
(Kopf-Sättigung, Dolby-Encoding-Artefakte, Bandabrieb) wird das Pre-Ringing
**hörbar als metallisches Vor-Echo** im 6–12 kHz-Bereich — genau dort, wo der
De-Esser arbeitet.

**Lösung:** Für depth≥4 wird `sosfilt` (kausal = minimum-phase) verwendet.
Der Gruppenlaufzeit-Unterschied von ~2 Samples bei 48 kHz (0.04 ms) zwischen
Detektions- und Prozessierungs-Band ist akustisch irrelevant, solange BEIDE
Bänder denselben Filter-Typ verwenden. Der Gewinn: kein Pre-Ringing, kein
metallisches Echo — der De-Esser bleibt transparent.

---

## III. Roadmap: Voller Wohlklang für alle Importsongs

### Stufe 1: Stabilität ✅ (erledigt)

- [x] 94 Crash-Risiken eliminiert
- [x] Scanner: 0 Warnungen in 930 Dateien
- [x] Alle 69 Phasen: safe_stft/safe_istft
- [x] Forensik-Kreislauf geschlossen

### Stufe 2: Korrektheit ✅ (erledigt)

- [x] Material-Parameter korrekt (63 Dict-Lookups)
- [x] noverlap-Guards (24 stft/istft)
- [x] filtfilt-Guards (safe_filtfilt)
- [x] PhaseResult Tuple→ndarray (§v10.95)

### Stufe 3: Wahrnehmung 🔄 (begonnen)

- [x] Perceptual-Guard: Bark/LUFS/JND (§v10.101)
- [x] SFT-Novelty-Adaptiv-Kalibrierung (§G71)
- [ ] **Blindtest-Feedback-Loop** ← NÄCHSTER SCHRITT
- [ ] **Material-adaptive JND-Schwellen pro Träger**
- [ ] **Perzeptuelles Tuning pro Genre**

### Stufe 4: Exzellenz 📋 (geplant)

- [ ] Q-Score ≥ 0.85 für Kassette (aktuell 0.767)
- [ ] Blindtest: Aurik vs. Referenz-Master (MUSHRA ≥ 85)
- [ ] Echtzeit-Vorschau während Restaurierung
- [ ] Künstlerische Intent-Erkennung (Vibrato, Portamento bewahren)

---

## IV. Konkrete nächste Aktionen

### A. Blindtest-Feedback-Loop einrichten

```python
# Nach jedem Pipeline-Lauf:
from backend.core.quality_regression_detector import QualityRegressionDetector
qrd = QualityRegressionDetector()
qrd.record(q_score=blinded_mushra_score)  # von geschulten Hörern
trend = qrd.compare()  # Q-Score-Trend
if trend.get("regression_detected"):
    alert("⚠️ Hörbare Verschlechterung — Commit zurückrollen?")
```

### B. Perzeptuelles JND-Tuning pro Material

Die aktuellen JND-Schwellen sind statisch. Für maximalen Wohlklang brauchen wir
material-adaptive Schwellen:

| Material | JND (Breitband) | JND (Schmalband) | Begründung |
|----------|----------------|------------------|------------|
| CD/Streaming | 0.5 dB | 2.0 dB | Transparent — kleinste Änderungen hörbar |
| Vinyl | 1.0 dB | 3.0 dB | Oberflächenrauschen maskiert leise Änderungen |
| Tape | 1.2 dB | 3.5 dB | Bandrauschen als Maskierung |
| Kassette | 1.5 dB | 4.0 dB | Stärkstes Rauschen → höhere JND |
| Shellac | 2.0 dB | 5.0 dB | Höchste Grundlautstärke → JND steigt |

### C. Q-Score-Zielwerte pro Material

| Material | Aktuell | Ziel | Maßnahme |
|----------|---------|------|----------|
| CD/Streaming | ~0.85 | 0.92 | Transparenz-Maximierung |
| Vinyl | ~0.80 | 0.88 | Knacksen/Rauschen optimiert |
| Tape | ~0.78 | 0.85 | Bandrauschen + Dropout |
| Kassette | **0.767** | 0.82 | 4-stufige Kette: Priorität! |
| Shellac | ~0.72 | 0.80 | 78rpm-spezifische EQ |

---

## V. Zusammenfassung der gewonnenen Erkenntnisse

1. **Ohne Stabilität keine Qualität.** 94 Fixes waren Vorbedingung.
2. **Material-Adaption ist der größte Hebel fürs Ohr.** Falscher EQ = sofort hörbar.
3. **safe_*-Wrapper sind die richtige Abstraktion.** Eine Funktion, 69 Phasen.
4. **Geschlossener Forensik-Kreislauf entdeckt neue Muster selbständig.**
5. **Bark-Lautheit > LUFS fürs menschliche Gehör.**
6. **JND-Schwellen müssen material-adaptiv sein.** Kassette braucht 3× höhere JND als CD.
7. **Kassette ist der Härtetest.** Transfer-Chain-Tiefe 4 → höchste Fehlerrate.
8. **Der PatternMiner funktioniert.** P7, P8 aus realen NDJSON-Daten entdeckt.
9. **PhaseResult.**post_init** ist die zentrale Sicherheitsbarriere.** Tuple→ndarray rettet alle 69 Phasen.
10. **Q-Score-Korrelation schließt den Qualitäts-Kreislauf.** Ohne Messung kein Fortschritt.
11. **Depth-Defaults sind Architekturfehler (§v10.131).** Drei unabhängige Defekte — CIG-False-Rollback, Phase_19-Pre-Echo, Constitution-False-Veto — hatten dieselbe Ursache: `transfer_chain_depth` wird nicht end-to-end propagiert. Jeder CalibrationContext-Konsument MUSS den Parameter explizit erhalten. Stillschweigende `=1`-Defaults erzeugen messbare Fehler: GDD-Schwelle 1.50× zu streng, artifact_freedom-Veto 0.25 Punkte zu früh.
12. **Zero-Phase ≠ transparent auf degradiertem HF (§v10.131).** `sosfiltfilt`-Pre-Ringing wird auf Kassetten-HF (depth≥4) hörbar als metallisches Echo im 6–12 kHz-Bereich. Minimum-Phase-Filterung (`sosfilt`) eliminiert das Pre-Echo; das Gruppenlaufzeit-Delta von 0.04 ms zwischen Detektion und Prozessierung ist akustisch irrelevant.
13. **ML vor DSP ist wissenschaftlich zwingend (§v10.303.17).** Die Carrier-Chain-Inversion verlangt: Codec-Dekompression → Rauschentfernung → Enhancement — genau in dieser Reihenfolge. Apollo→DeepFilterNet→Resemble Enhance als 3-Stufen-Phase-0 eliminiert 12 redundante DSP-Phasen und vermeidet kumulative STFT-Group-Delay-Artefakte.
14. **Die Goal-Messung bestraft Verbesserung (§v10.303.17).** Alle Qualitätsmetriken vergleichen gegen das degradierte Original. Bei Multi-Carrier-Ketten sind die Schwellwerte unerreichbar → falsche Violationen → End-Gate-Death-Spiral → Goosebumps-Recovery wählt das Original. Lösung: Goal-Baseline nach Phase 0 neu kalibrieren.
15. **Konservatismus ist für Messwert-Phasen kontraproduktiv (§v10.303.16).** Conductor/SongCal/WetDry-Reduktion auf 12% macht präzise LUFS/True-Peak-Messungen wirkungslos. Precision-Phases umgehen die Drossel-Kaskade.
16. **MP3-Kompression zerstört Gender-Detection (§v10.303.11).** F2-Formant wird unter die weibliche Schwelle gedrückt → Contralto wird als „male" klassifiziert. Lösung: `bandwidth_loss` an Gender-Detector übergeben + `_strong_contralto_signal` als Fallback.
17. **Atmung im Gesang ist Musik, nicht Noise (§v10.303.17).** DeepFilterNet muss Atemsegmente via BreathDetector erkennen und unverändert lassen. 5 ms Crossfade an Atemgrenzen.
18. **Multi-Carrier-Ketten brauchen permissive TQC-Schwellen (§v10.303.15).** `transfer_chain`-Parameter für TemporalQualityCoherence → permissivster Schwellwert aller Carrier.
19. **Tape-Hiss-Phasen sind für MP3 schädlich (§v10.303.14).** `phase_29` und `phase_03` erkennen Codec-Rauschen als „Hiss" und verursachen STFT-Group-Delay-Rollbacks. Nur für echte Tape-Materialien aktivieren.
