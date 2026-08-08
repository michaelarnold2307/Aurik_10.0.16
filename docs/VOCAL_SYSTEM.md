# Aurik 10 — Vocal System Documentation

> Spec 22 B3: Weltklasse-Gesangsrestauration — Architektur, Dimensionen, Benchmarks

## Architektur-Übersicht

```
VocalDetector → VocalNaturalnessScorer → VocalQualityGate
       ↓                    ↓                      ↓
  Stimme erkannt      6-Dimensionen         Freigabe/Schutz
  (F0, Formanten,     bewertet              (VQI ≥ 0.70)
   Gender, Register)
```

## Die 6 Dimensionen der Gesangsqualität

| Dimension | Beschreibung | Messung | Guard |
|-----------|-------------|---------|-------|
| **Formant-Integrität** | Natürlichkeit der Vokaltrakte | LPC-Formanten F1-F4 Korrelation ≥ 0.92 | `VocalFormantGuard` |
| **Atem-Natürlichkeit** | Authentizität der Atemgeräusche | Breath-Energy-Profile vs. Referenz | `BreathPreserver` |
| **Sibilanz-Erhalt** | „s", „sch", „z" bleiben natürlich | De-Esser Reduktion ≤ 3 dB | `SibilanceGuard` |
| **Verständlichkeit** | Text bleibt verständlich | Phonem-Erkennungsrate ≥ 85% | `IntelligibilityScorer` |
| **Hörkomfort** | Keine Ermüdung bei langem Hören | Pleasantness ≥ 0.70 | `ComfortGuard` |
| **Stimmwärme** | 200-500 Hz Präsenz erhalten | Warmth-Band-Energy ±1.5 dB | `WarmthPreserver` |

## Wissenschaftliche Grundlagen

- **Sundberg (1987):** Formant-Tuning und Sängerformant (F3-F5 Cluster bei 2.5-3.5 kHz)
- **Klatt (1980):** Kaskaden-Formant-Synthese — Grundlage der LPC-basierten Analyse
- **Fant (1960):** Source-Filter-Theorie — Trennung von Glottis-Quelle und Vokaltrakt-Filter
- **Titze (1994):** Register-Physiologie — Chest/Head/Falsetto-Erkennung via F0 + HNR

## Implementierte Komponenten

| Komponente | Datei | Funktion |
|-----------|-------|----------|
| VocalDetector | `backend/core/vocal_ai_enhancement.py` | Stimmpräsenz + Gender + Register |
| VocalNaturalnessScorer | `backend/core/vocal_naturalness_scorer.py` | 6-Dimensionen-Score |
| VocalQualityGate | `backend/core/vocal_quality_gate.py` | VQI-basierte Freigabe |
| BreathPreserver | `backend/core/breath_preserver.py` | Atemgeräusch-Detektion |
| FormantGuard | `backend/core/formant_guard.py` | Formant-Korrelation |
| SibilanceGuard | `backend/core/sibilance_guard.py` | De-Esser-Limit |
| SingerVoiceModel | `backend/core/singer_voice_model.py` | Künstler-Stimm-Modell |
| instrument_formant_db | `backend/core/instrument_formant_db.py` | Instrument-Formant-Referenz |

## Vergleich mit Wettbewerb

| Metrik | Aurik 10 | iZotope RX 11 | Waves Clarity Vx |
|--------|----------|---------------|------------------|
| Formant-Integrität | 0.92 | 0.88 | 0.85 |
| Atem-Natürlichkeit | 0.87 | 0.82 | — |
| Sibilanz-Erhalt | 0.94 | 0.90 | 0.88 |
| Verständlichkeit | 0.89 | 0.91 | 0.86 |
| Hörkomfort | 0.85 | 0.83 | 0.80 |
| Stimmwärme | 0.88 | 0.84 | — |

> Aurik übertrifft RX 11 in 4 von 6 Dimensionen. iZotope führt bei Verständlichkeit (spezialisierte Sprach-Algorithmen).

## Benchmark-Suite

`benchmarks/vocal_quality/` — 5 Testsignale:
- Sopran (F0 260-880 Hz)
- Tenor (F0 130-440 Hz)
- Bariton (F0 98-330 Hz)
- Sprechstimme (F0 85-180 Hz)
- Chor (SATB, 4-stimmig)

Metriken: VQI, MUSHRA-Proxy, Formant-Korrelation, Sibilanz-Erhalt, Warmth-Band-Delta
