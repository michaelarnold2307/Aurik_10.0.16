# Aurik 10 — Tonträgerketten (Media Transfer Chains)

> Stand: 10.0.20 | Dokumentiert die Logik der Medien-Erkennung

## Konzept

Eine **Tonträgerkette** beschreibt den vollständigen Medien-Lebenslauf einer Aufnahme:

```
Shellac (1930) → Vinyl (1955) → Cassette (1970) → MP3 (2005)
   └── Original        └── Neuauflage     └── Mixtape      └── Digitalisierung
```

**Jedes Glied** steht für ein physisches oder digitales Trägermedium, durch das die Aufnahme im Lauf der Zeit gegangen ist. Aurik erkennt diese Kette automatisch und passt die Restaurierung an JEDES Glied an.

## Warum ist die Kette wichtig?

| Kette | Bedeutung für die Restaurierung |
|-------|-------------------------------|
| `vinyl → mp3_high` | Vinyl-Knistern entfernen, MP3-Artefakte ignorieren (zu leise) |
| `cassette → mp3_low` | Bandrauschen + Dropouts + MP3-Kompressionsartefakte |
| `reel_tape → vinyl → cd` | Jedes Glied hat eigene Defekte — alle müssen behandelt werden |

**Falsche Kette = falsche Restaurierung.** Ein MP3-Rauschen wird anders behandelt als ein Vinyl-Knistern.

## Die 3 Detektoren

| Detektor | Was er erkennt | Konfidenz |
|----------|---------------|:---------:|
| **MediumDetector** | Physikalische Signalanalyse (Bandbreite, SNR, Stereo-Breite) | 50% Gewicht |
| **EraClassifier** | Ära → typisches Medium (z.B. 1960er → Vinyl) | 30% Gewicht |
| **DefectScanner** | Defektmuster → Medium (z.B. Knistern → Vinyl) | 20% Gewicht |

## Ära vs. Kette — kein Widerspruch

```
Ära 1960  =  WANN wurde aufgenommen  (Aufnahmejahr)
Kette     =  WORÜBER lief die Musik  (Medien-Historie)
```

**Beispiel:** Ein Song von 1930 als MP3 ist völlig normal — er wurde später digitalisiert.
Die Kette `shellac → vinyl → mp3_high` ist korrekt. Die Ära 1930 widerspricht NICHT dem mp3.

## Erkennbare Medien und ihre Produktionskette

**Kein Medium existiert isoliert.** Jedes Endformat hat zwingende Vorstufen:

| Endformat | Vollständige Produktionskette | Erklärung |
|-----------|------------------------------|-----------|
| `vinyl` | **reel_tape → lacquer_disc → vinyl** | Aufnahme auf Band → Lackfolie geschnitten → Vinyl gepresst |
| `cassette` | **reel_tape → cassette** | Aufnahme auf Band → Kassettenkopie |
| `cd` | **dat → cd** | Digitales Masterband → CD-Pressung |
| `mp3` | **cd → mp3** | CD → digital kodiert |
| `shellac` | **wax_cylinder → shellac** | Wachswalze → Schellack-Pressung |
| `streaming` | **cd → streaming** | CD → Opus/AAC kodiert |

**Beispiel für einen Song von 1977:**


## Material-Konsens

Bei widersprüchlichen Detektor-Ergebnissen entscheidet der **Material-Konsens** gewichtet:

```
MediumDetector: mp3_high × 0.46 × 0.50 = 0.230
EraClassifier:  vinyl    × 0.34 × 0.30 = 0.102
DefectScanner:  cassette × 0.54 × 0.20 = 0.108

→ Gewinner: mp3_high (höchster gewichteter Score)
→ Kette: ALLE erkannten Medien chronologisch: vinyl → cassette → mp3_high
```

## Dateien

| Datei | Funktion |
|-------|----------|
| `forensics/medium_detector.py` | Physikalische Signalanalyse (autoritativ) |
| `backend/core/era_classifier.py` | Ära-Klassifikation (CLAP + DSP) |
| `backend/core/defect_scanner.py` | Defektmuster → Material |
| `backend/core/material_consensus.py` | 3-Wege-Konsens + Kettenbau |
| `backend/core/pre_analysis.py` | Integration + Kette-Korrektur |
