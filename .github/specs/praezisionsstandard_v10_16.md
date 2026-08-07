# §v10.16 Präzisionsstandard — Selbstkalibrierung auf Toningenieur-Niveau

## Prinzip

> Jeder kalibrierbare Parameter in Aurik wird mit der Präzision eines
> Toningenieurs eingestellt — nicht mit der eines Rastschalters.

## Standard-Methoden

### 1. Binäre Suche (für kontinuierliche Parameter)

| Parameter | Wert |
|---|---|
| Algorithmus | Intervallhalbierung mit Regression-Feedback |
| Iterationen | 12 (1/4096 Grundauflösung = ±0.025%) |
| Effektive Präzision | ±0.005% (durch quadratische Interpolation am Endpunkt) |
| Abbruch | Intervallbreite < 0.5% ODER 12 Iterationen |

**Gilt für:**

- PMGG Phasen-Retry-Stärke (alle 64 Phasen)
- PostGate Komponenten-Stärke (5 Post-Processing-Komponenten)
- OneTakeExport LUFS-Gain-Korrektur (±0.1 dB)

### 2. Material-Kalibrierung (für parameterlose Entscheidungen)

| Parameter | Wert |
|---|---|
| Messgrößen | Mikrodynamik (Perzentil-Spread), Spektrale Varianz (Frame-Kosinus-Distanz) |
| Stützstellen | 200 RMS-Blöcke (50 ms), 40 FFT-Frames (2048-pt, Hanning) |
| Ausgabe-Präzision | ±2% des Stärke-Bereichs |

**Gilt für:**

- HumanizationPass.calibrate_strength()
- Excellence-Presence-Blend (aus Restorability abgeleitet)
- ListeningEQ Band-Korrekturen (10 Bänder, ±0.5 dB Mindestkorrektur)

### 3. Quality-Gate-Schwellwerte

| Gate | Schwelle | Begründung |
|---|---|---|
| PMGG Regression | Adaptiv (0.012–0.060 je Restorability) | ±1% Goal-Score = hörbare Veränderung |
| PostGate Regression | 0.015 (fix) | Strenger — Post-Processing ist finale Politur |
| True Peak Warn | −0.3 dBTP | EBU R128 |
| True Peak Fail | 0.0 dBTP | Digital Clipping |
| LUFS Toleranz | ±2 LUFS | EBU R128 |
| Fatigue Warn | 0.4 | Oberhalb beginnt subjektive Ermüdung |
| Stereo-Korrelation | −0.3 | Unterhalb: hörbare Phasenprobleme in Mono |

### 4. Messgenauigkeit

| Metrik | Rauschgrenze | Auflösung |
|---|---|---|
| PMGG _measure_quick | ±0.01 (1%) | 0.001 (0.1%) |
| ListeningFatigueMetric | ±0.03 (3%) | 0.01 (1%) |
| ExportQualityGate LUFS | ±0.5 dB | 0.1 dB |
| ExportQualityGate TruePeak | ±0.3 dB | 0.1 dB |
| STCG delay | ±0.5 Samples | 0.1 Samples |

## Implementierungsstatus

| Komponente | Methode | Präzision | Status |
|---|---|---|---|
| PMGG Retry-Stärke | Binäre Suche 12 Iter. | ±0.025% | ✅ v10.16 |
| PostGate Stärke | Binäre Suche 12 Iter. | ±0.025% | 🔲 TODO |
| HumanizationPass | Material-Kalibrierung | ±2% | ✅ v10.15 |
| ListeningEQ | Band-Δ-Analyse | ±0.5 dB/Band | ✅ v10.15 |
| Excellence-Blend | Restorability-Formel | stetig [0.03, 0.12] | ✅ v10.15 |
| OneTakeExport Gain | Direkt-Δ | ±0.1 dB | ✅ v10.15 |
| OneTakeExport Limiter | Brickwall −0.3 dBTP | ±0.1 dB | ✅ v10.15 |
| STCG Guard | 20ms universell | ±0.5 Samples | ✅ v10.14 |
| PMGG Goals | _measure_quick | ±1% | ✅ bestehend |

## Ausnahmen

Folgende Systeme verwenden KEINE binäre Suche, weil:

- **Detection-basierte Reparaturen** (VocalScratchRepair, TapeHeadArtifact, SibilanceMax, DirectDefect): binär — sie reparieren oder nicht
- **Deterministische Transformationen** (DC-Offset, Resampling, Silence-Maske): mathematisch exakt
- **ML-Inferenz** (FlashSR, Demucs, BS-RoFormer): Stärke nicht kontinuierlich kalibrierbar (Modell-Output ist diskret)
