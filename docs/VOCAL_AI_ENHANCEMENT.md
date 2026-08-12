# Vocal AI Enhancement — Gender-Aware Processing (Aurik 10.0.8)

## Übersicht

Das Aurik 10.0.8 Vocal AI Enhancement System implementiert die **stimmtyp-adaptive Gesangs- und Sibilantenverarbeitung** unter **vollständigem Erhalt von Atem, Emotion und Authentizität** (Phase 19 + Phase 42 + VocalAIEnhancement + ConsonantEnhancement).

---

## ✅ Vollständig implementierte Features

### 1. Gender Detection (KI-basiert)

**Eigenentwicklung** basierend auf:

- **Formant-Analyse (F1-F4)**: Erkennung geschlechtsspezifischer Vokaltrakt-Resonanzen
- **Fundamentalfrequenz (F0)**: Pitch-Detection via Autocorrelation
- **Spektraler Schwerpunkt**: Energieverteilung im Frequenzspektrum
- **Harmonics-Struktur**: Obertonverteilung

### WORLD-Vocoder-Quervalidierung

Formant-Kreuzvalidierung via DIO/Harvest f₀ + CheapTrick-Spektralkurve (Morise et al. 2016):

- Abweichung LPC ↔ WORLD > 15 % → WORLD-Wert bevorzugt

### Pitch-Tracking

- **Primär**: CREPE (full model, 85 MB ONNX, lokal gebundelt) — Voicing-Konfidenz ≥ 0.6
- **Fallback**: pYIN (Mauch & Dixon 2014)

### ConsonantEnhancement (Frikative: s, f, sch, th)

- Erkennungskriterium: ZCR > 0.3 + Energie in 4–16 kHz dominant
- HF-Anhebung ≤ +6 dB im Frikativ-Band (stimmtyp-adaptiv)
- MALE: 5–10 kHz | FEMALE: 6–12 kHz | CHILD: 7–14 kHz
- Invariante: SNR_frikativ_after ≥ SNR_frikativ_before + 3 dB
- Crossfade 5 ms (Hanning) an Voiced/Unvoiced-Übergängen

- Mature (55-70)
- Senior (>70)

---

### 2. Gender-Aware De-Esser (Phase 19)

**Adaptive Sibilanten-Reduktion** mit geschlechtsspezifischen Parametern:

#### Männer

```python
Frequenzbereich: 5.000 - 10.000 Hz
Threshold: -25 dB
Ratio: 3:1
Attack: 1.0 ms
Release: 50 ms
```

#### Frauen

```python
Frequenzbereich: 6.000 - 12.000 Hz
Threshold: -23 dB
Ratio: 2.5:1
Attack: 0.5 ms
Release: 40 ms
```

#### Kinder

```python
Frequenzbereich: 7.000 - 14.000 Hz
Threshold: -20 dB
Ratio: 2:1
Attack: 0.3 ms
Release: 30 ms
```

**Emotion Preservation Modes:**

1. **MAXIMUM**: Minimale Eingriffe, maximale Authentizität
   - Ratio × 0.5 (weniger Kompression)
   - Threshold -3 dB (höherer Schwellwert)

2. **BALANCED**: Balance zwischen Technik und Emotion (Standard)
   - Originale Parameter

3. **TECHNICAL**: Optimale technische Qualität
   - Ratio × 1.5 (stärkere Kompression)
   - Threshold +3 dB (niedrigerer Schwellwert)

4. **TRANSPARENT**: Unsichtbare Verarbeitung
   - Adaptive Parameter basierend auf Material

---

### 3. Breath Preservation (Atemerhalt)

**Intelligente Atemgeräusch-Verarbeitung:**

#### Unterscheidung zwischen:

- **Künstlerischen Atemgeräuschen** (werden erhalten):
  - Pre-Phrase Breaths (vor Gesangsphrasen)
  - Emotionale Atemgeräusche (hohe Intensität)
  - Stilistische Atemgeräusche (rhythmisch, gemustert)

- **Störenden Atemgeräuschen** (werden reduziert):
  - Zwischenpausen ohne musikalischen Kontext
  - Technische Artefakte
  - Übermäßig laute Atemgeräusche

#### Preservation Ratio

```python
preservation_ratio = 0.7  # 70% Erhalt (empfohlen)

# Künstlerische Atemgeräusche:
reduction_factor = 1.0 - (0.7 × 0.2) = 0.86  # Nur 14% Reduktion

# Störende Atemgeräusche:
reduction_factor = 1.0 - ((1-0.7) × 0.8) = 0.76  # 24% Reduktion
```

---

### 4. Formant Preservation (Voice Identity)

**Stimm-Identität wird vollständig erhalten:**

- **LPC-basierte Formant-Tracking** (F1-F4)
- **Spektrale Hüllkurven-Erhaltung**
- **Vokal-Transitions-Schutz**
- **Korrelationsbasierte Verifikation**

**Preservation Score:**

```python
formant_preservation_score = correlation(original_formants, processed_formants)
# Target: > 0.95 (95% Erhalt)
```

---

### 5. Emotion Preservation

**Emotionale Authentizität wird geschützt:**

#### Emotion Detection:

- **Pitch-Variation**: Std(F0) / Mean(F0)
- **Dynamic Range**: Max - Min Amplitude
- **High-Harmonic Energy**: Obertongehalt

#### Emotion Preservation Score:

```python
emotion_diff = abs(original_emotion - processed_emotion)
emotion_preservation_score = 1.0 - emotion_diff
# Target: > 0.85 (85% Erhalt)
```

**Adaptive Processing:**

- Hohe Emotion → Geringere Eingriffe
- Neutrale Emotion → Stärkere Enhancement möglich

---

## 🎯 Verwendung

### Basic Usage

```python
from core.vocal_ai_enhancement import UnifiedVocalAIEnhancer, EmotionPreservationMode

# Initialize
enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

# Enhance vocals
result = enhancer.enhance(
    audio,
    emotion_mode=EmotionPreservationMode.BALANCED,
    breath_preservation=0.7,  # 70% Erhalt
    sibilance_reduction=True
)

# Results
print(f"Gender: {result.characteristics.gender.value}")
print(f"Age: {result.characteristics.age_group.value}")
print(f"Sibilance Reduced: {result.sibilance_reduced_db:.1f} dB")
print(f"Breath Preserved: {result.breath_preserved_ratio:.1%}")
print(f"Emotion Preserved: {result.emotion_preservation_score:.1%}")
print(f"Formant Preserved: {result.formant_preservation_score:.1%}")
```

### Integration in AI Framework

```python
from core.ai_framework import AurikAIFramework

framework = AurikAIFramework(sample_rate=48000)

# Vocal enhancement
result = framework.enhance_vocals(
    audio,
    emotion_mode="balanced",  # "maximum", "balanced", "technical", "transparent"
    breath_preservation=0.7
)
```

---

## 📊 Qualitäts-Metriken

### Preservation Targets

| Metrik | Zielwert | Bedeutung |
| -------- | ---------- | ----------- |
| **Formant Preservation** | > 95% | Stimm-Identität erhalten |
| **Emotion Preservation** | > 85% | Emotionale Authentizität |
| **Breath Preservation** | 70% (konfigurierbar) | Natürlicher Atem |
| **Sibilance Reduction** | -10 bis -20 dB | Je nach Schweregrad |
| **Quality Improvement** | > 0 | Gesamtqualität verbessert |

### Gender Detection Accuracy

| Gender | F0 Range (Hz) | Formant F1 (Hz) | Formant F2 (Hz) |
| -------- | --------------- | ----------------- | ----------------- |
| **Male** | 85-180 | 270-730 | 840-2290 |
| **Female** | 165-255 | 310-860 | 920-2790 |
| **Child** | 200-500 | 370-1030 | 1170-3330 |

---

## 🔬 Wissenschaftliche Grundlagen

### Formant Theory

- **Peterson & Barney (1952)**: Formant frequencies for vowels
- **Fant (1960)**: Acoustic Theory of Speech Production
- **Hillenbrand et al. (1995)**: Formant frequencies in American English

### Sibilance Processing

- **ITU-T P.800**: Subjective quality assessment
- **ANSI/ASA S3.5**: Sibilance detection standards
- **Psychoacoustic Models**: Bark scale, critical bands

### Emotion in Voice

- **Scherer (2003)**: Vocal expression of emotion
- **Juslin & Laukka (2003)**: Communication of emotions in vocal expression
- **Russell (1980)**: Circumplex model of affect

---

## 🎼 Integration mit bestehenden Phasen

### Phase 19: Advanced De-Esser

- `processing/adaptive_deesser.py` (1854 Zeilen)
- Gender/Age-aware Sibilant Processing
- 24 Perceptual Sub-Bands (Bark-Scale)
- ML-based Phoneme Classification

### Phase 42: Vocal Enhancement

- `core/phases/phase_42_vocal_enhancement.py` (1708 Zeilen)
- 8-Stage Enhancement Pipeline
- Formant Analysis & Correction
- Harmonic Enhancement
- Singer's Formant Boost

### Unified Integration

- `core/vocal_ai_enhancement.py` (NEU, 870 Zeilen)
- `core/ai_framework.py` (ERWEITERT)
- Vollständige KI-Integration aller Vocal-Features

---

## ✅ Erfüllung der Anforderungen

### ✓ Individuelle Verbesserung von Gesang

- Gender-adaptive Parameter für Männer, Frauen, Kinder
- Altersgruppen-spezifische Anpassungen
- Adaptive Enhancement basierend auf Stimm-Charakteristika

### ✓ Sibilanten-Behandlung

- Geschlechtsspezifische Frequenzbereiche
- Adaptive Threshold und Ratio
- Emotion-preserving Kompression

### ✓ Erhalt von Atem

- Intelligente Klassifikation (künstlerisch vs. störend)
- Konfigurierbare Preservation Ratio
- Kontext-bewusste Reduktion

### ✓ Erhalt von Emotion

- Emotion Detection (Pitch-Variation, Dynamics, Harmonics)
- Emotion Preservation Score Tracking
- Adaptive Processing basierend auf Emotion Intensity
- 4 Preservation Modes (Maximum, Balanced, Technical, Transparent)

### ✓ Erhalt von Authentizität

- Formant-basierte Voice Identity Preservation
- Spektrale Hüllkurven-Erhaltung
- Minimal-invasive Processing-Strategien
- Quality Gates & Automatic Revert bei Qualitätsverlust

---

## 🧪 Tests

Vollständige Test Suite: `tests/test_vocal_ai_enhancement.py`

**Test Coverage:**

- Gender Detection (Male, Female, Child)
- Formant Detection
- Breathiness Detection
- Sibilance Detection
- Gender-Aware De-Essing
- Emotion Preservation Modes
- Breath Preservation Levels
- Formant Preservation
- Emotion Preservation
- Stereo Processing
- Integration Tests
- Performance Tests

**Run Tests:**

```bash
pytest tests/test_vocal_ai_enhancement.py -v
```

---

## 📈 Performance

### Processing Speed

- **1 Second Audio**: < 1.0s (Realtime-fähig)
- **10 Second Audio**: < 10s
- **Memory**: ~50 MB

### Latency

- **Gender Detection**: ~100-200ms
- **De-Essing**: ~50-100ms
- **Breath Processing**: ~50-100ms
- **Total Pipeline**: ~200-400ms (Buffer möglich)

---

## 🎯 Roadmap Status

| Phase | Status | Features |
| ------- | -------- | ---------- |
| **Phase 19** | ✅ COMPLETE | Gender-Aware De-Esser |
| **Phase 42** | ✅ COMPLETE | Vocal Enhancement Pipeline |
| **AI Integration** | ✅ COMPLETE | Unified Vocal AI Framework |
| **Testing** | ✅ COMPLETE | Comprehensive Test Suite |
| **Documentation** | ✅ COMPLETE | This Document |

---

## 👥 Authoren

**Aurik 10.0.0 Development Team**

- AI Framework Integration
- Gender Detection Algorithm
- Emotion Preservation System
- Breath Classification Logic

**Date**: 15. Februar 2026
**Version**: 1.0.0

---

## 📝 Lizenz

Apache 2.0 - Aurik 10.0.0
Musical Excellence First

---

## 🙏 Credits

Basierend auf wissenschaftlichen Erkenntnissen von:

- Fant, Peterson, Barney (Formant Theory)
- Scherer, Juslin (Emotion in Voice)
- ITU-T, ANSI Standards (Audio Quality)
- Open Source: SciPy, NumPy (Signal Processing)
