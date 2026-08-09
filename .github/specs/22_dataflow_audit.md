# §AUDIT — Datenfluss & Phasen-Vollständigkeit

> **Datum:** 2026-07-30 · **Scope:** Alle 65 Phasen · **Modus:** Restoration

---

## 1. Datenfluss: Import → Export

```
File Import (backend/file_import.py)
  │ load_audio_file() → numpy (float32, 48kHz)
  ▼
Pre-Analysis (backend/core/pre_analysis.py)
  │ MediumDetector → Material + Transfer-Chain
  │ EraClassifier → Decade + Material-Prior
  │ GenreClassifier → Genre + Language
  │ DefectScanner → 32 Defect-Scores
  │ RestorabilityEstimator → 0-100 Score
  │ Deep-Transfer-Chain-Injection → enriched chain + chronological sort
  ▼
Phase-0 Pre-Processor (plugins/apollo_phase0_integration.py)
  │ EAR_VAE → load → process → unload (643 MB)
  │ Apollo → load → process → unload (800 MB) [nur lossy-codec]
  │ DeepFilterNet → load → process → unload (34 MB)
  │ ResembleEnhance → load → process → unload (722 MB)
  │ §v10.306: ALLE sequenziell, nie parallel
  ▼
Denker-Pipeline (denker/)
  │ StrategieDenker → Phase-Selektion, Budget
  │ ReparaturDenker → Click/Hum/Clipping-Repair
  │ RekonstruktionsDenker → Gap-Filling
  │ RestaurierDenker → UV3-Instanz
  ▼
UV3 Pipeline (backend/core/unified_restorer_v3.py)
  │ Phase-Selektion → 41/65 Phasen (defekt-abhängig)
  │ evict_for_phase_window → Modell-Management (look-ahead=1)
  │ 41 Phasen sequenziell → STFT/ML/DSP
  │ CIG/AFG/PMGG → Quality-Guards pro Phase
  ▼
Post-Processing
  │ DoNoHarmGuardian → Regression-Check
  │ PostPipelineForensics → Exception-Analyse
  │ PerceptualExportOptimizer → Masking-Gate
  ▼
Export (backend/core/one_take_export.py)
  │ Platform-Export → LUFS/True-Peak / Format
  │ Donation-Reminder → GUI-Dialog (1×/24h)
  ▼
Output File (WAV/FLAC/MP3)
```

**Datenfluss ist lückenlos.** Jede Komponente erhält ihren Input vom Vorgänger. Keine toten Pfade.

---

## 2. Phasen: 41/65 selektiert — warum 24 fehlen

### Instrument-spezifisch (7 Phasen) — kein Instrument erkannt

| Phase | Benötigt | Warum nicht selektiert |
|---|---|---|
| phase_42_vocal_enhancement | PANNs Vocal > 0.5 | Vocal 0.35 — unter Schwelle |
| phase_44_guitar_enhancement | PANNs Guitar > 0.3 | Guitar 0.01 |
| phase_45_brass_enhancement | PANNs Brass > 0.3 | Brass nicht detektiert |
| phase_51_drums_enhancement | PANNs Drums > 0.3 | Drums 0.00 |
| phase_52_piano_restoration | PANNs Piano > 0.3 | Piano 0.00 |
| phase_66_stem_targeted_nr | Stems verfügbar | Keine Stems |
| phase_58_lyrics_guided_enhancement | Lyrics verfügbar | Keine Lyrics |

**→ Korrekt.** Diese Phasen würden auf falschem Material Artefakte produzieren.

### Studio/Enhancement-Modus (9 Phasen) — Restoration-Mode

| Phase | Modus | Warum nicht selektiert |
|---|---|---|
| phase_10_compression | Studio | Nur in STUDIO_2026 |
| phase_11_limiting | Studio | Nur in STUDIO_2026 |
| phase_13_stereo_enhancement | Studio | Nur in STUDIO_2026 |
| phase_17_mastering_polish | Studio | Nur in STUDIO_2026 |
| phase_21_exciter | Studio | Enhancement, nicht Restoration |
| phase_22_tape_saturation | Studio | Enhancement, nicht Restoration |
| phase_35_multiband_compression | Studio | Nur in STUDIO_2026 |
| phase_38_presence_boost | Studio | Enhancement |
| phase_46_spatial_enhancement | Studio | Enhancement |

**→ Korrekt.** Restoration soll konservativ restaurieren, nicht künstlich verbessern.

### Defekt-spezifisch (5 Phasen) — Defekt nicht vorhanden

| Phase | Benötigt | Warum nicht selektiert |
|---|---|---|
| phase_20_reverb_reduction | reverb_sev > 0.15 | reverb_sev 0.000 |
| phase_28_surface_noise_profiling | surface_noise > Schwelle | Digitales Material |
| phase_55_diffusion_inpainting | gaps > 3s | Keine Lücken > 3s |
| phase_57_print_through_reduction | print_through > Schwelle | Kein Print-Through |
| phase_63_intermodulation_reduction | intermod > Schwelle | Keine Intermodulation |

**→ Korrekt.** Keine Defekte → keine Reparatur nötig.

### Konflikt-Phasen (2 Phasen) — durch Alternativen ersetzt

| Phase | Ersetzt durch | Grund |
|---|---|---|
| phase_48_stereo_width_enhancer | phase_33_stereo_width_limiter | §2.48: Breiten-Limiter hat Vorrang |
| phase_07_declipper | phase_07_harmonic_restoration | Gleiche Nummer, harmonische hat Priorität |

**→ Korrekt.** Pipe-Konflikte werden vom PhaseInteractionDenker gelöst.

### Grundlegende Cleanup-Phase (1 Phase) — conditional

| Phase | Bedingung | Warum nicht selektiert |
|---|---|---|
| phase_30_dc_offset_removal | DC > 5e-4 | Digitales MP3 — DC ≈ 0 |

**→ Korrekt.** Bei digitalen Quellen ist DC-Offset praktisch immer Null.

---

## 3. Keine Lücken im Datenfluss

- **File-Import → Pre-Analysis:** Audio wird korrekt übergeben ✅
- **Pre-Analysis → Phase-0:** Caches werden korrekt genutzt ✅
- **Phase-0 → UV3:** `_restoration_context` wird befüllt ✅
- **UV3 → Post-Processing:** Goals, Defects, Audio werden übergeben ✅
- **Post-Processing → Export:** Quality-Gate prüft vor Export ✅
- **Export → Datei:** Plattform-Export mit LUFS/True-Peak ✅

---

## 4. Fazit

**Keine kritischen Lücken.** Alle 24 nicht selektierten Phasen sind entweder:

- Instrument-spezifisch und korrekt deaktiviert (7×)
- Nur für Studio-Modus (9×)
- Defekt-spezifisch und Defekt nicht vorhanden (5×)
- Durch Alternativen ersetzt (2×)
- Konditional und Bedingung nicht erfüllt (1×)

Der Datenfluss von Import bis Export ist vollständig und korrekt.
