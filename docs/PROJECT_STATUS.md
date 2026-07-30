# 📊 Aurik 10.0.0 — Project Status Report

**Datum:** Juli 2026
**Version:** 10.0.14 (§v10.305 Startup-Integration + Unified Progress + Context-Aware Communication)
**Status:** ✅ Produktionsbereit | GPU-Detection failsafe | Dual-Progress live | Kontextbewusste Kommunikation | 10 GEBOTE (G71–G80) | Startup-Smoke-Test | 12 neue i18n-Keys

> Verbindlicher Ist-Stand: `.github/specs/01-14`, `.github/specs/v10.303.17_phase0_architecture.md`, `.github/specs/v10.304_ast_and_deep_chain_correction.md`, `.github/copilot-instructions.md`, `CLAUDE.md`.

---

## Executive Summary

**Aurik 10.0.0 ist ein autonom denkendes Musik-Restaurierungssystem — jeder Song wird individuell gemessen und optimiert.**

| Kennzahl | Wert |
| --- | --- |
| Tests | **~18.400** pytest-IDs (Juli 2026), 511 mit Markern |
| Phasen | **68 + 3 (Phase-0)** = 71 (davon 12 redundant durch Phase 0) |
| Materialien | **16** auto-erkannte Typen + Multi-Generation-Chain |
| Musical Goals | **14** psychoakustisch fundierte Ziele (Pleasantness-First) |
| SNR-Adaption | ✅ Click, Tape-Splice, MATERIAL_SENSITIVITY, CAUSE_PARAMS |
| Spectrum-Aware | ✅ Phase 16 (Final EQ), Phase 17 (Mastering Polish) |
| Harmonic-Aware | ✅ Phase 17 Saturation (misst Even/Odd-Harmonic-Ratio) |
| PQS MOS | **>= 4.0** (Minimum) / **>= 4.5** (internes Spitzenziel) |

> Evidenzhinweis: Diese Datei ist ein historischer Snapshot. Aussagen zu Spitzenqualität,
> Wettbewerbsführung oder formaler Hörtest-Nähe sind intern als Ziel- und Steuerungsrahmen
> zu verstehen und werden erst durch externe Blindtests und reproduzierbare Vergleichsstudien
> belastbar.
| DefectTypes | **62** erkennbare Defektarten (DetectionTypes in DefectScanner) |
| Kausal-Ursachen | **62** Behandlungs-Ursachen (CAUSES in CausalDefectReasoner) |
| Hardware | CPU + optionale AMD-GPU (ROCM/DirectML), Desktop (Linux AppImage & Windows 10/11) |
| Netzwerk | Keine Cloud, keine Serverabhängigkeiten — 100 % offline |

---

## 🧠 Kognitive Architektur — Vollständig implementiert

| Modul | Datei | Status |
| --- | --- | --- |
| **Phase-0 Apollo (Codec-Decompression)** | `plugins/apollo_phase0_integration.py` | ✅ v10.303.17 |
| **Phase-0 DeepFilterNet v3 (Denoising)** | `plugins/apollo_phase0_integration.py` | ✅ v10.303.17 |
| **Phase-0 Resemble Enhance (Enhancement)** | `plugins/apollo_phase0_integration.py` | ✅ v10.303.17 |
| **ChainedPhase0Preprocessor** | `plugins/apollo_phase0_integration.py` | ✅ v10.303.17 |
| **ApolloPlugin + Hallucination-Guard** | `plugins/apollo_plugin.py` | ✅ erweitert §2.46e |
| --- | --- | --- |
| `PerceptualEmbedder` | `core/perceptual_embedder.py` | ✅ |
| `CausalDefectReasoner` | `core/causal_defect_reasoner.py` | ✅ |
| `GPParameterOptimizer` (MOO-Pareto) | `core/gp_parameter_optimizer.py` | ✅ |
| `PerceptualQualityScorer` | `core/perceptual_quality_scorer.py` | ✅ |
| `MusicalGoalsChecker` (14 Ziele) | `backend/core/musical_goals/musical_goals_metrics.py` | ✅ |
| `MediumDetector` | `forensics/medium_detector.py` | ✅ |
| `DefectScanner` (62 DefectTypes) | `core/defect_scanner.py` | ✅ |
| `VocalAIEnhancement` | `core/vocal_ai_enhancement.py` | ✅ |
| `ExcellenceOptimizer` | `core/excellence_optimizer.py` | ✅ |
| `FeedbackChain` | `core/feedback_chain.py` | ✅ |
| `UnifiedRestorerV3` | `core/unified_restorer_v3.py` | ✅ |
| `EraClassifier` (§v10.303.42 chain-aware) | `backend/core/era_classifier.py` | ✅ Deep-Chain-Correction aktiv |
| `GermanSchlagerClassifier` | `core/genre_classifier.py` | ✅ |
| `TransientDecoupledProcessing` | `core/transient_decoupled_processor.py` | ✅ |
| `HarmonicPreservationGuard` | `core/harmonic_preservation_guard.py` | ✅ |
| `PerPhaseMusicalGoalsGate` | `core/per_phase_musical_goals_gate.py` | ✅ |
| `MicroDynamicsEnvelopeMorphing` | `core/micro_dynamics_envelope_morphing.py` | ✅ |
| `RestorabilityEstimator` | `core/restorability_estimator.py` | ✅ |
| `StemRemixBalancer` | `core/stem_remix_balancer.py` | ✅ |
| `RemasterDetector` | `core/remaster_detector.py` | ✅ |
| `AdaptiveGoalThresholds` | `backend/core/musical_goals/adaptive_goals_system.py` | ✅ |
| `GoalApplicabilityFilter` | `core/goal_applicability_filter.py` | ✅ |
| `GoalPriorityProtocol` | `core/goal_priority_protocol.py` | ✅ |
| `PhysicalCeilingEstimator` | `core/physical_ceiling_estimator.py` | ✅ |
| `EraAuthenticPerceptualCompletion` | `core/era_authentic_perceptual_completion.py` | ✅ |
| `IntroducedArtifactDetector` | `core/introduced_artifact_detector.py` | ✅ |
| `TemporalQualityCoherenceMetric` | `core/temporal_quality_coherence.py` | ✅ |
| `EmotionalArcPreservationMetric` | `core/emotional_arc_preservation.py` | ✅ |
| `EnsembleProcessor` | `core/ensemble_processor.py` | ✅ |
| `PerceptualAttentionModel` | `core/perceptual_attention_model.py` | ✅ |
| `MusikalischerGlobalplanDienst` | `backend/core/musikalischer_globalplan.py` | ✅ |
| `BatchSessionLearner` | `core/batch_session_learner.py` | ✅ |
| `ReferenceAnchorSynthesizer` | `core/reference_anchor_synthesizer.py` | ✅ |
| `LyricsGuidedEnhancement` (§2.36, §v10.303.50) | `backend/core/lyrics_guided_enhancement.py` | ✅ HF Decoder aktiv |
| `PhonemeTimeline` | `backend/core/phoneme_timeline.py` | ✅ |
| `GermanSchlagerClassifier` (Genre-Phase-1) | `backend/core/genre_classifier.py` | ✅ |
| `AstAudioSetClassifier` (§v10.304) | `backend/core/ast_audio_set_classifier.py` | ✅ |
| `OOMRecoveryCheckpoint` (§2.39) | `backend/core/recovery_checkpoint.py` | ✅ |
| `PerceptualSalienceEstimator` | `backend/core/perceptual_salience.py` | ✅ |

---

## 🎯 14 Musical Goals — Qualitätsstatus (v10.0.0 Pareto-Differenzierung)

Alle 14 Ziele werden durch `MusicalGoalsChecker.measure_all()` nach jeder Restaurierung geprüft.
Regression in einem anwendbaren Ziel macht das Feature ungültig.

| Ziel | Klasse | Prio | Restoration | Studio 2026 |
| --- | --- | --- | --- | --- |
| Natürlichkeit | `NatuerlichkeitMetric` | P1 | >= 0.90 | >= 0.90 |
| Authentizität | `AuthentizitaetMetric` | P1 | >= 0.88 | >= 0.88 |
| Tonales Zentrum | `TonalCenterMetric` | P2 | >= 0.95 | >= 0.97 |
| Timbre-Authentizität | `TimbralAuthenticityMetric` | P2 | >= 0.87 | >= 0.87 |
| Artikulation | `ArticulationMetric` | P2 | >= 0.85 | >= 0.85 |
| Emotionalität | `EmotionalitaetMetric` | P3 | >= 0.82 | >= 0.87 |
| Mikro-Dynamik | `MicroDynamicsMetric` | P3 | >= 0.88 | >= 0.92 |
| Groove | `GrooveMetric` | P3 | >= 0.83 | >= 0.88 |
| Transparenz | `TransparenzMetric` | P4 | >= 0.82 | >= 0.89 |
| Wärme | `WaermeMetric` | P4 | >= 0.75 | >= 0.80 |
| Bass-Kraft | `BassKraftMetric` | P4 | >= 0.78 | >= 0.85 |
| Separation-Treue | `SeparationFidelityMetric` | P4 | >= 0.78 | >= 0.82 |
| Brillanz | `BrillanzMetric` | P5 | >= 0.78 | >= 0.85 |
| Raumtiefe | `SpatialDepthMetric` | P5 | >= 0.70 | >= 0.75 |

`GoalApplicabilityFilter` deaktiviert physikalisch irrelevante Ziele automatisch (z. B. SpatialDepthMetric
bei Mono-Aufnahmen <= 1950). Mindestens 6 Ziele bleiben immer aktiv: Natürlichkeit, Authentizität,
Emotionalität, Transparenz, Timbre-Authentizität, Artikulation.

---

## 📋 68-Phasen-Pipeline (kanonisch, v10.0.8)

```text
DCOffset-Removal
-> TransientDecoupledProcessing (TDP/HPSS)
-> RestorabilityEstimator -> SongCalibrationProfile
-> EraClassifier + GermanSchlagerClassifier + MediumDetector (parallel)
-> GoalApplicabilityFilter -> AdaptiveGoalThresholds
-> DefectScanner (62 DetectionTypes) -> CausalDefectReasoner (62 CAUSES)
-> GPParameterOptimizer -> HarmonicPreservationGuard
-> PerPhaseMusicalGoalsGate (umhüllt jede Phase)
-> Phasen-Ausführung (01–68)
-> FeedbackChain -> PhysicalCeilingEstimator
-> MusicalGoalsChecker (14 Ziele)
-> MicroDynamicsEnvelopeMorphing
-> RestorationResult
```

- Phase 01–30: Defektkorrektur (Noise, Hum, Crackle, Clicks, Wow/Flutter, …)
- Phase 31–46: Enhancement (EQ, Stereo, Gesang, Instrumente)
- Phase 47–56: Mastering + SpectralBandGapRepair + Inpainting
- Phase 57: Print-Through-Reduktion (reel_tape)
- Phase 58: LyricsGuidedEnhancement (§2.36, Whisper-Tiny ONNX + wav2vec2)
- Phase 59–64: Neue Spezialdefekte (ModulationNoise, InnerGrooveDistortion, GrooveEcho, Crosstalk, IntermodulationDistortion, TapeSplice)

---

## 📦 17 Materialien

| Material | Prioritäts-Phasen | PQS MOS |
| --- | --- | --- |
| `tape` | 24, 29, 12 | >= 4.2 |
| `reel_tape` | 29, 03, 24, 55 | >= 4.3 |
| `vinyl` | 09, 12, 30 | >= 4.0 |
| `shellac` | 03, 06, 01 | >= 3.8 |
| `wax_cylinder` | 03, 06, 01, 29 | >= 3.5 |
| `wire_recording` | 12, 24, 03, 29 | >= 3.6 |
| `lacquer_disc` | 01, 09, 03, 29 | >= 3.7 |
| `dat` | 24, 02, 23 | >= 4.4 |
| `cd_digital` | 23, 06, 40 | >= 4.5 |
| `mp3_low` | 23, 03, 50 | >= 3.9 |
| `mp3_high` | 23, 50 | >= 4.2 |
| `aac` | 23, 38, 06 | >= 4.2 |
| `minidisc` | 23, 06, 07 | >= 4.0 |
| `streaming` | 03, 23, 50 | >= 4.1 |
| `unknown` | Alle Tier-1 | >= 3.8 |

---

## 🎤 Stimmtyp-Adaptierung (VocalAIEnhancement)

| Typ | F0-Bereich | F1-Bereich | De-Essing-Ziel |
| --- | --- | --- | --- |
| MALE | 85–180 Hz | 270–730 Hz | 5–10 kHz |
| FEMALE | 165–255 Hz | 310–860 Hz | 6–12 kHz |
| CHILD | 200–500 Hz | 370–1030 Hz | 7–14 kHz |
| ANDROGYNOUS | auto-detect | auto-detect | adaptiv |
| UNKNOWN | — | — | FEMALE-Fallback |

Invarianten: Formant-Pearson >= 0.95 · Breathiness +/-0.05 · Vibrato +/-0.3 Hz
ConsonantEnhancement: Frikative-SNR >= +3 dB · HF-Anhebung <= +6 dB · Crossfade 5 ms

---

## 🔧 Entwicklungs-Roadmap

### ✅ Abgeschlossen

| Version | Milestone | Tests |
| --- | --- | --- |
| v10.0.0 | UnifiedRestorerV3, Material-Auto-Detektion | 6 Tests |
| v10.0.0 | ML-Hybrid, 12 Materialien, 21 DefectTypes, 55 Phasen | 166 Tests |
| v10.0.0 | Kognitive Architektur (5 Kernmodule), VoiceGender, PANNs | 206 Tests |
| v10.0.0 | Über-SOTA DSP (OMLSA/IMCRA, pYIN, NMF-b, PGHI) | 222 Tests |
| v10.0.0 | GrooveMetric (#8), MRSA, Psychoakust. Masking, HarmonicLattice | 5169 Tests |
| v10.0.0 | 14 Musical Goals, EraClassifier, TonalCenter, MicroDynamics | 6073 Tests |
| v10.0.0 | StemRemixBalancer, EnsembleProcessor, IAD, BatchSessionLearner | 6180 Tests |
| v10.0.0 | TDP, HPG, PMGG, MDEM | 6312 Tests |
| v10.0.0 | E2E-Tests, TIER-Invarianten, PMGG-Fixes, v2-Cleanup | 6312 Tests |
| v10.0.0 | WPE als kanonisches Dereverb (SGMSE+ entfernt) | 6312 Tests |
| v10.0.0 | RemasterDetector, EraResult.is_remaster_suspected, temporale Defektverortung | 6347 Tests |
| v10.0.0 | Spec-Konsistenz-Audit, JSON-Schema, Genre-Profile, DDSP, UI-Shortcuts | 6312 Tests |
| v10.0.0 | Spec-Konsistenz-Audit: 6 Korrekturen (EraResult, PMGG-Default, MaterialQuality, GP-Genre-Keys) | 6312 Tests |
| v10.0.0 | Infrastruktur: SBOM, GP-Backup, i18n-Tests, Export-Roundtrip | 6312 Tests |
| v10.0.0 | Performance: SHA256-Cache, parallele Eingangs-Analyse, PMGG-Sample-Dauer, Warmup-Thread | 6312 Tests |
| v10.0.0 | §Dach: MusikalischerGlobalplan, 13 Ära-Profile, Genre-Modifikatoren, 17 Phase-Adjustments | 6312 Tests |
| v10.0.0.x | §SR-Invariante: assert sample_rate==48000 lückenlos an allen API-Einstiegspunkten | historischer Teststand |
| v10.0.0–83 | KMV Stufe-2, ML-Headroom-Guard, OOM-Checkpoint, Denker-Differenzierung, Song-Kalibrierung | 7.500+ Tests |
| v10.0.0–91 | Dual-SR-Vertrag, PMGG SNR-Proxy-Fixes (§9.7.11–14), Stab.-Invarianten | 8.500+ Tests |
| v10.0.0–99 | PMGG SNR-Proxies brillanz/transparenz/waerme, Codec-Repair, AMRB-Kalibrierung | 9.500+ Tests |
| v10.0.0–102 | Lyrics-Produktivpfad, Phasen 59–64, Genre-Phase-1 (Family+Top-k+Open-Set) | ~18.400 Tests |

### 🔜 Geplant

| Version | Milestone |
| --- | --- |
| v10.0 | Multi-Modal-Restaurierung (Audio + Metadaten + Visual) |

---

## 📤 Export & Qualitätsnormen

**Importformate:** WAV, AIFF, FLAC, MP3, AAC/M4A, OGG, WMA, Opus, CAF  
**Exportformate:** FLAC (24-bit), WAV (24-/16-bit), MP3 CBR/VBR (LAME), OGG (q9), AIFF (24-bit)

Lautheit: EBU R128 — -14 LUFS (Streaming) / -18 LUFS (Archiv)  
True-Peak: -1.0 dBTP (ITU-R BS.1770-5)  
Dithering: POW-r Typ 3 bei 24->16-bit; Fallback: TPDF

---

## ⚙️ Technische Konstanten

| Parameter | Wert |
| --- | --- |
| Interne SR | 48 000 Hz (Pflicht, `assert sample_rate == 48000`) |
| Bit-Tiefe intern | float32, [-1, 1] |
| Hardware | CPU + optionale AMD-GPU (`providers=["CPUExecutionProvider", "ROCMExecutionProvider"]`) |
| Resampling | Lanczos-4 (`scipy.signal.resample_poly`, Kaiser b=14) |
| GP-Gedächtnis | `~/.aurik/gp_memory/<material>.json` |
| FeedbackChain | max. 5 Iterationen, D\|MOS\| < 0.02 |
| PMGG Regression-Threshold | adaptiv: 0.012 / 0.040 / 0.060 |
| PMGG Max-Retries | 5 (strength x 0.65 -> 0.50 -> 0.35 -> 0.20 -> 0.10) |
| Chunk-Verarbeitung | defektdichte-adaptiv: 5 s / 15 s / 60 s / 120 s |

---

## 🔬 Primäre ML-Modelle (lokal gebündelt, 100 % offline)

| Modell | Anwendungsfall | Größe | Fallback |
| --- | --- | --- | --- |
| DeepFilterNet v3.II | Breitrauschen (NR) | ~37 MB ONNX | OMLSA/IMCRA DSP |
| MDX23C Kim_Vocal_2/Kim_Inst | Stem-Separation | 2x 64 MB ONNX | NMF-b |
| Apollo | Codec-Artefakte | ~65 MB ONNX | DSP Spectral Repair |
| Vocos 24 kHz | Neuronaler Vocoder | ~52 MB ONNX | HiFi-GAN -> PGHI |
| CREPE full | Pitch-Tracking f0 | ~85 MB ONNX | pYIN DSP |
| PANNs CNN14 | Audio-Tagging | ~81 KB ONNX | DSP Fingerprint |
| DiffWave | Dropout-Inpainting | ~552 KB ONNX | NMF-b + Sinusoidal |
| Resemble-Enhance | Apollo-Fallback | ~41 MB ONNX | DSP Spectral Repair |
| HiFi-GAN | Vocoder-Fallback | ~3.6 MB ONNX | PGHI-ISTFT |

---

## ✅ Universelle Garantien

| Garantie | Prüfung |
| --- | --- |
| Kein NaN/Inf im Audio-Ausgang | `np.isfinite(audio).all()` |
| Kein Clipping | `np.max(np.abs(audio)) <= 1.0` |
| Chroma-Korrelation | Pearson >= 0.95 |
| Pass-Through (sauberes Material) | PQS-MOS-Verlust <= 0.05, alle 14 Goals stabil +/-0.02 |
| Rauschboden (Studio-2026) | Residual <= -72 dBFS, A-gew. <= -75 dB(A), 0 Musical-Noise |
| Temporale Kohärenz | MOS-Spanne <= 0.30, sigma(MOS) <= 0.15 |
| Stereo-Authentizität | Mono-Ära M/S-Korrelation >= 0.97 |
| HF-Kumulativ-Limit | Presence + Air kumulativ <= +4 dB |

---

## §v10.303 Patch Notes (2026-07-27)

### Kritische Bug-Fixes
- **safe_istft Monkey-Patch**: `_scipy_signal.safe_istft = _safe_stft` (STFT statt ISTFT) → betraf 5 Phasen über Monate
- **`_compute_current_restorability` außerhalb Klasse**: Methode war nach Klassen-Ende definiert → Exception-Handler crashte
- **Phase 64 Broadcast-Crash**: `shape[-1]` statt `shape[0]` → `_n_samples=2` statt 10.8M
- **`original_audio_reference` NameError**: Variable im falschen Scope → UV3 crashte, Fallback-Pfad

### Material-Awareness (Planer-Intelligenz)
- **Low-Confidence-Gate**: Enhancement-Phasen bei `material_confidence < threshold` gestrippt
- **Adaptiver Threshold**: `0.30 + rs × 0.001` statt starrer 0.35
- **Denker-Feedback-Loop**: Gestrippte Familien über Songs gelernt
- **PID Confidence-Strip**: Planer strippt ab Song 1 (nicht erst ab Song 2)
- **ExzellenzDenker Convergence Guard**: Max 3 Reparatur-Versuche (statt 6×9s Loop)

### De-Esser Optimierungen
- Crest-Faktor-basierte Sibilanz/Artefakt-Unterscheidung
- Graduierte Response (0.0–1.0) statt Binär-Gate
- Material-adaptive Crest-Kalibrierung (Shellac: Baseline 10.0)
- Aurik-8 Degradation Guard (bw_loss>0.7 → Stack skip)
- bw_loss Enum-Key-Fallback (DefectType.BANDWIDTH_LOSS)

### Exception-Forensik
- KNOWN_PATTERNS: 7 → 15 Einträge (0 unklassifiziert)
- Tuple→ndarray Type Guard: `_normalize_phase_result()` eliminiert 70 P7-Exceptions
- Prognose: 507 → ~200 Exceptions/Run

### GUI
- `_draw_icon`, `_row_y`, `_bar_rect` implementiert (MusicalGoalsRadar)
- Prognose-Phasenschätzung jetzt Confidence-bewusst (34→40 → 13→22)
- OneTakeExport: Adaptiver Gain-Cap (5→1-2 Retries)

### Erkenntnisse
1. **Ein 3-Zeilen-Bug kann 5 Phasen über Monate korrumpieren** — ohne sichtbaren Absturz
2. **Enhancement ≠ Core-Restauration** — getrennte Behandlung ist architektonisch notwendig
3. **Familien-basierte Gates sind robuster als Phasen-Whitelists** — 10 Core-Familien statt 6 Phasen-IDs
4. **Crest-Faktor ist der beste Sibilanz-Indikator** — zuverlässiger als HF-Ratio allein
5. **Exception-Zählung ohne Klassifikation ist wertlos** — erst P1-P15 macht Muster sichtbar
6. **Material-Adaptivität ist Pflicht, nicht Optimierung** — Shellac ohne Crest-Guard würde Knistern de-essen

---

_Aurik 10.0.8 — Juli 2026 (Stand: 03.04.2026)_
