# Aurik Spezifikation — §-Referenz-Index

> **852 eindeutige §-Referenzen** in `backend/core/`, `denker/`, `forensics/`.
> Dieser Index ist eine **Referenzhilfe** — er dokumentiert die häufigsten und
> architektonisch wichtigsten §-Referenzen. Verbindlich sind `AGENTS.md`
> (Repo-Root) und die dort definierte normative Kette; Einträge hier werden
> von **keinem** Pre-Commit-Hook erzwungen.

## §-Kategorien

| Präfix | Bedeutung |
|---|---|
| `§0` | Fundamentale Invarianten (Primum non nocere, NaN/Inf, LAG) |
| `§1-4` | Architektur (Module, Singletons, Contracts, Threading) |
| `§2` | UV3-Kern (Kalibrierung, Phasen, Goals, DSP) |
| `§4` | Phasen-spezifische Regeln |
| `§6` | Forensik (Medium-Detector, Defekt-Scanner, Era-Klassifikator) |
| `§7-8` | Pipeline-Intelligenz (CausalReasoner, Optimizer) |
| `§9` | Qualitäts-Metriken (Goal-Scoring, PQS, MOS) |
| `§V` | Vintage-Ästhetik-Guards (Soft-Saturation, Wärmeband, etc.) |
| `§SFT/UQ/AC/AF` | Subsysteme (Safety, Uncertainty, Phase-Pruning, Cross-Guard) |

---

## Fundamentale Invarianten (§0)

| §Ref | Bedeutung | Dateien |
|---|---|---|
| `§0` | Primum non nocere — keine Verschlechterung | UV3 |
| `§0a` | NaN/Inf-Schutz auf allen Ein- und Ausgaben | UV3 |
| `§0p` | Vocal-Focus-Analyzer: Singstimme erkennen und schützen | UV3, VFA |
| `§0c` | Short-Clip-Handling (< 10s): reduzierte Analyse | UV3 |
| `§0d` | LAG-Probe: Sample-genaue Latenzmessung | UV3 |
| `§0h` | Stereo-L/R-Konsistenz | UV3 |
| `§0j` | DC-Offset-Erkennung vor DSP | UV3 |
| `§0l` | Phasen-Linearität bewahren | UV3 |

## Architektur (§1-4)

| §Ref | Bedeutung | Dateien |
|---|---|---|
| `§2.8` | UV3 REST API Contract | UV3 |
| `§2.29` | PMGG Datenfluss-Invariante (Restorability) | UV3 |
| `§2.31` | MidCalibrate: Progress-basierte Rekalibrierung | UV3 |
| `§3.1` | NaN/Inf-Schutz (normativ) | UV3, Denker |
| `§3.2` | Singleton-Pattern (Thread-Safe, Double-Checked Locking) | Alle Denker |
| `§4.4` | Phase-Executor: OOM-Probe, PLM, Wall-Budget | UV3 |
| `§4.5` | Phase-ID-Validierung | UV3 |
| `§4.11` | Phase-Verbote (kein Denoise auf NR-Output, etc.) | UV3 |

## UV3-Kern (§2) — Kalibrierung & Phasen

| §Ref | Bedeutung | Dateien |
|---|---|---|
| `§2.44` | Per-Phase-Musical-Goals-Gate (PMGG) | UV3, PMGG |
| `§2.45` | Pegel-Monitoring (Pre/Post-Pegel pro Phase) | UV3 |
| `§2.45a` | Pegel-Drop-Guard (±0.5 dB Toleranz) | UV3 |
| `§2.46` | Tilt-Cap (Spektrale Neigungs-Begrenzung) | UV3 |
| `§2.46a` | Carrier-Chain-Invariante | UV3 |
| `§2.46b` | Source-Fidelity Spectral Tilt | UV3 |
| `§2.46e` | Room-Acoustics-Fingerprint | UV3 |
| `§2.46f` | Blind-Internal-Reference (BIR) | UV3 |
| `§2.47` | Material-Defect-Consistency | UV3 |
| `§2.48` | Cumulative-Interaction-Guard (CIG) | UV3 |
| `§2.49` | Artifact-Freedom (IAD-gate) | UV3 |
| `§2.51` | Phase-Skipping (deterministischer PID-Executor) | UV3 |
| `§2.54` | Effective-Targets (Physical-Ceiling) | UV3 |
| `§2.55` | Excellence-Optimizer (Core-Guard) | UV3 |
| `§2.56` | Song-Goal-Importance (Genre/Era/Material) | UV3, SGI |
| `§2.59` | **Contract-Validierung & Defekt-Namen-Sync (NEU 2026-07-09)** | CV, DM, SP |
| `§2.62` | Feedback-Chain (Post-Phasen-Retries) | UV3 |
| `§2.64` | Goal-Defizit-Feedback-Chain | UV3 |

## Forensik (§6)

| §Ref | Bedeutung | Dateien |
|---|---|---|
| `§6.2a` | Carrier-Chain-Invariante (Tape-Stufe in mp3-Chain) | UV3 |
| `§6.2c` | Dolby-NR-Erkennung | MD |
| `§6.3` | DefectScanner: 62 DefectTypes | DS |
| `§6.7` | Medium-Detector: Bayesian-Fusion (v10.0.0) | MD |
| `§6.7b` | File-Extension-Prior (Digital vs Analog) | MD |
| `§6.8` | Era-Precursor (reel_tape-Injektion) | UV3 |

## Qualitäts-Metriken (§9)

| §Ref | Bedeutung | Dateien |
|---|---|---|
| `§9.5` | Quality-Tracking (Vorher/Nachher-Baseline) | UV3 |
| `§09.2` | Song-Goal-Targets (Era/Material/Studio-Mode) | UV3 |

## Vintage-Ästhetik (§V)

| §Ref | Bedeutung | Dateien |
|---|---|---|
| `§V19` | Noise-Texture-Detector (erhält Rausch-Charakter) | UV3 |
| `§V24` | Tilt-Cap: Spektrale Balance bewahren | UV3 |
| `§V38` | Soft-Saturation-Guard (Röhren/Tape-Charakter) | UV3 |
| `§V40` | Wärmeband-Guard (200-800 Hz) | UV3 |
| `§V41` | Referenz-Konsistenz (kein Oversmoothing) | UV3 |

## Subsysteme

| §Ref | Bedeutung | Dateien |
|---|---|---|
| `§AC` | Intelligent Phase Pruning | PP |
| `§AF` | Cross-Guard (Denker-Teamwork) | UV3 |
| `§SFT` | Safety-Session-Tracker | UV3 |
| `§UQ` | Uncertainty-Quantification (Pipeline-UQ) | UV3 |
| `§SLR-1` | Lyrics-Guided-Enhancement | UV3 |
| `§CHT-1` | Cumulative-Hallucination-Tracker | UV3 |
| `§PID` | Phase-Interaction-Denker-Plan | UV3, PID |
| `§CSTC` | Cross-Segment-Timbral-Coherence | UV3 |

---

## Legende der Datei-Abkürzungen

| Kürzel | Datei |
|---|---|
| UV3 | `backend/core/unified_restorer_v3.py` |
| MD | `forensics/medium_detector.py` |
| DS | `backend/core/defect_scanner.py` |
| PMGG | `backend/core/per_phase_musical_goals_gate.py` |
| SGI | `backend/core/song_goal_importance.py` |
| PP | `backend/core/phase_pruner.py` |
| CV | `backend/core/defect_contract_validator.py` |
| DM | `backend/core/defect_manifest.py` |
| SP | `backend/core/safe_dict.py` |
| VFA | `backend/core/vocal_focus_analyzer.py` |
| PID | `denker/phase_interaction_denker.py` |

---

## Wie neue §-Referenzen hinzufügen

1. Im Code: `# §2.XX Beschreibung`
2. In diesem Dokument: Eintrag unter der passenden Kategorie (Konvention, kein Gate)

Hinweis: `scripts/compliance/check_spec_refs.py` existiert, ist aber **nicht** im
Pre-Commit verdrahtet. Beachte die ID-Zitierdisziplin aus `AGENTS.md` §5:
immer mit Fundstelle zitieren, nie nackte IDs.
