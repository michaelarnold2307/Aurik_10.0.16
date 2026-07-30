# Session Insights — 2026-07-28

## Phase 0 ML-Pre-Processor Architektur & Elke-Best-Lauf-Analyse

### Gelöste Probleme

| Problem | Root Cause | Fix | Spec |
|---|---|---|---|
| 8/15 Goal-Verletzungen + End-Gate-Death-Spiral + Goosebumps=Original | Goal-Baseline gegen degradiertes Original → Schwellwerte unerreichbar | Goal-Budget nach Phase 0 neu kalibriert | §v10.303.17 |
| Phase 40: 528s mit 12% Wirkung | Conductor ×0.35 × SongCal ×0.12 auf präzise LUFS-Messung | Precision-Phases umgehen Drossel-Kaskade | §v10.303.16 |
| De-Esser: Elke Best als "male" klassifiziert | `bandwidth_loss` nicht an Gender-Detector übergeben → F2-Degradation unerkannt | bw_loss-Parameter + `_strong_contralto_signal` | §v10.303.11 |
| CIG-Rollback nach Phase 29 (group delay 38.9ms) | Tape-Hiss-Phase für MP3-Material aktiviert → 3 STFT-Phasen kaskadiert | Phase 29 nur für echte Tape-Materialien | §v10.303.14 |
| TQC: "cassette" span=1.823 > 1.80 → FAIL | Multi-Carrier-Kette (3 analog + 1 digital) braucht permissivere Schwellen | transfer_chain → max aller Carrier | §v10.303.15 |
| Phase 0 lief isoliert ohne Pipeline-Kontext | Keine Kommunikation mit DefectScanner/PMGG/Goal-Budget | Phase-0 → Pipeline-Context-Sync | §v10.303.17 |

### Neue Dateien

- `.github/specs/v10.303.17_phase0_architecture.md` — Master-Spec
- `plugins/apollo_phase0_integration.py` — 3-Stufen-Pre-Processor
- `backend/tests/apollo_hallucination_test.py` — Standalone-Test

### Geänderte Dateien

- `plugins/apollo_plugin.py` — Hallucination-Guard (§2.46e)
- `backend/core/unified_restorer_v3.py` — Phase-0-Integration + Precision-Phases + Tape-Hiss-Guard + Goal-Recalibration + Phase-0-Aware-Skips
- `backend/core/phases/phase_19_de_esser.py` — Gender-Detection MP3-resistent
- `backend/core/temporal_quality_coherence.py` — Multi-Carrier-Thresholds
- `docs/PROJECT_STATUS.md` — Version 10.0.10, Phase 0 dokumentiert
- `.github/specs/20_erkenntnisse_maximale_restaurierung.md` — 7 neue Erkenntnisse
