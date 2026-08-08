# Aurik 10 — Architektur

> Stand: 10.0.20 | Alle neuen Module aus Spec 18, 22, 15, 03, 11, 13, 14

## Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                     GUI / CLI / Batch                        │
│              Aurik10/ui/  ·  cli/  ·  workflow/              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Bridge-API (§V4)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  backend/api/bridge.py                       │
│   get_presence_embedding()  get_era_completion()             │
│   get_rollback_sanity()     get_preview_mode()               │
│   get_artist_fingerprint()  get_model_downloader()           │
│   get_ml_device_manager()   get_restorer()                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              unified_restorer_v3.py (42000+ Zeilen)          │
│   _execute_pipeline() → 69 Phasen → Post-Processing         │
│                                                              │
│   §G90 PresenceEmbedding  ← vor Export                       │
│   §G90 EraCompletion      ← BW < 10 kHz                     │
│   §G91 GddBudgetManager   ← proaktive STFT-Drosselung        │
│   §G92 RollbackSanityCheck ← nach Rollback                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌──────────┐   ┌──────────┐   ┌──────────────┐
   │ 69 Phasen │   │  Denker  │   │  Spec-Module │
   │ phase_*   │   │ denker/  │   │ backend/core │
   └──────────┘   └──────────┘   └──────────────┘
```

## Schichten

| Schicht | Verzeichnis | Zugriff |
|---------|------------|---------|
| **UI** | `Aurik10/ui/`, `cli/` | Nur via `bridge.py` (§V4) |
| **Bridge** | `backend/api/bridge.py` | Alle Exporte, Lazy-Import |
| **Pipeline** | `backend/core/unified_restorer_v3.py` | Orchestrierung |
| **Phasen** | `backend/core/phases/` | 69 Phasen via `PhaseInterface` |
| **Denker** | `denker/` | Strategie, Cross-Phase |
| **DSP** | `backend/core/dsp/`, `dsp/` | Signalverarbeitung |
| **ML** | `backend/ml/`, `plugins/` | ONNX-Modelle |

## Neue Module (10.0.18–10.0.20)

| Modul | Spec | Funktion |
|-------|------|----------|
| `presence_embedding.py` | §G90 | 5-Dimensionen-Präsenz-Metrik |
| `era_authentic_completion.py` | Spec 03 | Ära-BW-Erweiterung |
| `rollback_sanity_check.py` | §G92 | Stille/NaN nach Rollback |
| `ab_comparison.py` | Spec 14 | A/B-Vergleich + Blindtest |
| `artist_fingerprint.py` | Spec 13 | Künstler/Track-Modelle |
| `preview_mode.py` | Spec 11 | 30s-Preview |
| `ml/batch_processor.py` | Spec 15 | Batch + Session-Recycling |
| `audio_validator.py` | Spec 08 | MAX_AUDIO_BYTES_RAM |
| `dsp/powr_dither.py` | §V5 | POW-r Type 3 Dither |
| `core/expert_mode.py` | v10.206 | Experten-Modus |
| `core/session_memory.py` | v10.206 | Session-Gedächtnis |
| `core/result_enrichment.py` | v10.206 | RT-Faktor, Phase-Report |
| `core/spectrum_comparison.py` | v10.206 | Spektrum-Vergleich |
| `core/batch_overview.py` | v10.206 | Batch-Übersicht |
| `core/defect_map.py` | v10.206 | Defekt-Karte |
