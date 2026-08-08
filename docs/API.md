# Aurik 10 — Bridge API Reference

> Stand: 10.0.20 | Alle öffentlichen Bridge-API-Funktionen

## Import

```python
from backend.api.bridge import (
    get_restorer, get_presence_embedding, get_era_completion,
    get_rollback_sanity_guard, get_preview_mode,
    get_artist_fingerprint_store, get_model_downloader,
    get_ml_device_manager, get_expert_mode, get_session_memory,
)
```

## Pipeline & Restaurierung

| Funktion | Rückgabe | Spec |
|----------|----------|------|
| `get_restorer()` | `UnifiedRestorerV3` | Haupt-Pipeline |
| `get_restorer_classes()` | `(RestorationConfig, UnifiedRestorerV3)` | Klassen-Tuple |
| `get_aurik_denker_class()` | `AurikDenker` | 8-Stufen-Denker |

## Qualität & Wahrnehmung (§G90)

| Funktion | Rückgabe | Beschreibung |
|----------|----------|-------------|
| `get_presence_embedding()` | `PresenceEmbedding` | 5-Dimensionen-Präsenz-Metrik |
| `get_era_completion()` | `EraAuthenticPerceptualCompletion` | Ära-BW-Erweiterung |
| `get_rollback_sanity_guard()` | `RollbackSanityGuard` | Stille/NaN nach Rollback |

## Vorschau & Vergleich

| Funktion | Rückgabe | Beschreibung |
|----------|----------|-------------|
| `get_preview_mode()` | `PreviewMode` | 30s-Preview |
| `get_ab_comparison()` | `ABComparison` | A/B-Vergleich |

## Künstler & Session

| Funktion | Rückgabe | Beschreibung |
|----------|----------|-------------|
| `get_artist_fingerprint_store()` | `ArtistFingerprintStore` | Persistente Stimm/Track-Modelle |
| `get_session_memory()` | `SessionMemory` | History + Fenster-Position |

## ML & GPU

| Funktion | Rückgabe | Beschreibung |
|----------|----------|-------------|
| `get_model_downloader()` | `ModelDownloader` | SOTA-Modell-Downloads |
| `get_ml_device_manager()` | `MLDeviceManager` | GPU/CPU-Backend |

## GUI-Features

| Funktion | Rückgabe | Beschreibung |
|----------|----------|-------------|
| `get_expert_mode()` | `ExpertMode` | Experten-Modus-Toggle |

## Export

Alle Exporte laufen zentral über `UnifiedRestorerV3._execute_pipeline()` → `audio_exporter.py`:

```
Pipeline-Ende:
  → PresenceEmbedding (5-Dimensionen)
  → EraCompletion (wenn BW < 10kHz)
  → CD-Rauschprofil (§G4)
  → POW-r Type 3 Dither (§V5)
  → Atomic Write (.tmp → os.replace)
```
