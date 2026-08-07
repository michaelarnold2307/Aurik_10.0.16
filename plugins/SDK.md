# Aurik Plugin-SDK — Entwicklerhandbuch

> **Version:** 10.14.0 | **Stand:** August 2026

Aurik-Plugins erweitern die Restaurierungs-Pipeline um eigene Phasen,
ML-Modelle oder DSP-Algorithmen — ohne Änderungen am Core.

## Quick Start

```bash
# Neues Plugin aus Template erstellen
cp -r plugins/sdk/example_plugin plugins/my_plugin
cd plugins/my_plugin

# manifest.json anpassen (Name, Version, Beschreibung)
# plugin.py implementieren (AurikPlugin.process())
# Test ausführen
python ../../scripts/validate_plugin.py .

# Aktivieren: Aurik startet → Plugin wird automatisch erkannt
```

## Plugin-Struktur

```
plugins/my_plugin/
├── manifest.json     # Pflicht: Metadaten
├── plugin.py         # Pflicht: AurikPlugin-Implementierung
├── requirements.txt  # Optional: Python-Abhängigkeiten
└── README.md         # Optional: Dokumentation
```

## manifest.json

```json
{
  "name": "my_plugin",
  "version": "1.0.0",
  "description": "Meine benutzerdefinierte DSP-Phase",
  "author": "Max Mustermann",
  "entry_point": "plugin.py",
  "dependencies": ["numpy>=1.26"],
  "aurik_version_min": "10.14.0",
  "category": "dsp_phase"
}
```

| Feld | Pflicht | Beschreibung |
|------|:------:|-------------|
| `name` | ✅ | Eindeutiger Plugin-Name (nur Kleinbuchstaben, Unterstrich) |
| `version` | ✅ | SemVer (z.B. `1.0.0`) |
| `description` | ✅ | Kurzbeschreibung (1 Satz) |
| `author` | | Autor-Name |
| `entry_point` | ✅ | Python-Datei mit der Plugin-Klasse |
| `dependencies` | | Liste der pip-Abhängigkeiten |
| `aurik_version_min` | | Minimale Aurik-Version |
| `category` | | `dsp_phase`, `ml_model`, `export_filter`, `analysis_tool` |

## AurikPlugin-Base-Class

```python
# plugin.py
import numpy as np
from plugins.sdk.aurik_plugin_base import AurikPlugin


class MyPlugin(AurikPlugin):
    \"\"\"Meine benutzerdefinierte DSP-Phase.\"\"\"

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
        \"\"\"Hauptverarbeitung.

        Args:
            audio: Eingabe-Audio (float32, [-1.0, 1.0])
            sample_rate: Abtastrate in Hz
            **kwargs: Zusätzliche Parameter (material_type, strength, ...)

        Returns:
            Verarbeitetes Audio (gleiche Shape, float32)
        \"\"\"
        # Beispiel: Sanftes Tiefpass-Filter
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, 0.9 * sample_rate / 2, btype="low", fs=sample_rate, output="sos")
        return sosfiltfilt(sos, audio).astype(np.float32)

    def get_info(self) -> dict:
        \"\"\"Plugin-Metadaten für die Registry.\"\"\"
        return {
            "name": "my_plugin",
            "version": "1.0.0",
            "phase_name": "Mein Custom-Filter",
            "material_types": ["vinyl", "tape", "digital"],
            "default_strength": 0.5,
        }
```

## Validierung

```bash
# Einzelnes Plugin validieren
python scripts/validate_plugin.py plugins/my_plugin

# Alle Plugins validieren
python scripts/validate_plugin.py --all

# CI-Integration
python scripts/validate_plugin.py --all --ci  # Exit 1 bei Fehlern
```

## Plugin-Kategorien

| Kategorie | Beschreibung | Wann aufgerufen |
|-----------|-------------|-----------------|
| `dsp_phase` | DSP-Verarbeitungsphase | Während der Pipeline (Phase 01-64) |
| `ml_model` | ML-Modell (ONNX/PyTorch) | Vor/nach DSP-Phasen |
| `export_filter` | Export-Nachbearbeitung | Nach Pipeline, vor sf.write |
| `analysis_tool` | Analyse-Werkzeug | Pre-Analysis oder Debug |

## Best Practices

1. **Keine `print()`** — verwende `logging.getLogger(__name__)`
2. **NaN/Inf-Schutz** — `np.nan_to_num()` auf Ausgabe
3. **Float32-Garantie** — Ausgabe MUSS `dtype=np.float32` sein
4. **Stereo-Erhalt** — Stereo-Input → Stereo-Output (gleiche Shape)
5. **Strength-Parameter** — Unterstütze `strength=0.0` (Passthrough) bis `strength=1.0` (voll)
6. **Determinismus** — Gleicher Input + gleicher Seed → gleicher Output
