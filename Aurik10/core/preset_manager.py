"""
Preset-Manager für Aurik 10.

Speichert und verwaltet Restaurierungs-Presets lokal in
~/.aurik/presets/<name>.json (§11.4 der Aurik 10-Spezifikation).
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PresetCategory(Enum):
    """Kategorie eines Restaurierungs-Presets."""

    FACTORY = "Factory"
    USER = "User"
    IMPORTED = "Imported"


@dataclass
class Preset:
    """Einzelnes Restaurierungs-Preset."""

    name: str
    description: str
    category: PresetCategory
    params: dict[str, Any] = field(default_factory=dict)


class PresetManager:
    """Lädt, speichert und verwaltet Presets lokal.

    Persistenz: ~/.aurik/presets/<sanitized_name>.json
    Werksseitige Presets sind schreibgeschützt.
    """

    _FACTORY_PRESETS: list[Preset] = [
        Preset(
            "Standard-Restaurierung",
            "Konservative Restaurierung für alle Materialien — erhält den Original-Klang.",
            PresetCategory.FACTORY,
            {"mode": "RESTORATION"},
        ),
        Preset(
            "Highend Studio 2026",
            "Moderner Streaming-Sound mit maximaler Brillanz und Kraft.",
            PresetCategory.FACTORY,
            {"mode": "STUDIO_2026"},
        ),
        Preset(
            "Vinyl-Schallplatte",
            "Optimiert für Knistern, Verformungen und Rillenpops von LP/Single.",
            PresetCategory.FACTORY,
            {"mode": "RESTORATION", "material": "vinyl"},
        ),
        Preset(
            "Kassette / Tonband",
            "Optimiert für Rauschen, Aussetzer und Gleichlaufschwankungen von Magnetband.",
            PresetCategory.FACTORY,
            {"mode": "RESTORATION", "material": "tape"},
        ),
        Preset(
            "Schellack 78rpm",
            "Optimiert für schweres Breitrauschen und Bandbreite ≤ 8 kHz.",
            PresetCategory.FACTORY,
            {"mode": "RESTORATION", "material": "shellac"},
        ),
        Preset(
            "Kassette Deutsch Schlager",
            "Schlager-Profil: Akkordeon-Klangfarben bewahrt, Wärme betont, keine Tonart-Verschiebung.",
            PresetCategory.FACTORY,
            {"mode": "RESTORATION", "material": "tape", "genre": "schlager"},
        ),
    ]

    def __init__(self, presets_dir: pathlib.Path | None = None) -> None:
        self._dir = presets_dir or (pathlib.Path.home() / ".aurik" / "presets")
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Lesen ────────────────────────────────────────────────────────────────

    def get_all_presets(self) -> list[Preset]:
        """Gibt alle Presets zurück (werksseitig + Nutzer-eigene)."""
        presets: list[Preset] = list(self._FACTORY_PRESETS)
        for p in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                cat_str = data.get("category", "User")
                try:
                    cat = PresetCategory(cat_str)
                except ValueError:
                    cat = PresetCategory.USER
                presets.append(
                    Preset(
                        data["name"],
                        data.get("description", ""),
                        cat,
                        data.get("params", {}),
                    )
                )
            except Exception:
                logger.warning("preset_manager.py::get_all_presets Ersatzpfad", exc_info=True)
        return presets

    def get_preset(self, name: str) -> Preset | None:
        """Sucht ein Preset nach exaktem Namen."""
        for p in self.get_all_presets():
            if p.name == name:
                return p
        return None

    # ── Schreiben ─────────────────────────────────────────────────────────────

    def save_preset(self, preset: Preset) -> bool:
        """Speichert ein Nutzer-Preset als JSON. Gibt True bei Erfolg zurück."""
        safe = "".join(c if c.isalnum() or c in " ._-" else "_" for c in preset.name).strip() or "preset"
        path = self._dir / f"{safe}.json"
        try:
            path.write_text(
                json.dumps(
                    {
                        "name": preset.name,
                        "description": preset.description,
                        "category": preset.category.value,
                        "params": preset.params,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return True
        except OSError:
            return False

    def delete_preset(self, name: str) -> bool:
        """Löscht ein Nutzer-Preset. Factory-Presets können nicht gelöscht werden."""
        if any(fp.name == name for fp in self._FACTORY_PRESETS):
            return False
        safe = "".join(c if c.isalnum() or c in " ._-" else "_" for c in name).strip()
        path = self._dir / f"{safe}.json"
        try:
            if path.exists():
                path.unlink()
            return True
        except OSError:
            return False

    # ── Import / Export ───────────────────────────────────────────────────────

    def import_preset(self, path: pathlib.Path) -> Preset | None:
        """Importiert ein Preset aus einer JSON-Datei."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            preset = Preset(
                data["name"],
                data.get("description", ""),
                PresetCategory.IMPORTED,
                data.get("params", {}),
            )
            self.save_preset(preset)
            return preset
        except Exception:
            logger.warning("preset_manager.py::import_preset Ersatzpfad", exc_info=True)
            return None

    def export_preset(self, name: str, path: pathlib.Path) -> bool:
        """Exportiert ein Preset in eine JSON-Datei."""
        preset = self.get_preset(name)
        if preset is None:
            return False
        try:
            path.write_text(
                json.dumps(
                    {
                        "name": preset.name,
                        "description": preset.description,
                        "category": preset.category.value,
                        "params": preset.params,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return True
        except OSError:
            return False
