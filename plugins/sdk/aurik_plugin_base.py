"""AurikPlugin — Abstrakte Basisklasse für das Plugin-System.

§15.6: Plugin-SDK für Drittentwickler. Jedes Aurik-Plugin erbt von dieser
ABC und implementiert die erforderlichen Methoden.

Usage::

    from plugins.sdk.aurik_plugin_base import AurikPlugin, PluginManifest

    class MeinPlugin(AurikPlugin):
        manifest = PluginManifest(
            name="mein-plugin",
            version="1.0.0",
            description="Mein erstes Aurik-Plugin",
        )

        def process_audio(self, audio, sr, **kwargs):
            # Audio-Verarbeitung hier
            return audio

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PluginManifest:
    """Plugin-Metadaten (manifest.json-kompatibel).

    Attributes:
        name:            Eindeutiger Plugin-Name (lowercase, kebab-case).
        version:         SemVer-Version (z.B. "1.0.0").
        description:     Kurzbeschreibung (1-2 Sätze).
        author:          Autor/Organisation.
        license:         SPDX-Lizenz-Identifier (z.B. "MIT").
        min_aurik_version: Minimale Aurik-Version.
        max_aurik_version: Maximale Aurik-Version (None = unbeschränkt).
        dependencies:    Python-Paket-Abhängigkeiten.
        tags:            Schlagwörter für Plugin-Registry.
    """

    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = "MIT"
    min_aurik_version: str = "10.0.0"
    max_aurik_version: str | None = None
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialisierung für manifest.json."""
        result = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "min_aurik_version": self.min_aurik_version,
            "dependencies": self.dependencies,
            "tags": self.tags,
        }
        if self.max_aurik_version:
            result["max_aurik_version"] = self.max_aurik_version
        return result


@dataclass
class PluginResult:
    """Ergebnis einer Plugin-Verarbeitung.

    Attributes:
        audio:          Verarbeitetes Audio (np.ndarray).
        sample_rate:    Abtastrate.
        warnings:       Warnungen während der Verarbeitung.
        metrics:        Optionale Qualitäts-Metriken.
        success:        True wenn Verarbeitung erfolgreich.
        error:          Fehlermeldung wenn success=False.
    """

    audio: np.ndarray  # type: ignore[name-defined]
    sample_rate: int
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: str = ""


class AurikPlugin(ABC):
    """Abstrakte Basisklasse für alle Aurik-Plugins.

    Subclass-Mindestanforderungen:
        - ``manifest`` als Klassenattribut (PluginManifest)
        - ``process_audio()`` implementieren

    Optionale Hooks:
        - ``on_phase_start(audio, sr, phase_name, **kwargs)``
        - ``on_phase_end(audio, sr, phase_name, **kwargs)``
        - ``validate()`` — Selbsttest vor Aktivierung
    """

    manifest: PluginManifest

    @abstractmethod
    def process_audio(
        self,
        audio: np.ndarray,  # type: ignore[name-defined]
        sr: int = 48000,
        **kwargs,
    ) -> np.ndarray:  # type: ignore[name-defined]
        """Audio-Verarbeitung — Hauptmethode.

        Args:
            audio:  Eingabe-Audio (float32, shape=(samples,) oder (samples, channels)).
            sr:     Abtastrate in Hz.
            **kwargs: Zusätzliche Parameter (material, defects, etc.).

        Returns:
            Verarbeitetes Audio (gleiche Shape wie Eingabe).
        """
        ...

    @abstractmethod
    def on_phase_start(
        self,
        audio: np.ndarray,  # type: ignore[name-defined]
        sr: int = 48000,
        phase_name: str = "",
        **kwargs,
    ) -> None:
        """Hook: Wird VOR einer Phase aufgerufen (optional)."""

    @abstractmethod
    def on_phase_end(
        self,
        audio: np.ndarray,  # type: ignore[name-defined]
        sr: int = 48000,
        phase_name: str = "",
        **kwargs,
    ) -> None:
        """Hook: Wird NACH einer Phase aufgerufen (optional)."""

    def validate(self) -> tuple[bool, str]:
        """Selbsttest: Prüft ob Plugin korrekt konfiguriert ist.

        Returns:
            (ok, message) — True wenn Plugin bereit, sonst Fehlermeldung.
        """
        if not self.manifest:
            return False, "Kein PluginManifest definiert"
        if not self.manifest.name:
            return False, "Plugin-Name fehlt"
        if not self.manifest.version:
            return False, "Plugin-Version fehlt"
        return True, "OK"

    def get_manifest(self) -> PluginManifest:
        """Gibt das Plugin-Manifest zurück."""
        return self.manifest

    def safe_process(
        self,
        audio: np.ndarray,  # type: ignore[name-defined]
        sr: int = 48000,
        **kwargs,
    ) -> PluginResult:
        """Sichere Verarbeitung mit Fehlerbehandlung.

        Args:
            audio:  Eingabe-Audio.
            sr:     Abtastrate.
            **kwargs: Zusätzliche Parameter.

        Returns:
            PluginResult mit audio, warnings und success-Flag.
        """
        import traceback

        try:
            result_audio = self.process_audio(audio, sr, **kwargs)
            return PluginResult(
                audio=result_audio,
                sample_rate=sr,
                success=True,
            )
        except Exception as exc:
            return PluginResult(
                audio=audio,  # unverändert zurückgeben
                sample_rate=sr,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                warnings=[traceback.format_exc()[-500:]],
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.manifest.name!r} v{self.manifest.version}>"


__all__ = [
    "AurikPlugin",
    "PluginManifest",
    "PluginResult",
]
