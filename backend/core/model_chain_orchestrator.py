"""
backend/core/model_chain_orchestrator.py — ML-Model-Chain-Orchestrator (§v10.9)
================================================================================

Zentrale Instanz für shared ML-Modelle. Verhindert doppeltes Laden (z.B.
DeepFilterNet wird von Phase 03 UND MRN-Plugins separat geladen = ~500 MB RAM).
Lädt jedes Modell einmal und teilt die Instanz zwischen allen Konsumenten.

Usage:
    from backend.core.model_chain_orchestrator import get_model_chain
    mco = get_model_chain()
    df = mco.get("deepfilternet")  # shared instance
    bw = mco.get("bw_reconstructor")
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# RAM-Budget-Grenzen (§v10.9)
_MIN_FREE_RAM_GB_FOR_ML: float = 2.0  # Minimum freies RAM vor ML-Load
_MAX_TOTAL_ML_RAM_GB: float = 6.0  # Maximal insgesamt für ML-Modelle


def _get_available_ram_gb() -> float:
    """Ermittelt verfügbares RAM in GB (cross-platform)."""
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024**3)
    except Exception:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
        return 8.0  # Konservativer Fallback


class ModelChainOrchestrator:
    """Zentraler Model-Loader mit RAM-Budget und Shared-Instance-Pool."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._instances: dict[str, Any] = {}
        self._ram_used_gb: float = 0.0
        self._model_sizes_gb: dict[str, float] = {
            "deepfilternet": 0.25,
            "banquet_vinyl": 0.09,
            "bw_reconstructor": 0.005,
            "sgmse": 0.50,
            "bs_roformer": 0.86,
            "melband_roformer": 0.86,
            "demucs": 0.30,
            "nvsr": 0.001,
            "rmvpe": 0.35,
            "beats": 0.35,
            "clap": 0.40,
        }

    def get(self, model_name: str) -> Any | None:
        """Gibt shared Model-Instanz zurück oder None wenn RAM nicht reicht.

        Args:
            model_name: Eindeutiger Modell-Name (z.B. 'deepfilternet').

        Returns:
            Model-Instanz oder None wenn Laden nicht möglich.
        """
        with self._lock:
            if model_name in self._instances:
                return self._instances[model_name]

        _avail_gb = _get_available_ram_gb()
        _size_gb = self._model_sizes_gb.get(model_name, 0.5)

        if _avail_gb < _MIN_FREE_RAM_GB_FOR_ML:
            logger.warning(
                "⚠️ MCO: %s nicht geladen — nur %.1f GB RAM frei (Minimum: %.1f GB)",
                model_name,
                _avail_gb,
                _MIN_FREE_RAM_GB_FOR_ML,
            )
            return None

        if self._ram_used_gb + _size_gb > _MAX_TOTAL_ML_RAM_GB:
            logger.warning(
                "⚠️ MCO: %s nicht geladen — ML-Grenze erschöpft (%.1f/%.1f GB)",
                model_name,
                self._ram_used_gb,
                _MAX_TOTAL_ML_RAM_GB,
            )
            return None

        instance = self._load_model(model_name)
        if instance is not None:
            with self._lock:
                self._instances[model_name] = instance
                self._ram_used_gb += _size_gb
            logger.info(
                "✅ MCO: %s geladen (%.2f GB) — total ML: %.1f/%.1f GB",
                model_name,
                _size_gb,
                self._ram_used_gb,
                _MAX_TOTAL_ML_RAM_GB,
            )
        return instance

    def _load_model(self, model_name: str) -> Any | None:
        """Lädt ein spezifisches ML-Modell (nur einmal)."""
        try:
            if model_name == "deepfilternet":
                from plugins.deepfilternet_v3_ii_plugin import enhance_audio

                return enhance_audio
            elif model_name == "banquet_vinyl":
                from plugins.banquet_vinyl_plugin import BanquetVinylPlugin

                return BanquetVinylPlugin()
            elif model_name == "bw_reconstructor":
                from plugins.bw_reconstructor_plugin import BWReconstructorPlugin

                bw = BWReconstructorPlugin()
                return bw if bw.available else None
            elif model_name == "sgmse":
                from plugins.sgmse_plugin import SGMSEPlusPlugin

                return SGMSEPlusPlugin()
            elif model_name == "demucs":
                from plugins.demucs_v4_plugin import DemucsV4Plugin

                return DemucsV4Plugin()
            else:
                logger.debug("MCO: unknown model %s", model_name)
                return None
        except Exception as exc:
            logger.warning("⚠️ MCO: %s Laden fehlgeschlagen: %s", model_name, exc)
            return None

    def evict(self, model_name: str) -> None:
        """Entlädt ein Modell und gibt RAM frei."""
        with self._lock:
            if model_name in self._instances:
                _size = self._model_sizes_gb.get(model_name, 0.5)
                del self._instances[model_name]
                self._ram_used_gb = max(0.0, self._ram_used_gb - _size)
                logger.info("🗑️ MCO: %s evicted — ML RAM: %.1f GB", model_name, self._ram_used_gb)

    @property
    def ram_used_gb(self) -> float:
        return self._ram_used_gb

    @property
    def loaded_models(self) -> list[str]:
        with self._lock:
            return list(self._instances.keys())


# Singleton
_mco_instance: ModelChainOrchestrator | None = None
_mco_lock = threading.Lock()


def get_model_chain() -> ModelChainOrchestrator:
    """Gibt die Singleton-Instanz des ModelChainOrchestrator zurück."""
    global _mco_instance
    if _mco_instance is None:
        with _mco_lock:
            if _mco_instance is None:
                _mco_instance = ModelChainOrchestrator()
    return _mco_instance
