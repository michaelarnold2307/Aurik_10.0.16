"""
Aurik 6.0 – SOTA-konformer Import-Workflow
Initialisiert PolicyManager, FeatureExtractor und Quality-Gates direkt beim Import.
Auditierbar, modular, musikalisch fokussiert.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from backend.core.core_utils import log_message
from backend.core.forensics.analysis_and_modules import FeatureExtractor, PolicyManager

logger = logging.getLogger(__name__)


_instance: ImportPipeline | None = None
_lock = threading.Lock()


def get_import_pipeline(policy_template: dict[str, Any] | None = None) -> ImportPipeline:
    """Gibt zurück: or create ImportPipeline singleton.

    Args:
        policy_template: Policy template dict (only used on first call)

    Returns:
        ImportPipeline singleton instance
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ImportPipeline(policy_template)
    return _instance


class ImportPipeline:
    def __init__(self, policy_template: dict[str, Any] | None = None) -> None:
        self.policy = policy_template or {}
        self.policy_manager = PolicyManager(self.policy)
        self.extractor = FeatureExtractor()
        self.audit_log: list[dict[str, Any]] = []
        logger.info("ImportPipeline initialisiert")

    def import_audio(
        self, audio: np.ndarray, sr: int, reference: np.ndarray | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Schritt 1: Feature-Extraktion und Quality-Gates
        features = self.extractor.extract(audio, sr, reference, policy_manager=self.policy_manager)
        self.audit_log.append({"step": "feature_extraction", "features": features})
        log_message(f"Import: Features extrahiert {features}")
        # Schritt 2: Policy-Update und Audit
        self.policy_manager.update(features.get("quality_gates", {}))
        self.audit_log.append({"step": "policy_update", "policy": self.policy_manager.policy})
        log_message(f"Import: PolicyManager aktualisiert {self.policy_manager.policy}")
        # Schritt 3: Audit-Log speichern
        with open("logs/import_audit_log.ndjson", "a") as f:
            for entry in self.audit_log:
                f.write(str(entry) + "\n")
        return features, self.policy_manager.policy


# Beispiel für die Initialisierung und Nutzung:
# pipeline = ImportPipeline(policy_template={})
# features, policy = pipeline.import_audio(audio_array, sample_rate)
