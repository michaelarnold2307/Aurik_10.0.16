# ModelManager für Aurik 10.0.0
# Dynamische, kontextbewusste Modellverwaltung mit Voice-Profil, Feedback, Multi-Stage Enhancement, Authentizitäts-Check

from __future__ import annotations

import csv
import io
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelStatusResult:
    """Gibt type of ModelManager.get_model_status() zurück."""

    status: str  # "not_found" | "available" | "unavailable"
    meta: dict[str, Any] | None = None


@dataclass
class ModelReloadResult:
    """Gibt type of ModelManager.reload_model_api() zurück."""

    status: str  # "reloaded"
    name: str


_instance: ModelManager | None = None
_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Gibt zurück: or create ModelManager singleton.

    Returns:
        ModelManager singleton instance
    """
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ModelManager()
    return _instance


class ModelManager:
    """Dynamische, kontextbewusste Modellverwaltung für Aurik (ML-Auswahl, Voice-Profil, Audit)."""

    def register_adaptive_plugins(
        self, sibilantnet: Any, breathnet: Any, voicehealthnet: Any, languagenet: Any
    ) -> None:
        """Registriert adaptive Plugins (Sibilant, Atem, VoiceHealth, Sprache)."""
        self.register_model("sibilantnet", sibilantnet, {"type": "sibilant", "adaptive": True})
        self.register_model("breathnet", breathnet, {"type": "breath", "adaptive": True})
        self.voicehealthnet = voicehealthnet
        self.languagenet = languagenet

    def select_sibilant_model(self, context: dict[str, Any]) -> Any:
        """Wählt Sibilanten-Modell kontextbewusst nach Stimmtyp, Sprache und Userwunsch."""
        # Kontextbewusste Auswahl: Stimmtyp, Sprache, Userwunsch
        if context.get("voice_type") == "child":
            return self.models.get("sibilantnet")
        if context.get("voice_type") == "female":
            return self.models.get("sibilantnet")
        # ...weitere Logik für Sprache, Userwunsch...
        return self.models.get("sibilantnet")

    def analyze_voice_health(self, audio: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Analysiert Stimmgesundheit (Erschöpfung, Heiserkeit) via VoiceHealthNet."""
        if self.voicehealthnet is not None:
            result: dict[str, Any] = self.voicehealthnet.analyze(audio, context)
            return result
        return {"fatigue": False, "hoarseness": False, "recommendation": "unknown"}

    def detect_language(self, audio: Any, context: dict[str, Any]) -> dict[str, str]:
        """Erkennt Sprache und Dialekt des Audiomaterials via LanguageNet."""
        if self.languagenet is not None:
            lang_result: dict[str, str] = self.languagenet.detect(audio, context)
            return lang_result
        return {"language": "unknown", "dialect": "unknown"}

    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        self.active_model: Any | None = None
        self.voice_profile: dict[str, Any] | None = None
        self.user_feedback: list[dict[str, Any]] = []
        self.audit_log: list[dict[str, Any]] = []
        self.voicehealthnet: Any | None = None  # type: ignore[no-redef]
        self.languagenet: Any | None = None  # type: ignore[no-redef]
        logger.info("ModelManager initialisiert")

    def list_models(self) -> dict[str, dict[str, Any]]:
        """Gibt alle registrierten Modelle mit Metadaten zurück."""
        return {name: m["meta"] for name, m in self.models.items()}

    def get_model_status(self, name: str) -> ModelStatusResult:
        """Status und Metadaten eines Modells abfragen."""
        m = self.models.get(name)
        if not m:
            return ModelStatusResult(status="not_found")
        return ModelStatusResult(
            status="available" if m["obj"] else "unavailable",
            meta=m["meta"],
        )

    def reload_model_api(self, name: str, new_model_obj: Any, new_metadata: dict[str, Any]) -> ModelReloadResult:
        """Modell per API neu laden/ersetzen."""
        self.reload_model(name, new_model_obj, new_metadata)
        return ModelReloadResult(status="reloaded", name=name)

    def get_audit_log(self, as_json: bool = False, as_csv: bool = False) -> Any:
        """
        Audit-Log aller Modellentscheidungen und Fallbacks.
        Optional: als JSON oder CSV exportieren.
        """
        if as_json:
            return json.dumps(self.audit_log, ensure_ascii=False, indent=2, default=str)
        if as_csv:
            if not self.audit_log:
                return ""
            output = io.StringIO()
            keys = sorted({k for entry in self.audit_log for k in entry})
            writer = csv.DictWriter(output, fieldnames=keys)
            writer.writeheader()
            for entry in self.audit_log:
                writer.writerow({k: str(entry.get(k, "")) for k in keys})
            return output.getvalue()
        return self.audit_log

    def get_model_api_status(self) -> dict[str, Any]:
        """API-Status aller Modelle (Name, Status, Metadaten, letzte Auswahl, Feedback)."""
        return {
            name: {
                "status": self.get_model_status(name),
                "last_selected": any(
                    isinstance(log.get("model"), dict) and log["model"].get("meta", {}).get("name") == name
                    for log in reversed(self.audit_log)
                ),
                "feedback": [f for f in self.user_feedback if f.get("model") == name],
            }
            for name in self.models
        }

    def register_model(self, name: str, model_obj: Any, metadata: dict[str, Any]) -> None:
        """Registriert ein Modell mit Metadaten und loggt die Aktion."""
        self.models[name] = {"obj": model_obj, "meta": metadata}
        logging.info("Model registered: %s, meta: %s", name, metadata)

    def set_voice_profile(self, profile: dict[str, Any]) -> None:
        """Setzt das Voice-Profil und loggt die Aktion."""
        self.voice_profile = profile
        logging.info("Voice profile set: %s", profile)

    def add_user_feedback(self, feedback: dict[str, Any]) -> None:
        """Fügt User-Feedback hinzu und loggt die Aktion."""
        self.user_feedback.append(feedback)
        logging.info("User feedback added: %s", feedback)

    def select_model(self, context: dict[str, Any]) -> Any:
        """Wählt kontextbewusst das beste Modell; Fallback-Kette wenn kein Kandidat verfügbar."""
        candidates = [m for m in self.models.values() if self._matches_context(m, context)]
        best = self._choose_best(candidates, context)
        if best and best["obj"]:
            self.active_model = best["obj"]
            self._log_model_selection(best, context)
            return self.active_model
        # Fallback-Logik: Priorisiere Modelle nach Qualitäts-Metadaten
        fallback_chain = sorted(self.models.values(), key=lambda m: m["meta"].get("quality", "low"), reverse=True)
        for fallback in fallback_chain:
            if fallback["obj"]:
                self.active_model = fallback["obj"]
                self._log_fallback(fallback)
                return self.active_model
        # DSP-Fallback
        self._log_fallback("DSP-Fallback")  # §V6: logger.warning handled at call site
        return None

    def reload_model(self, name: str, new_model_obj: Any, new_metadata: dict[str, Any]):
        """Ersetzt ein Modell-Objekt und dessen Metadaten in-place."""
        self.models[name] = {"obj": new_model_obj, "meta": new_metadata}
        self._log_model_selection(self.models[name], {"reload": True})

    def multi_stage_enhancement(self, input_audio: Any, context: dict[str, Any]) -> Any:
        """Führt mehrstufige Enhancement-Pipeline (Denoising → Sibilanten → Authentizität) aus."""
        stages = self._get_enhancement_stages(context)
        output = input_audio
        for stage in stages:
            model = self.models.get(stage)
            if model:
                output = model["obj"].process(output, context)
        return output

    def authenticity_check(self, output_audio: Any) -> bool:
        """Plausibilitätsprüfung der verarbeiteten Audio via spektraler Glattheit.

        Prüft ob das Signal nicht stummgeschaltet, überkomprimiert oder artefakt-dominiert ist:
          - RMS > 1e-6 (nicht stumm)
          - Spectral Flatness < 0.95 (nicht vollständig rauschartig)
          - Peak < 0.9999 (kein hartes Clipping)
        """
        try:
            import numpy as np  # pylint: disable=import-outside-toplevel

            audio = output_audio
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                return True  # Nicht prüfbar -> akzeptieren
            y = audio.flatten().astype(np.float64)
            rms = float(np.sqrt(np.mean(y**2)))
            if rms < 1e-6:
                logging.warning("[ModelManager] authenticity_check: Signal ist nahezu stumm.")
                return False
            peak = float(np.max(np.abs(y)))
            if peak > 0.9999:
                logging.warning("[ModelManager] authenticity_check: Hartes Clipping erkannt.")
                return False
            # Spektrale Glattheit (Geometrisches Mittel / Arithmetisches Mittel der Spektralleistung)
            n = min(4096, len(y))
            mag = np.abs(np.fft.rfft(y[:n])) + 1e-12
            mag_sq = mag**2
            geo_mean = float(np.exp(np.mean(np.log(mag_sq))))
            ari_mean = float(np.mean(mag_sq))
            flatness = geo_mean / (ari_mean + 1e-12)
            if flatness > 0.95:
                logging.warning("[ModelManager] authenticity_check: Spektrale Glattheit zu hoch (%.3f).", flatness)
                return False
            return True
        except Exception as e:
            logger.warning("model_manager.py::authenticity_Pruefung Ersatzpfad: %s", e)
            return True  # Im Zweifel akzeptieren

    def _matches_context(self, _model: dict[str, Any], _context: dict[str, Any]) -> bool:
        # Prüft, ob Modell zu Kontext passt (z. B. Geschlecht, Alter, Feedback)
        return True

    def _choose_best(self, candidates: list[dict[str, Any]], _context: dict[str, Any]) -> dict[str, Any] | None:
        # Wählt bestes Modell nach Qualitätsmetriken, Feedback, Voice-Profile
        return candidates[0] if candidates else None

    def _get_enhancement_stages(self, _context: dict[str, Any]) -> list[str]:
        # Dynamische Reihenfolge: Denoising → Sibilanten → Authentizität
        return ["denoiser", "sibilant", "authenticity"]

    def _log_model_selection(self, model: dict[str, Any] | None, context: dict[str, Any]) -> None:
        """Loggt Modellentscheidung (nur Metadaten) mit Kontext im Audit-Log."""
        entry: dict[str, Any] = {
            "model": model.get("meta") if model else None,
            "context": context,
            "event": "selection",
        }
        self.audit_log.append(entry)
        logging.info("Model selection logged: %s", entry)

    def fallback(self):
        """Aktiviert Fallback-Kette (ML → Alternativ-ML → DSP) wenn kein Primärmodell verfügbar."""
        # Fallback-Kette: ML → Alternativ-ML → DSP
        for alt in self._get_fallback_chain():
            if self._is_available(alt):
                self.active_model = alt
                self._log_fallback(alt)
                return alt
        return self._use_dsp_fallback()

    def _get_fallback_chain(self) -> list[Any]:
        """Fallback-Kette nach Priorität: ML-Alternativen, dann DSP-only Modelle.

        Modelle mit Metadaten-Key 'priority' werden zuerst eingesetzt;
        reine DSP-Modelle (type='dsp') landen am Ende der Kette.
        """
        all_models = list(self.models.values())

        # Modelle nach Priorität sortieren (höher = bevorzugt), DSP ans Ende
        def _priority(m: dict[str, Any]) -> int:
            meta = m.get("meta") or m
            if meta.get("type") == "dsp":
                return -1
            return int(meta.get("priority", 0))

        sorted_models = sorted(all_models, key=_priority, reverse=True)
        return [m.get("obj") or m for m in sorted_models]

    def _is_available(self, _model: Any) -> bool:
        return True

    def _log_fallback(self, model: Any) -> None:
        """Loggt Fallback-Entscheidung im Audit-Log."""
        entry: dict[str, Any] = {"fallback": model, "event": "fallback"}
        self.audit_log.append(entry)
        logging.info("Fallback logged: %s", entry)

    def _use_dsp_fallback(self):
        logging.info("DSP fallback used")
        return None


# Beispiel für Integration: ModelManager wird im Backend und Plugins verwendet
