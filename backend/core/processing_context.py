"""
core/processing_context.py
Global Processing Context für Module Cooperation
================================================

Zentraler Shared State für alle AURIK Module.
Ermöglicht:
- Session-weite State Verwaltung
- Module können State lesen/schreiben
- Forensic Analysis, Processing History, Confidence Scores
- Thread-safe Operations
- Persistent Storage (optional)

Version: 1.0.0
Author: AURIK Team
Date: 10. Februar 2026
"""

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModuleState(Enum):
    """Status eines Moduls in der Verarbeitungskette."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingPhase(Enum):
    """Verarbeitungsphase in der Pipeline."""

    INITIALIZATION = "initialization"
    ANALYSIS = "analysis"
    FORENSICS = "forensics"
    PRE_PROCESSING = "pre_processing"
    RESTORATION = "restoration"
    ENHANCEMENT = "enhancement"
    POST_PROCESSING = "post_processing"
    FINALIZATION = "finalization"
    COMPLETED = "completed"


class ModuleType(Enum):
    """Module type for cooperation and over-processing tracking."""

    DEESSER = "deesser"
    COMPRESSOR = "compressor"
    LIMITER = "limiter"
    NOISE_REDUCTION = "noise_reduction"
    DEREVERB = "dereverb"
    DECLICKER = "declicker"
    DEHUMMER = "dehummer"
    EQUALIZER = "equalizer"
    ENHANCER = "enhancer"
    FORENSICS = "forensics"
    MASTERING = "mastering"
    RESTORATION = "restoration"
    ANALYSIS = "analysis"
    OTHER = "other"


@dataclass
class ModuleInfo:
    """Information über ein Modul in der Pipeline."""

    name: str
    state: ModuleState = ModuleState.NOT_STARTED
    module_type: ModuleType = ModuleType.OTHER
    start_time: float | None = None
    end_time: float | None = None
    duration_ms: float | None = None
    confidence: float | None = None
    strength: float | None = None  # Processing strength applied (0-1)
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dictionary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "module_type": self.module_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class SessionMetadata:
    """Metadaten einer Processing Session."""

    session_id: str
    start_time: float
    end_time: float | None = None
    sample_rate: int = 48000
    audio_duration_sec: float | None = None
    num_channels: int = 1
    processing_mode: str = "restoration"
    user_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dictionary."""
        return asdict(self)


class ProcessingContext:
    """
    Global Processing Context für AURIK Module.

    Features:
    - Session-weite State Verwaltung
    - Thread-safe Operations
    - Module State Tracking
    - Forensic Analysis Storage
    - Processing History
    - Confidence Scores
    - Event Listeners
    - Persistent Storage (optional)

    Usage:
        # Initialize context
        context = ProcessingContext(session_id="session_001")

        # Register module
        context.register_module("DCBlocker")

        # Update module state
        context.set_module_state("DCBlocker", ModuleState.IN_PROGRESS)

        # Store data
        context.set("forensic_analysis", analysis_result)

        # Retrieve data
        analysis = context.get("forensic_analysis")

        # Complete module
        context.complete_module("DCBlocker", confidence=0.95, metrics={"snr": 25.3})
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        session_id: str,
        sample_rate: int = 48000,
        processing_mode: str = "restoration",
        user_id: str | None = None,
        persistent_storage: bool = False,
        storage_path: Path | None = None,
    ):
        """
        Initialisiert Processing Context.

        Args:
            session_id: Unique session identifier
            sample_rate: Audio sample rate
            processing_mode: Processing mode (restoration, enhancement, etc.)
            user_id: Optional user identifier
            persistent_storage: Enable persistent storage
            storage_path: Path for persistent storage
        """
        self.session_id = session_id
        self.metadata = SessionMetadata(
            session_id=session_id,
            start_time=time.time(),
            sample_rate=sample_rate,
            processing_mode=processing_mode,
            user_id=user_id,
        )

        # Thread-safe state storage
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {}
        self._modules: dict[str, ModuleInfo] = {}
        self._phase = ProcessingPhase.INITIALIZATION
        self._event_listeners: dict[str, list[Callable]] = {}

        # Persistent storage
        self.persistent_storage = persistent_storage
        self.storage_path = storage_path or Path("./sessions")

        # Logger
        self.logger = logging.getLogger(__name__)
        logger.info("ProcessingContext initialisiert: %s", session_id)

    # === Core State Management ===

    def set(self, key: str, value: Any) -> None:
        """
        Setzt a value in the context.
        Thread-safe operation.

        Args:
            key: Key for the value
            value: Value to store
        """
        with self._lock:
            self._state[key] = value
            self._trigger_event("state_changed", {"key": key, "value": value})

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gibt zurück: a value from the context.
        Thread-safe operation.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value or default
        """
        with self._lock:
            return self._state.get(key, default)

    def has(self, key: str) -> bool:
        """Prüft if key exists in context."""
        with self._lock:
            return key in self._state

    def delete(self, key: str) -> None:
        """Delete a key from context."""
        with self._lock:
            if key in self._state:
                del self._state[key]
                self._trigger_event("state_changed", {"key": key, "deleted": True})

    def get_all(self) -> dict[str, Any]:
        """Gibt zurück: all state as dictionary (copy)."""
        with self._lock:
            return self._state.copy()

    # === Module Management ===

    def register_module(self, module_name: str, parameters: dict | None = None) -> None:
        """
        Registriert a module in the context.

        Args:
            module_name: Name of the module
            parameters: Optional module parameters
        """
        with self._lock:
            if module_name not in self._modules:
                self._modules[module_name] = ModuleInfo(name=module_name, parameters=parameters or {})
                self._trigger_event("module_registered", {"module": module_name})
                logger.debug("Module registered: %s", module_name)

    def set_module_state(self, module_name: str, state: ModuleState) -> None:
        """
        Setzt module state.

        Args:
            module_name: Name of the module
            state: New state
        """
        with self._lock:
            if module_name not in self._modules:
                self.register_module(module_name)

            module = self._modules[module_name]
            old_state = module.state
            module.state = state

            # Track timing
            if state == ModuleState.IN_PROGRESS:
                module.start_time = time.time()
            elif state in [ModuleState.COMPLETED, ModuleState.FAILED] and module.start_time:
                module.end_time = time.time()
                module.duration_ms = (module.end_time - module.start_time) * 1000

            self._trigger_event(
                "module_state_changed", {"module": module_name, "old_state": old_state, "new_state": state}
            )

    def complete_module(
        self,
        module_name: str,
        confidence: float | None = None,
        metrics: dict | None = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """
        Mark module as completed with optional metrics.

        Args:
            module_name: Name of the module
            confidence: Confidence score (0-1)
            metrics: Optional metrics dict
            errors: Optional list of errors
            warnings: Optional list of warnings
        """
        with self._lock:
            if module_name not in self._modules:
                self.register_module(module_name)

            module = self._modules[module_name]
            module.state = ModuleState.COMPLETED

            if confidence is not None:
                module.confidence = confidence
            if metrics:
                module.metrics.update(metrics)
            if errors:
                module.errors.extend(errors)
            if warnings:
                module.warnings.extend(warnings)

            # Calculate duration
            if module.start_time:
                module.end_time = time.time()
                module.duration_ms = (module.end_time - module.start_time) * 1000

            self._trigger_event(
                "module_completed", {"module": module_name, "confidence": confidence, "duration_ms": module.duration_ms}
            )

            self.logger.debug(
                f"Module completed: {module_name} (confidence: {confidence:.2f}, duration: {module.duration_ms:.1f}ms)"
                if confidence and module.duration_ms
                else f"Module completed: {module_name}"
            )

    def fail_module(self, module_name: str, error: str, metrics: dict | None = None) -> None:
        """
        Mark module as failed.

        Args:
            module_name: Name of the module
            error: Error message
            metrics: Optional metrics dict
        """
        with self._lock:
            if module_name not in self._modules:
                self.register_module(module_name)

            module = self._modules[module_name]
            module.state = ModuleState.FAILED
            module.errors.append(error)

            if metrics:
                module.metrics.update(metrics)

            # Calculate duration
            if module.start_time:
                module.end_time = time.time()
                module.duration_ms = (module.end_time - module.start_time) * 1000

            self._trigger_event("module_failed", {"module": module_name, "error": error})

            logger.error("Module fehlgeschlagen: %s - %s", module_name, error)

    def get_module_info(self, module_name: str) -> ModuleInfo | None:
        """
        Gibt zurück: module information.

        Args:
            module_name: Name of the module

        Returns:
            ModuleInfo or None
        """
        with self._lock:
            return self._modules.get(module_name)

    def get_all_modules(self) -> dict[str, ModuleInfo]:
        """Gibt zurück: all module information (copy)."""
        with self._lock:
            return self._modules.copy()

    def get_completed_modules(self) -> list[str]:
        """Gibt zurück: list of completed module names."""
        with self._lock:
            return [name for name, info in self._modules.items() if info.state == ModuleState.COMPLETED]

    def get_failed_modules(self) -> list[str]:
        """Gibt zurück: list of failed module names."""
        with self._lock:
            return [name for name, info in self._modules.items() if info.state == ModuleState.FAILED]

    # === Phase Management ===

    def set_phase(self, phase: ProcessingPhase) -> None:
        """
        Setzt current processing phase.

        Args:
            phase: New phase
        """
        with self._lock:
            old_phase = self._phase
            self._phase = phase
            self._trigger_event("phase_changed", {"old_phase": old_phase, "new_phase": phase})
            logger.info("Verarbeitungsschritt changed: %s → %s", old_phase.value, phase.value)

    def get_phase(self) -> ProcessingPhase:
        """Gibt zurück: current processing phase."""
        with self._lock:
            return self._phase

    # === Event System ===

    def add_listener(self, event_name: str, callback: Callable) -> None:
        """
        Fügt hinzu: event listener.

        Args:
            event_name: Name of the event
            callback: Callback function (receives event data dict)
        """
        with self._lock:
            if event_name not in self._event_listeners:
                self._event_listeners[event_name] = []
            self._event_listeners[event_name].append(callback)

    def remove_listener(self, event_name: str, callback: Callable) -> None:
        """
        Entfernt event listener.

        Args:
            event_name: Name of the event
            callback: Callback function to remove
        """
        with self._lock:
            if event_name in self._event_listeners and callback in self._event_listeners[event_name]:
                self._event_listeners[event_name].remove(callback)

    def _trigger_event(self, event_name: str, event_data: dict[str, Any]) -> None:
        """
        Löst aus: event (internal).

        Args:
            event_name: Name of the event
            event_data: Event data
        """
        if event_name in self._event_listeners:
            for callback in self._event_listeners[event_name]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error("Event listener error (%s): %s", event_name, e)

    # === Convenience Methods ===

    def set_forensic_analysis(self, analysis: Any) -> None:
        """Store forensic analysis result."""
        self.set("forensic_analysis", analysis)

    def get_forensic_analysis(self) -> Any | None:
        """Gibt zurück: forensic analysis result."""
        return self.get("forensic_analysis")

    def set_processing_chain(self, chain: Any) -> None:
        """Store processing chain."""
        self.set("processing_chain", chain)

    def get_processing_chain(self) -> Any | None:
        """Gibt zurück: processing chain."""
        return self.get("processing_chain")

    def set_audio_metadata(self, duration_sec: float, num_channels: int) -> None:
        """Setzt audio metadata."""
        with self._lock:
            self.metadata.audio_duration_sec = duration_sec
            self.metadata.num_channels = num_channels

    def add_tag(self, tag: str) -> None:
        """Fügt hinzu: a tag to the session."""
        with self._lock:
            if tag not in self.metadata.tags:
                self.metadata.tags.append(tag)

    # === Statistics & Reporting ===

    def get_statistics(self) -> dict[str, Any]:
        """
        Gibt zurück: processing statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_modules = len(self._modules)
            completed = len(self.get_completed_modules())
            failed = len(self.get_failed_modules())
            in_progress = len([m for m in self._modules.values() if m.state == ModuleState.IN_PROGRESS])

            # Calculate average confidence
            confidences = [m.confidence for m in self._modules.values() if m.confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None

            # Calculate total duration
            durations = [m.duration_ms for m in self._modules.values() if m.duration_ms is not None]
            total_duration_ms = sum(durations) if durations else None

            return {
                "session_id": self.session_id,
                "phase": self._phase.value,
                "total_modules": total_modules,
                "completed_modules": completed,
                "failed_modules": failed,
                "in_progress_modules": in_progress,
                "success_rate": completed / total_modules if total_modules > 0 else 0,
                "average_confidence": avg_confidence,
                "total_duration_ms": total_duration_ms,
                "has_forensic_analysis": self.has("forensic_analysis"),
                "has_processing_chain": self.has("processing_chain"),
            }

    def get_summary(self) -> dict[str, Any]:
        """
        Gibt zurück: comprehensive session summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            return {
                "metadata": self.metadata.to_dict(),
                "phase": self._phase.value,
                "statistics": self.get_statistics(),
                "modules": {name: info.to_dict() for name, info in self._modules.items()},
                "state_keys": list(self._state.keys()),
            }

    # === Persistence ===

    def save(self, path: Path | None = None) -> Path:
        """
        Speichert context to disk.

        Args:
            path: Optional custom path

        Returns:
            Path where context was saved
        """
        save_path = path or (self.storage_path / f"{self.session_id}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            summary = self.get_summary()

            with open(save_path, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info("Context gespeichert: %s", save_path)
            return save_path

    @classmethod
    def load(cls, path: Path) -> "ProcessingContext":
        """
        Lädt context from disk.

        Args:
            path: Path to load from

        Returns:
            Loaded ProcessingContext
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct context
        metadata = data["metadata"]
        context = cls(
            session_id=metadata["session_id"],
            sample_rate=metadata["sample_rate"],
            processing_mode=metadata["processing_mode"],
            user_id=metadata.get("user_id"),
        )

        # Restore metadata
        context.metadata = SessionMetadata(**metadata)

        # Restore phase
        context._phase = ProcessingPhase(data["phase"])

        # Restore modules
        for name, module_data in data["modules"].items():
            module_info = ModuleInfo(
                name=module_data["name"],
                state=ModuleState(module_data["state"]),
                start_time=module_data.get("start_time"),
                end_time=module_data.get("end_time"),
                duration_ms=module_data.get("duration_ms"),
                confidence=module_data.get("confidence"),
                parameters=module_data.get("parameters", {}),
                metrics=module_data.get("metrics", {}),
                errors=module_data.get("errors", []),
                warnings=module_data.get("warnings", []),
            )
            context._modules[name] = module_info

        logging.getLogger(__name__).info(f"Context loaded: {path}")
        return context

    # === Lifecycle ===

    def finalize(self) -> None:
        """Finalize session (mark as completed)."""
        with self._lock:
            self.metadata.end_time = time.time()
            self.set_phase(ProcessingPhase.COMPLETED)

            if self.persistent_storage:
                self.save()

            logger.info("Sitzung finalized: %s", self.session_id)

    # === Module Cooperation & Over-Processing Prevention ===

    def has_module_type_processed(self, module_type: ModuleType) -> bool:
        """
        Prüft if a module of this type has already been processed.

        Args:
            module_type: Module type to check

        Returns:
            True if module type was already processed
        """
        with self._lock:
            return any(
                m.module_type == module_type and m.state == ModuleState.COMPLETED for m in self._modules.values()
            )

    def get_processing_history(self, module_type: ModuleType) -> list[ModuleInfo]:
        """
        Gibt zurück: processing history for a specific module type.

        Args:
            module_type: Module type to query

        Returns:
            List of ModuleInfo for completed modules of this type
        """
        with self._lock:
            return [
                m for m in self._modules.values() if m.module_type == module_type and m.state == ModuleState.COMPLETED
            ]

    def get_accumulated_strength(self, module_type: ModuleType) -> float:
        """
        Gibt zurück: total accumulated processing strength for a module type.

        Args:
            module_type: Module type to query

        Returns:
            Sum of all strength values applied (0 if none)
        """
        with self._lock:
            history = self.get_processing_history(module_type)
            return sum(m.strength for m in history if m.strength is not None)

    def check_over_processing_risk(self, module_type: ModuleType, proposed_strength: float) -> dict[str, Any]:
        """
        Prüft if proposed processing would cause over-processing.

        Over-processing thresholds:
        - DEESSER: > 2.0 cumulative strength
        - COMPRESSOR: > 6.0 cumulative ratio
        - NOISE_REDUCTION: > 24 dB cumulative reduction
        - DEREVERB: > 0.8 cumulative strength
        - LIMITER: > 1 application

        Args:
            module_type: Module type to check
            proposed_strength: Proposed processing strength

        Returns:
            Dict with 'safe', 'current', 'proposed_total', 'threshold', 'recommendation'
        """
        with self._lock:
            current = self.get_accumulated_strength(module_type)
            proposed_total = current + proposed_strength

            # Define thresholds
            thresholds = {
                ModuleType.DEESSER: 2.0,
                ModuleType.COMPRESSOR: 6.0,
                ModuleType.NOISE_REDUCTION: 24.0,
                ModuleType.DEREVERB: 0.8,
                ModuleType.LIMITER: 1.0,
                ModuleType.EQUALIZER: 12.0,  # dB gain cumulative
                ModuleType.ENHANCER: 1.5,
            }

            threshold = thresholds.get(module_type, float("inf"))
            safe = proposed_total <= threshold

            # Generate recommendation
            if not safe:
                remaining = max(0, threshold - current)
                recommendation = (
                    f"OVER-PROCESSING RISK: {module_type.value} cumulative {current:.2f} + {proposed_strength:.2f} = {proposed_total:.2f} "
                    f"exceeds threshold {threshold:.2f}. Reduce strength to max {remaining:.2f}."
                )
            else:
                headroom = threshold - proposed_total
                recommendation = (
                    f"Safe: {module_type.value} {proposed_total:.2f}/{threshold:.2f}. Headroom: {headroom:.2f}."
                )

            return {
                "safe": safe,
                "current": current,
                "proposed_strength": proposed_strength,
                "proposed_total": proposed_total,
                "threshold": threshold,
                "headroom": threshold - proposed_total if safe else 0,
                "recommendation": recommendation,
            }

    def get_recommended_strength(
        self, module_type: ModuleType, base_strength: float, confidence: float, material_type: str | None = None
    ) -> dict[str, Any]:
        """
        Gibt zurück: recommended processing strength based on context.

        Adjustments:
        - Confidence-based: Low confidence (<0.7) reduces strength by 30%
        - Material-aware: Vinyl gets -20% compression, +10% denoise
                         Digital gets +20% denoise, -10% dereverb
        - Over-processing aware: Reduce if approaching threshold

        Args:
            module_type: Module type
            base_strength: Base strength value
            confidence: Confidence score (0-1)
            material_type: Optional material type ('vinyl', 'digital', 'tape')

        Returns:
            Dict with 'recommended_strength', 'adjustments', 'reasoning'
        """
        with self._lock:
            adjustments = []
            recommended = base_strength

            # Confidence-based adjustment
            if confidence < 0.7:
                reduction = 0.3
                recommended *= 1 - reduction
                adjustments.append(f"Low confidence ({confidence:.2f}): -{reduction * 100:.0f}%")
            elif confidence > 0.9:
                boost = 0.1
                recommended *= 1 + boost
                adjustments.append(f"High confidence ({confidence:.2f}): +{boost * 100:.0f}%")

            # Material-aware adjustments
            if material_type:
                if material_type.lower() == "vinyl":
                    if module_type == ModuleType.COMPRESSOR:
                        recommended *= 0.8
                        adjustments.append("Vinyl material: -20% compression (preserve dynamics)")
                    elif module_type == ModuleType.NOISE_REDUCTION:
                        recommended *= 1.1
                        adjustments.append("Vinyl material: +10% denoise (surface noise)")

                elif material_type.lower() == "digital":
                    if module_type == ModuleType.NOISE_REDUCTION:
                        recommended *= 1.2
                        adjustments.append("Digital material: +20% denoise (recording noise)")
                    elif module_type == ModuleType.DEREVERB:
                        recommended *= 0.9
                        adjustments.append("Digital material: -10% dereverb (preserve space)")

                elif material_type.lower() == "tape":
                    if module_type == ModuleType.DEHUMMER:
                        recommended *= 1.2
                        adjustments.append("Tape material: +20% dehummer (50/60Hz hum)")
                    elif module_type == ModuleType.COMPRESSOR:
                        recommended *= 0.85
                        adjustments.append("Tape material: -15% compression (tape saturation present)")

            # Over-processing safety check
            risk_check = self.check_over_processing_risk(module_type, recommended)
            if not risk_check["safe"]:
                # Reduce to fit within threshold
                # Use remaining headroom from current usage
                safe_strength = max(0, risk_check["threshold"] - risk_check["current"])
                adjustments.append(
                    f"Over-processing prevention: Reduced from {recommended:.2f} to {safe_strength:.2f} "
                    f"(current: {risk_check['current']:.2f}, threshold: {risk_check['threshold']:.2f})"
                )
                recommended = safe_strength

            # Clamp to valid range
            recommended = max(0.0, min(recommended, base_strength * 2.0))

            reasoning = (
                f"Base: {base_strength:.2f} → Recommended: {recommended:.2f} "
                f"({((recommended / base_strength - 1) * 100):+.1f}%)"
            )

            return {
                "recommended_strength": recommended,
                "base_strength": base_strength,
                "final_adjustment": (recommended / base_strength - 1) * 100,
                "adjustments": adjustments,
                "reasoning": reasoning,
                "over_processing_check": risk_check,
            }

    def register_processing(
        self,
        module_name: str,
        module_type: ModuleType,
        strength: float,
        confidence: float,
        parameters: dict | None = None,
    ) -> None:
        """
        Registriert a module and its processing parameters.
        Convenience method combining register_module with type/strength tracking.

        Args:
            module_name: Module name
            module_type: Module type
            strength: Processing strength applied
            confidence: Confidence score
            parameters: Optional additional parameters
        """
        with self._lock:
            if module_name not in self._modules:
                self._modules[module_name] = ModuleInfo(
                    name=module_name,
                    module_type=module_type,
                    strength=strength,
                    confidence=confidence,
                    parameters=parameters or {},
                )
            else:
                module = self._modules[module_name]
                module.module_type = module_type
                module.strength = strength
                module.confidence = confidence
                if parameters:
                    module.parameters.update(parameters)

            self._trigger_event(
                "processing_registered",
                {"module": module_name, "type": module_type.value, "strength": strength, "confidence": confidence},
            )

            self.logger.debug(
                f"Processing registered: {module_name} ({module_type.value}) "
                f"strength={strength:.2f}, confidence={confidence:.2f}"
            )

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"ProcessingContext(session_id='{self.session_id}', "
            f"phase={self._phase.value}, "
            f"modules={stats['total_modules']}, "
            f"completed={stats['completed_modules']})"
        )


# === Global Context Manager ===


class ContextManager:
    """
    Global Context Manager (Singleton).
    Manages multiple processing contexts.
    """

    _instance = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:  # pylint: disable=access-member-before-definition
            return

        self._contexts: dict[str, ProcessingContext] = {}
        self._context_lock = threading.RLock()
        self._initialized = True

        self.logger = logging.getLogger(__name__)

    def create_context(
        self, session_id: str, sample_rate: int = 48000, processing_mode: str = "restoration", **kwargs
    ) -> ProcessingContext:
        """
        Erstellt a new processing context.

        Args:
            session_id: Unique session ID
            sample_rate: Sample rate
            processing_mode: Processing mode
            **kwargs: Additional arguments

        Returns:
            New ProcessingContext
        """
        with self._context_lock:
            if session_id in self._contexts:
                logger.warning("Context already exists: %s", session_id)
                return self._contexts[session_id]

            context = ProcessingContext(
                session_id=session_id, sample_rate=sample_rate, processing_mode=processing_mode, **kwargs
            )
            self._contexts[session_id] = context

            logger.info("Context erstellt: %s", session_id)
            return context

    def get_context(self, session_id: str) -> ProcessingContext | None:
        """
        Gibt zurück: existing context.

        Args:
            session_id: Session ID

        Returns:
            ProcessingContext or None
        """
        with self._context_lock:
            return self._contexts.get(session_id)

    def remove_context(self, session_id: str) -> None:
        """
        Entfernt context.

        Args:
            session_id: Session ID
        """
        with self._context_lock:
            if session_id in self._contexts:
                del self._contexts[session_id]
                logger.info("Context removed: %s", session_id)

    def get_all_contexts(self) -> dict[str, ProcessingContext]:
        """Gibt zurück: all contexts (copy)."""
        with self._context_lock:
            return self._contexts.copy()

    def clear_all(self) -> None:
        """Löscht all contexts."""
        with self._context_lock:
            self._contexts.clear()
            self.logger.info("All contexts cleared")


# === Convenience Functions ===


def get_context_manager() -> ContextManager:
    """Gibt zurück: global context manager (singleton)."""
    return ContextManager()
