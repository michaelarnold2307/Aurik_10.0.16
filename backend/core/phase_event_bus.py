"""
§v10.118 PhaseEventBus — Minimaler Event-Bus für Cross-Phase-Kommunikation.

Eliminiert die Notwendigkeit, systemische Features (wie FeedbackChain-Awareness)
an mehreren verteilten Stellen zu verdrahten. Phasen feuern Events, andere
Phasen abonnieren sie — ein Kommunikationskanal statt N direkter Abhängigkeiten.

Initiale Events:
  - fc_pass_started: FeedbackChain-Durchlauf beginnt
  - fc_pass_ended:   FeedbackChain-Durchlauf endet

Usage:
  from backend.core.phase_event_bus import bus
  bus.emit("fc_pass_started")
  bus.on("fc_pass_started", my_callback)
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PhaseEventBus:
    """Thread-safe minimal event bus for cross-phase communication.

    §v10.118: Jede Phase kann Events emittieren und abonnieren.
    Keine Phase muss wissen, WER ihre Events konsumiert.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._event_history: list[tuple[str, float]] = []  # (event, timestamp)

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Registriere einen Callback für ein Event.

        Der Callback wird synchron aufgerufen, wenn emit() feuert.
        Non-blocking: Exception → Debug-Log, kein Abbruch.
        """
        with self._lock:
            self._listeners[event].append(callback)
        logger.debug("PhaseEventBus: '%s' → %s registriert", event, callback.__name__)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Feuere ein Event — alle registrierten Callbacks werden aufgerufen.

        Args:
            event: Event-Name (z.B. "fc_pass_started")
            *args, **kwargs: Werden an alle Callbacks weitergereicht
        """
        import time as _time

        with self._lock:
            listeners = list(self._listeners.get(event, []))
            self._event_history.append((event, _time.monotonic()))

        if not listeners:
            return

        for callback in listeners:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                logger.debug(
                    "PhaseEventBus: callback %s für '%s' fehlgeschlagen: %s",
                    callback.__name__,
                    event,
                    exc,
                )

    def is_active(self, event: str) -> bool:
        """Prüft, ob ein Event aktuell aktiv ist (letztes emit ohne matching end)."""
        # Simple check: count starts vs ends for paired events
        starts = sum(1 for e, _ in self._event_history if e == event)
        if event.endswith("_started"):
            end_event = event.replace("_started", "_ended")
            ends = sum(1 for e, _ in self._event_history if e == end_event)
            return starts > ends
        return False

    def clear(self) -> None:
        """Lösche alle Listener (für Tests)."""
        with self._lock:
            self._listeners.clear()
            self._event_history.clear()


# Singleton
bus = PhaseEventBus()
