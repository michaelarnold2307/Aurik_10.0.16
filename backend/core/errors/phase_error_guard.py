"""Phase Error Guard — Decorator für Graceful Degradation.

§15.8 [RELEASE_MUST]: Jede Phase MUSS Fehler abfangen und degradieren können.
Der ``@phase_error_guard``-Decorator ist der zentrale Mechanismus dafür.

Usage::

    @phase_error_guard(phase_name="Phase 03 – Noise Reduction", fail_fast=False)
    def phase_03_noise_reduction(audio: np.ndarray, sr: int = 48000, **kwargs) -> np.ndarray:
        ...

    @phase_error_guard(fail_fast=True)  # Kritische Phase — keine Degradation
    def phase_01_load(audio: np.ndarray, sr: int = 48000, **kwargs) -> np.ndarray:
        ...

Autor: Aurik 10 — 11. Juli 2026
Referenz: Spec 15 §15.8
"""

from __future__ import annotations

import functools
import logging
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

from backend.core.errors.degraded_output import DegradedOutput

logger = logging.getLogger(__name__)

# ── Nicht abfangbare Exceptions ────────────────────────────────────────────
# Diese werden IMMER durchgereicht, auch bei fail_fast=False:
_PASSTHROUGH_EXCEPTIONS = (
    KeyboardInterrupt,
    SystemExit,
    MemoryError,  # Wenn kein RAM mehr — keine Chance auf Degradation
)


def phase_error_guard(
    phase_name: str = "",
    fail_fast: bool = False,
    default_output: np.ndarray | None = None,
) -> Callable:
    """Decorator: Fängt Fehler in Phasen-Funktionen und degradiert statt abzustürzen.

    Args:
        phase_name:     Name der Phase für Logging/Debugging.
        fail_fast:      True → Exception re-raisen (für kritische Phasen).
        default_output: Fallback-Audio wenn kein Input verfügbar ist.

    Returns:
        Decorierte Funktion die statt Exceptions ``DegradedOutput`` liefert.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # ── Phase-Name extrahieren ────────────────────────────────────
            name = phase_name or func.__name__

            # ── Input-Audio für Fallback sichern (erstes ndarray-Argument) ─
            input_audio: np.ndarray | None = default_output
            input_sr: int = 48000
            for arg in args:
                if isinstance(arg, np.ndarray):
                    input_audio = arg.copy()
                    break
            if input_audio is None:
                for v in kwargs.values():
                    if isinstance(v, np.ndarray):
                        input_audio = v.copy()
                        break
            if "sr" in kwargs:
                input_sr = kwargs["sr"]
            elif len(args) > 1 and isinstance(args[1], int):
                input_sr = args[1]

            # ── Ausführung ────────────────────────────────────────────────
            try:
                result = func(*args, **kwargs)

                # Falls die Funktion bereits ein DegradedOutput zurückgibt:
                if isinstance(result, DegradedOutput):
                    if not result.phase_name:
                        result.phase_name = name
                    return result
                return result

            except _PASSTHROUGH_EXCEPTIONS:
                # System-Kill-Signale immer durchreichen
                raise

            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.exception("Phase '%s' fehlgeschlagen: %s", name, error_msg)

                if fail_fast:
                    raise

                # ── DegradedOutput bauen ──────────────────────────────────
                if input_audio is None:
                    # Kein Input-Audio verfügbar → leeres Array als Notlösung
                    logger.error(
                        "Verarbeitungsschritt '%s': Kein Eingabe-Audio für Ersatzpfad verfügbar. Leeres Array zurückgegeben (1 s Stille).",
                        name,
                    )
                    input_audio = np.zeros(input_sr, dtype=np.float32)

                degraded = DegradedOutput(
                    audio=input_audio,
                    sample_rate=input_sr,
                    warnings=[
                        f"Phase '{name}' degradiert: {error_msg}",
                        f"Traceback (gekürzt): {traceback.format_exc()[-500:]}",
                    ],
                    phase_name=name,
                    original_error=error_msg,
                    _is_degraded=True,
                )

                logger.warning(
                    "Verarbeitungsschritt '%s': Graceful Degradation aktiviert. Audio unverändert durchgereicht. Fehler: %s",
                    name,
                    error_msg,
                )
                return degraded

        return wrapper

    return decorator
