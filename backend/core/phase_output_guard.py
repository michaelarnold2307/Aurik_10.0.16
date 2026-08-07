"""
backend/core/phase_output_guard.py — §3.9.1 + §3.9.3 Stability Invariants
===========================================================================

§3.9.1  Per-Phase-Inference-Timeout:
    Heavy ML inference (≥ 0.5 GB model) MUST run inside
    _run_inference_with_timeout() so a hung ONNX/torch call can be
    cancelled instead of blocking the pipeline forever.

§3.9.3  Phase-Output-Guard:
    The @phase_output_guard decorator sanitises every phase return value:
    nan_to_num → clip(-1,1) → assert isfinite → assert float32.
    NaN propagation from a corrupt ML inference is structurally forbidden.
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §3.9.1  Constants
# ---------------------------------------------------------------------------

PHASE_INFERENCE_TIMEOUT_S: float = 300.0  # 5 min wall-clock; exceeded = hung model


# ---------------------------------------------------------------------------
# §3.9.1  Custom exceptions
# ---------------------------------------------------------------------------


class InferenceTimeoutError(RuntimeError):
    """Raised when ML inference exceeds PHASE_INFERENCE_TIMEOUT_S.

    Caller MUST catch and fall back to DSP path; affected phase MUST be
    added to RestorationResult.deferred_phases for KMV Stufe 2.
    """


class PhaseOutputError(RuntimeError):
    """Raised by @phase_output_guard when post-sanitisation audio is still
    non-finite or has wrong dtype — indicates a severe ML inference failure.
    Caller MUST treat this as a DSP-fallback trigger.
    """


# ---------------------------------------------------------------------------
# §3.9.1  Inference timeout helper
# ---------------------------------------------------------------------------


def _run_inference_with_timeout(
    fn: Callable[..., Any],
    /,
    *args: Any,
    timeout: float = PHASE_INFERENCE_TIMEOUT_S,
    **kwargs: Any,
) -> Any:
    """Führt aus: *fn* in a daemon thread with a wall-clock *timeout*.

    On timeout: logs error and raises InferenceTimeoutError.
    On any other exception: propagates as-is.

    Thread is marked daemon so it does not prevent process exit.
    The executor uses a single worker thread to avoid spawning many threads
    for sequential heavy inference calls.

    Usage::

        result = _run_inference_with_timeout(
            session.run, None, {"input": tensor}, timeout=300.0
        )

    Args:
        fn:      Callable to run (ONNX session.run, torch model forward, …).
        *args:   Positional arguments forwarded to *fn*.
        timeout: Wall-clock timeout in seconds (default: 300 s).
        **kwargs: Keyword arguments forwarded to *fn*.

    Raises:
        InferenceTimeoutError: Inference exceeded *timeout* seconds.
        Any exception raised by *fn*: propagated unchanged.
    """
    exc = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="aurik-inf")
    fut = exc.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.error(
            "Inference Zeitlimit after %.0f s — fn=%s. DSP Ersatzpfad required; Verarbeitungsschritt added to deferred_phases.",
            timeout,
            getattr(fn, "__qualname__", repr(fn)),
        )
        raise InferenceTimeoutError(
            f"Inference timeout ({timeout:.0f} s): {getattr(fn, '__qualname__', repr(fn))}"
        ) from None
    except Exception:
        raise
    finally:
        # shutdown(wait=False) so we never block on a hung inference thread.
        # cancel_futures=True cancels any pending (not yet started) futures.
        # The worker thread is left to finish naturally in the background.
        exc.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# §3.9.3  Phase output guard decorator
# ---------------------------------------------------------------------------


def phase_output_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: sanitise phase return audio and enforce structural invariants.

    Applies to every value returned by *fn* that is an np.ndarray:

        1. ``np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)``
        2. ``np.clip(audio, -1.0, 1.0)``
        3. ``assert np.isfinite(audio).all()``  — hard-fail if guard insufficient
        4. ``assert audio.dtype == np.float32``

    If step 3 or 4 fails: logs CRITICAL and raises PhaseOutputError.
    Caller (PerPhaseMusicalGoalsGate / UV3) should catch PhaseOutputError
    and trigger a DSP fallback.

    Non-array return values (e.g. tuple/list containing additional metadata)
    are handled by extracting the first ndarray element.

    NaN propagation from ML outputs is structurally forbidden (§3.9.3).
    """

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return _guard_result(result, fn)

    return _wrapper


def _guard_result(result: Any, fn: Callable[..., Any]) -> Any:
    """Wendet §3.9.3-Invarianten auf *result* an und gibt bereinigten Wert zurück."""
    if isinstance(result, np.ndarray):
        return _sanitise_audio(result, fn)

    if isinstance(result, tuple):
        parts = list(result)
        for i, part in enumerate(parts):
            if isinstance(part, np.ndarray):
                parts[i] = _sanitise_audio(part, fn)
        return tuple(parts)

    # Non-audio return (e.g. a dataclass / dict) — pass through unchanged.
    return result


def _sanitise_audio(audio: np.ndarray, fn: Callable[..., Any]) -> np.ndarray:
    """In-place-safe sanitisation: nan_to_num → clip → assert."""
    fn_name = getattr(fn, "__qualname__", repr(fn))

    # Step 1: replace NaN / ±Inf with 0 (silence is safe)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 2: hard clip to [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)

    # Step 3: verify — should always pass after steps 1+2
    if not np.isfinite(audio).all():
        msg = f"Phase output still non-finite after guard: fn={fn_name}"
        logger.critical(msg)
        raise PhaseOutputError(msg)

    # Step 4: dtype contract (all pipeline audio must be float32 @ 48 kHz)
    if audio.dtype != np.float32:
        try:
            audio = audio.astype(np.float32)
        except Exception as exc:
            msg = f"Phase output dtype={audio.dtype} cannot be cast to float32: fn={fn_name}"
            logger.critical(msg)
            raise PhaseOutputError(msg) from exc

    return audio
