"""
Aurik10/ui/ml_refinement_thread.py — MLRefinementThread §2.38 KMV Stufe 2
==========================================================================

Implements the background ML-refinement thread that re-runs deferred phases
with no RT limit after the Stufe-1 (fast) export is already on disk.

Spec §2.38 invariants enforced here:
  - LIMIT_BACKGROUND = float("inf") — no RT cap for this thread only.
  - Stufe-2 export only overwrites Stufe-1 file when stufe2_quality >= stufe1_quality.
  - Atomic write: .tmp → os.replace (POSIX-safe).
  - ml_memory_budget.try_allocate("kmv_job", size_gb) registered before processing;
    released on both success and failure paths.
  - isInterruptionRequested() checked between phases (escapable).
  - Single-active-invariant enforced by caller (ModernMainWindow).
  - Priority: QThread.LowPriority + os.nice(10) on Linux.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

if TYPE_CHECKING:
    from PyQt5.QtCore import QThread, pyqtSignal
else:
    try:
        from PyQt5.QtCore import QThread, pyqtSignal
    except ImportError:  # Headless test environment

        class QThread:
            """Minimaler QThread-Stub für headless Testumgebungen."""

            LowPriority = 1

            def __init__(self) -> None:
                """Initialisiert den Stub-Threadzustand."""
                self._interrupt = False

            def isInterruptionRequested(self) -> bool:
                """Kompatible Qt-API: liefert den lokalen Interrupt-Status."""
                return self._interrupt

            def requestInterruption(self) -> None:
                """Kompatible Qt-API: markiert den Thread als unterbrochen."""
                self._interrupt = True

            def setPriority(self, _p) -> None:
                """No-op im Stub: Prioritäten werden ohne Qt nicht gesetzt."""

            def start(self, _p=None) -> None:
                """Kompatible Qt-API: startet synchron per direktem run()-Aufruf."""
                self.run()

        def pyqtSignal(*_args):
            """Einfacher Signal-Stub mit connect/emit-Kompatibilität."""

            class _Sig:
                """Signalobjekt-Stub für Tests ohne PyQt5."""

                def connect(self, _fn):
                    """No-op connect für API-Kompatibilität."""

                def emit(self, *a, **kw):
                    """No-op emit für API-Kompatibilität."""

            return _Sig()


try:
    from backend.api.bridge import get_deferred_refinement_job_class as _get_drj_class

    DeferredRefinementJob = _get_drj_class()
except ImportError as _bridge_exc:

    class DeferredRefinementJob:  # type: ignore[no-redef]
        """UI-lokaler Fallback-Typ bei fehlender Bridge (nur fuer Typvertrag-Tests)."""

        def release_buffer(self) -> None:
            """No-op im Fallback: kein ML-Memory-Budget zu releasen."""

    logger = logging.getLogger(__name__)
    logger.warning(
        "MLRefinementThread: Bridge-Import fehlgeschlagen (%s), verwende lokalen Fallback fuer DeferredRefinementJob",
        _bridge_exc,
    )

logger = logging.getLogger(__name__)

# ── Spec §2.38: Hintergrundthread läuft ohne RT-Limit ────────────────────────
_LIMIT_BACKGROUND: float = float("inf")


def _set_job_budget_registered(job: object, registered: bool) -> None:
    """Setzt _budget_registered best-effort für release_buffer()-Semantik."""
    try:
        object.__setattr__(job, "_budget_registered", bool(registered))
    except Exception:
        logger.debug(
            "KMV Stufe 2: _budget_registered konnte nicht gesetzt werden (non-fatal)",
            exc_info=True,
        )


class MLRefinementThread(QThread):
    """KMV Stufe-2 background refinement thread.

    Signals (§2.38 Spec contract):
        refinement_started(output_path: str, n_deferred_phases: int)
        refinement_phase_done(phase_id: str, quality_improvement_delta: float)
        refinement_progress(pct: int, phase_name: str)       # 0–100
        refinement_complete(output_path: str, result: object) # final RestorationResult
        refinement_cancelled(output_path: str)               # Stufe-1 export kept
    """

    refinement_started = pyqtSignal(str, int)
    refinement_phase_done = pyqtSignal(str, float)
    refinement_progress = pyqtSignal(int, str)
    refinement_complete = pyqtSignal(str, object)
    refinement_cancelled = pyqtSignal(str)

    def __init__(self, job: Any) -> None:
        super().__init__()
        self.job = job
        self._result = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def should_start(job: Any) -> bool:
        """Gibt True when all Stufe-2 start conditions are met (§2.38) zurück."""
        if not job.deferred_phase_ids:
            return False
        try:
            if _psutil is None:
                raise RuntimeError("psutil nicht verfuegbar")
            avail_gb = _psutil.virtual_memory().available / 1024**3
            if avail_gb < 4.0:
                logger.info("KMV Stufe 2 nicht gestartet: nur %.1f GB RAM frei (< 4 GB)", avail_gb)
                return False
        except Exception:
            logger.debug("KMV should_start: psutil nicht verfuegbar, RAM-Check uebersprungen", exc_info=True)
            # psutil not available -> proceed without check
        return True

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Führt aus: KMV Stufe-2: full ML pass → conditional atomic overwrite."""
        job = self.job
        output_path = job.output_path

        # Set low OS priority (§2.38: do not compete with UI)
        try:
            self.setPriority(getattr(QThread, "LowPriority", 2))  # type: ignore[arg-type]  # Qt::LowPriority, int fallback vs Priority enum stub
        except Exception:
            logger.debug("KMV Stufe 2: QThread-Prioritaet konnte nicht gesetzt werden", exc_info=True)
        try:
            os.nice(10)
        except Exception:
            logger.debug("KMV Stufe 2: os.nice(10) nicht anwendbar", exc_info=True)

        self.refinement_started.emit(output_path, job.n_deferred)
        logger.info(
            "KMV Stufe 2 gestartet: %d deferred phases | output=%s",
            job.n_deferred,
            output_path,
        )

        # ── 1. Register audio in ml_memory_budget ────────────────────────────
        _budget_registered = False
        try:
            from backend.api.bridge import get_ml_memory_budget  # pylint: disable=import-outside-toplevel

            _budget = get_ml_memory_budget()
            _budget_registered = _budget.try_allocate("kmv_job", job.audio_size_gb)
            _set_job_budget_registered(job, _budget_registered)
            if not _budget_registered:
                logger.warning(
                    "KMV Stufe 2: ml_memory_budget.try_allocate('kmv_job', %.2f GB) fehlgeschlagen"
                    " — Stufe-2-Export abgebrochen.",
                    job.audio_size_gb,
                )
                self.refinement_cancelled.emit(output_path)
                return
        except ImportError as _be:
            logger.debug("ml_memory_budget nicht verfügbar (KMV): %s", _be)
            # Continue without budget guard if module absent (test environments)
            _budget_registered = False
            _set_job_budget_registered(job, False)

        t0 = time.perf_counter()
        _result = None
        try:
            if self.isInterruptionRequested():
                self.refinement_cancelled.emit(output_path)
                return

            # ── 2. Full UV3 pass via AurikDenker with no_rt_limit=True ───────
            self.refinement_progress.emit(5, "ML-Veredelung: Analyse …")

            try:
                from backend.api.bridge import (  # pylint: disable=import-outside-toplevel
                    get_aurik_denker_instance as _get_denker,
                )

                _denker = _get_denker()
            except ImportError as _imp_err:
                logger.error("KMV Stufe 2: AurikDenker nicht verfügbar: %s", _imp_err)
                self.refinement_cancelled.emit(output_path)
                return

            if _denker is None:
                logger.error("KMV Stufe 2: AurikDenker-Singleton ist None")
                self.refinement_cancelled.emit(output_path)
                return

            # Progress reporter that maps 0–100 → 10–90 (leave room for pre/post steps)
            def _progress_cb(pct: int, msg: str, elapsed_s: float = 0.0) -> None:
                if self.isInterruptionRequested():
                    return
                mapped = 10 + int(pct * 0.80)
                self.refinement_progress.emit(min(mapped, 90), msg)

            _result = _denker.denke(
                audio=job.audio_original,
                sr=job.sr,
                mode=job.mode,
                progress_callback=_progress_cb,
                no_rt_limit=True,  # §2.38 KMV Stufe 2: kein RT-Limit
                cached_defect_result=job.cached_defect_result,
                cached_era_result=job.cached_era_result,
                cached_medium_result=job.cached_medium_result,
                input_path=job.input_path,
                output_path=output_path,
            )

            if self.isInterruptionRequested():
                logger.info("KMV Stufe 2 abgebrochen nach Denker-Lauf")
                self.refinement_cancelled.emit(output_path)
                return

            self.refinement_progress.emit(91, "ML-Veredelung: Qualitätsprüfung …")

            # ── 3. Quality invariant: only overwrite when Stufe 2 ≥ Stufe 1 ─
            _r = _result  # may be AurikErgebnis or RestorationResult
            _stufe2_quality = _extract_quality(_r)

            if _stufe2_quality is None or _stufe2_quality < job.stufe1_quality:
                logger.info(
                    "KMV Stufe 2: Qualität nicht verbessert (Stufe2=%.3f < Stufe1=%.3f) — Stufe-1-Export behalten.",
                    _stufe2_quality if _stufe2_quality is not None else float("nan"),
                    job.stufe1_quality,
                )
                self.refinement_cancelled.emit(output_path)
                return

            self.refinement_progress.emit(93, "ML-Veredelung: Export …")

            # ── 4. Extract audio and write atomically ──────────────────────
            _audio = _extract_audio(_r)
            if _audio is None or _audio.size == 0:
                logger.warning("KMV Stufe 2: Kein Audio im Ergebnis — Stufe-1-Export behalten.")
                self.refinement_cancelled.emit(output_path)
                return

            # Validate audio
            _audio = np.nan_to_num(_audio, nan=0.0, posinf=0.0, neginf=0.0)
            _audio = np.clip(_audio, -1.0, 1.0)

            _tmp_path = _build_atomic_temp_path(output_path)
            try:
                _write_audio(_audio, job.sr, _tmp_path)
                os.replace(_tmp_path, output_path)
            except Exception as _write_err:
                logger.error("KMV Stufe 2: Atomic-Write fehlgeschlagen: %s", _write_err)
                try:
                    Path(_tmp_path).unlink(missing_ok=True)
                except Exception:
                    logger.debug("KMV Stufe 2: Temp-Datei konnte nicht entfernt werden: %s", _tmp_path, exc_info=True)
                self.refinement_cancelled.emit(output_path)
                return

            # ── 5. Mark result as Stufe-2-complete ───────────────────────
            try:
                _r.refinement_complete = True
                _r.stufe2_quality_estimate = float(_stufe2_quality)
            except Exception:
                logger.debug("KMV Stufe 2: Result-Metadaten konnten nicht gesetzt werden", exc_info=True)

            elapsed = time.perf_counter() - t0
            logger.info(
                "KMV Stufe 2 erfolgreich: Stufe2=%.3f (Δ+%.3f) in %.1f s → %s",
                _stufe2_quality,
                _stufe2_quality - job.stufe1_quality,
                elapsed,
                output_path,
            )
            self.refinement_progress.emit(100, "ML-Veredelung abgeschlossen ✓")
            self.refinement_complete.emit(output_path, _r)

        except Exception as _run_err:
            logger.error("KMV Stufe 2: Unerwarteter Fehler: %s", _run_err, exc_info=True)
            self.refinement_cancelled.emit(output_path)
        finally:
            # §3.9.9: ALWAYS release buffer — success, cancellation, or exception.
            # DeferredRefinementJob.release_buffer() calls ml_memory_budget.release()
            # and sets audio_original=None so GC can reclaim the large array.
            # §2.38a Invariante: Nur release() wenn try_allocate() erfolgreich war
            try:
                job.release_buffer()
            except Exception as _rel_err:
                logger.debug("KMV buffer release failed (non-fatal): %s", _rel_err)


# ── Private helpers ────────────────────────────────────────────────────────────


def _extract_quality(result: object) -> float | None:
    """Extrahiert quality_estimate from AurikErgebnis or RestorationResult."""
    if result is None:
        return None
    # AurikErgebnis
    for attr in ("quality_estimate", "restoration_result"):
        val = getattr(result, attr, None)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return float(val)
        # Nested RestorationResult
        inner = getattr(val, "quality_estimate", None)
        if isinstance(inner, (int, float)):
            return float(inner)
    return None


def _extract_audio(result: object) -> np.ndarray | None:
    """Extrahiert audio array from AurikErgebnis or RestorationResult."""
    if result is None:
        return None
    # Direct .audio attribute
    audio = getattr(result, "audio", None)
    if isinstance(audio, np.ndarray) and audio.size > 0:
        return audio
    # Nested .restoration_result.audio
    rr = getattr(result, "restoration_result", None)
    if rr is not None:
        audio2 = getattr(rr, "audio", None)
        if isinstance(audio2, np.ndarray) and audio2.size > 0:
            return audio2
    return None


def _write_audio(audio: np.ndarray, sr: int, path: str) -> None:
    """Schreibt audio to file — tries soundfile, falls back to scipy.io.wavfile."""
    audio = np.clip(np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    try:
        import soundfile as sf  # pylint: disable=import-outside-toplevel

        mono_or_stereo = audio if audio.ndim == 2 else audio[:, None]
        sf.write(path, mono_or_stereo, sr, subtype="FLOAT")
        return
    except Exception:
        logger.debug("KMV _write_audio: soundfile write fehlgeschlagen, nutze scipy-Fallback", exc_info=True)
    import scipy.io.wavfile as _wf  # pylint: disable=import-outside-toplevel

    _a = (audio * 32767).astype(np.int16)
    _wf.write(path, sr, _a)


def _build_atomic_temp_path(path: str) -> str:
    """Erzeugt einen temporären Pfad und behält die finale Dateiendung bei."""
    target = Path(path)
    if target.suffix:
        return str(target.with_name(f"{target.stem}.kmv_tmp{target.suffix}"))
    return str(target.with_name(f"{target.name}.kmv_tmp"))
