"""Pre-analysis module — single authoritative backend entry point for all
pre-restoration analysis tasks.

Replaces the four scattered frontend background threads (_carrier_bg,
_detect_era_genre_bg, _estimate_restorability_bg, _run_defect_scan_bg)
with a single, testable, spec-compliant backend call.

Spec compliance:
  - Analysis modules run at native import SR (no resampling before analysis).
  - Restorability estimator receives the 48 kHz-resampled processing audio
    (its API requires 48 kHz per spec §2.26).
  - DefectScanner.scan() receives native-SR audio with file_ext for
    Bayesian posterior-zeroing (Bug-15 fix).
  - MediumDetector.detect() receives native-SR audio with file_ext.
  - EraClassifier and GermanSchlagerClassifier receive native-SR audio.
  - All four analyses run in parallel (ThreadPoolExecutor max_workers=4).
  - Result is stored in bridge cache so UV3 never re-runs any classifier.

Usage (frontend)::

    from backend.core.pre_analysis import run_pre_analysis, PreAnalysisResult

    result: PreAnalysisResult = run_pre_analysis(
        audio_native=audio_before_resample,
        sr_native=sr_native,
        audio_48k=audio_after_resample,
        file_path="/path/to/song.mp3",
        progress_callback=lambda pct, msg: ...,   # optional
    )
    # Display result.medium, result.era, result.genre, etc. in UI.
    # Pass to UV3 via bridge cache — no kwarg threading required.

Usage (UV3 / CLI — no frontend)::

    result = run_pre_analysis(audio_native=audio, sr_native=sr,
                              audio_48k=audio_48k, file_path=path)
"""

from __future__ import annotations

import gc
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from concurrent import futures as _cf
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Observed: DefectScanner needs ~80s for 60s audio (133% overhead on this hardware).
# 150s = 1.87x buffer. concurrent.futures.TimeoutError != builtins.TimeoutError in Python 3.10.
_SUBSTEP_TIMEOUT_S = 240.0

# ---------------------------------------------------------------------------
# Progress state (single source of truth, §G19/V71)
# ---------------------------------------------------------------------------


@dataclass
class ProgressState:
    """Single source of truth for all pre-analysis progress callbacks.

    ALL callbacks (scan_progress, progress_callback, emit_load_progress)
    write to this object. The GUI reads from it. No more disconnected streams.
    """

    pct: float = 0.0  # 0.0–100.0 — drives the progress bar
    step_msg: str = ""  # Human-readable step description
    step_pct: int = 0  # Integer 0–100 for the current step
    total_steps: int = 4  # Expected number of parallel steps
    done_steps: int = 0  # Completed parallel steps
    errors: list[str] = field(default_factory=list)
    last_update: float = 0.0  # time.monotonic() of last write


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PreAnalysisResult:
    """Aggregated result of all pre-restoration analysis steps.

    All fields use the canonical result types from their respective modules.
    Every field is Optional so callers can handle partial failures gracefully.
    """

    # Carrier / medium chain (forensics.medium_detector)
    medium: object | None = None  # MediumDetectionResult

    # Recording era (backend.core.era_classifier)
    era: object | None = None  # EraResult

    # Genre classification (backend.core.genre_classifier)
    genre: object | None = None  # SchlagerClassificationResult

    # Defect scan (backend.core.defect_scanner)
    defects: object | None = None  # DefectAnalysisResult

    # Restorability estimate (backend.core.restorability_estimator)
    restorability: object | None = None  # RestorabilityResult

    # Metadata
    native_sr: int = 0
    file_path: str = ""
    elapsed_seconds: float = 0.0

    # Per-step error messages (populated on exception, step still gets None above)
    errors: dict[str, str] = field(default_factory=dict)


_run_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_mono_native(audio: np.ndarray) -> np.ndarray:
    """Konvertiert to mono without SR change; clip & nan-guard."""
    if audio.ndim == 2:
        # shape (N, 2) or (2, N)
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T
        mono = audio.mean(axis=1)
    else:
        mono = audio.copy()
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    mono_f32: np.ndarray = np.asarray(np.clip(mono, -1.0, 1.0), dtype=np.float32)
    return mono_f32


def _load_symbol(module_name: str, symbol_name: str) -> object:
    """Lädt optional or heavy symbols lazily without inline import statements."""
    return getattr(import_module(module_name), symbol_name)


def _resample_for_restorability(audio_native: np.ndarray, sr_native: int) -> tuple[np.ndarray, int]:
    """Resample to 48 kHz if not already there (restorability estimator requires 48 kHz)."""
    if sr_native == 48_000:
        return audio_native, sr_native
    try:
        _rp = cast(Callable[..., np.ndarray], _load_symbol("scipy.signal", "resample_poly"))

        gcd = math.gcd(int(sr_native), 48_000)
        audio_48 = _rp(
            audio_native,
            48_000 // gcd,
            int(sr_native) // gcd,
            axis=0 if audio_native.ndim > 1 else -1,
        ).astype(np.float32)
        return audio_48, 48_000
    except Exception as exc:
        logger.warning("pre_Analyse: resample for restorability fehlgeschlagen (%s) — using native SR", exc)
        return audio_native, sr_native


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pre_analysis(
    audio_native: np.ndarray,
    sr_native: int,
    *,
    audio_48k: np.ndarray | None = None,
    file_path: str = "",
    progress_callback: Callable[[int, str], None] | None = None,
    scan_progress_callback: Callable[[float], None] | None = None,
    store_in_bridge_cache: bool = True,
) -> PreAnalysisResult:
    """Führt aus: all pre-restoration analyses in parallel and return a PreAnalysisResult.

    This is the single authoritative entry point for all pre-restoration analysis.
    It replaces the four scattered frontend background threads.

    Args:
        audio_native:       Audio at native import SR (no resampling applied).
                            Used for medium, era, genre, defect analysis.
        sr_native:          Sample rate of audio_native [Hz].
        audio_48k:          Optional pre-resampled audio at 48 kHz, used for
                            restorability estimation (saves one resample if
                            the caller already has this). If None, computed
                            internally.
        file_path:          Absolute path to source file. Used for file_ext
                            (Bayesian posterior-zeroing) and bridge cache key.
        progress_callback:  Optional (pct: int, msg: str) -> None callback.
                            Reports 0, 25, 50, 75, 100.
        scan_progress_callback:
                    Optional (pct: float) -> None callback forwarded to
                            DefectScanner.scan() for fine-grained progress.
        store_in_bridge_cache:
                            When True, stores each sub-result in the bridge
                            LRU cache so UV3 never re-runs any classifier.

    Returns:
        PreAnalysisResult with all sub-results populated (or None on failure).
    """
    t0 = time.monotonic()

    _cb = progress_callback or (lambda pct, msg: None)
    _cb(0, "Voranalyse gestartet…")

    _cached_parts: dict[str, object | None] = {}
    # Fast-Path: Falls alle Voranalyse-Subresultate bereits im Bridge-Cache liegen,
    # liefern wir deterministisch aus dem Cache statt die Analyzer erneut zu starten.
    if store_in_bridge_cache and file_path:
        _cached_parts = _load_cached_parts(file_path)
        _cached_result = _build_result_from_cached_parts(_cached_parts, sr_native=sr_native, file_path=file_path)
        if _cached_result is not None:
            _cached_result.elapsed_seconds = time.monotonic() - t0
            _cb(100, "Voranalyse aus Cache geladen.")
            logger.info("pre_Analyse: Zwischenspeicher-hit for %s (%.3fs)", file_path, _cached_result.elapsed_seconds)
            return _cached_result

    file_ext = os.path.splitext(file_path)[1].lower() if file_path else ""

    # Prepare 48 kHz audio for restorability if not supplied
    if audio_48k is None:
        audio_48k, _ = _resample_for_restorability(audio_native, sr_native)

    # Derive material hint from era result later; use "unknown" for initial defect scan
    result = PreAnalysisResult(native_sr=sr_native, file_path=file_path)

    # ------------------------------------------------------------------
    # Step 1 — Medium detection (native SR, with file_ext) — run first
    # so material hint is available for restorability.
    # Steps 2-5 run in parallel after medium finishes.
    # ------------------------------------------------------------------
    _medium_result = _cached_parts.get("medium")
    if _medium_result is not None:
        result.medium = _medium_result
        logger.debug("pre_Analyse: medium aus Zwischenspeicher geladen")

    _medium_primary_error: str | None = None
    if _medium_result is None:
        try:
            _get_md = cast(Callable[[], Any], _load_symbol("forensics.medium_detector", "get_medium_detector"))

            _medium_result = _get_md().detect(audio_native, sr_native, file_ext=file_ext)
            result.medium = _medium_result
            _medium_result_any = cast(Any, _medium_result)

            logger.info(
                "pre_Analyse: medium=%s conf=%.2f chain=%s",
                _medium_result_any.primary_material,
                _medium_result_any.confidence,
                _medium_result_any.chain_label,
            )
        except Exception as exc:
            _medium_primary_error = str(exc)
            logger.warning("pre_Analyse: primary medium detection fehlgeschlagen (%s)", exc)

    # Strict detector-only policy:
    # - Primary detector exactly once
    # - No legacy MediumClassifier fallback in production chain detection (§6.7)
    if _medium_result is None:
        if _medium_primary_error is not None:
            result.errors["medium"] = f"primary_failed={_medium_primary_error}; no_legacy_fallback=true"
        else:
            result.errors["medium"] = "medium_detection_failed; no_legacy_fallback=true"
        logger.warning(
            "pre_Analyse: medium detection nicht verfuegbar; continuing without medium Ergebnis (legacy Ersatzpfad deaktiviert)"
        )

    _cb(20, "Tonträger erkannt — analysiere Ära, Genre und Defekte…")

    # Material string for downstream modules
    _material_str = "unknown"
    if _medium_result is not None:
        _material_str = str(getattr(_medium_result, "primary_material", None) or "unknown")
        _transfer_chain = list(getattr(_medium_result, "transfer_chain", None) or [])
    else:
        _transfer_chain = []

    # ------------------------------------------------------------------
    # Steps 2–5 — Era, Genre, DefectScan, Restorability in parallel
    # ------------------------------------------------------------------
    def _run_era() -> object:
        """Era classification via Bridge (kanonischer Pfad)."""
        _classify = cast(Callable[..., Any], _load_symbol("backend.api.bridge", "get_era_classifier_fn"))
        return _classify()(audio_native, sr_native, transfer_chain=_transfer_chain)

    def _run_genre() -> object:
        """Genre classification via Bridge (kanonischer Pfad)."""
        _classify = cast(Callable[..., Any], _load_symbol("backend.api.bridge", "get_genre_classifier_fn"))
        return _classify()(audio_native, sr_native)

    def _run_defects() -> object:
        _DS = cast(Callable[..., Any], _load_symbol("backend.core.defect_scanner", "DefectScanner"))

        scanner = _DS(sample_rate=sr_native, material_type=None)
        _kw: dict = {
            "sample_rate": sr_native,
            "file_ext": file_ext,
            "forensic_medium_result": _medium_result,
        }
        # §2.47a: Pass forensically-detected material to scan() so threshold setup
        # uses the MediumDetector result rather than the internal heuristic fallback.
        if _material_str not in ("unknown", ""):
            _kw["material_type"] = _material_str
        if scan_progress_callback is not None:
            _kw["progress_callback"] = scan_progress_callback
        return scanner.scan(audio_native, **_kw)

    def _run_restorability() -> object:
        _er = cast(
            Callable[..., object],
            _load_symbol("backend.core.restorability_estimator", "estimate_restorability"),
        )

        return _er(audio_48k, 48_000, material=_material_str)

    _step_fns: dict[str, Callable[[], object]] = {}
    if _cached_parts.get("era") is not None:
        result.era = _cached_parts["era"]
        logger.debug("pre_Analyse: step=era aus Zwischenspeicher geladen")
    else:
        _step_fns["era"] = _run_era

    if _cached_parts.get("genre") is not None:
        result.genre = _cached_parts["genre"]
        logger.debug("pre_Analyse: step=genre aus Zwischenspeicher geladen")
    else:
        _step_fns["genre"] = _run_genre

    if _cached_parts.get("defects") is not None:
        result.defects = _cached_parts["defects"]
        logger.debug("pre_Analyse: step=defects aus Zwischenspeicher geladen")
    else:
        _step_fns["defects"] = _run_defects

    if _cached_parts.get("restorability") is not None:
        result.restorability = _cached_parts["restorability"]
        logger.debug("pre_Analyse: step=restorability aus Zwischenspeicher geladen")
    else:
        _step_fns["restorability"] = _run_restorability

    if _step_fns:
        _total_steps = len(_step_fns)
        _done_steps = 0

        # Era + Genre laufen ASYNCHRON als Daemon-Thread (wie alte _detect_era_genre_bg).
        # CLAP-Kaltstart dauert 200+s auf ROCm — synchrones Warten blockiert
        # die gesamte Pre-Analysis. Der Daemon-Thread lädt CLAP im Hintergrund
        # und setzt result.era/result.genre wenn fertig.
        _clap_steps = {k: v for k, v in _step_fns.items() if k in ("era", "genre")}
        _other_steps = {k: v for k, v in _step_fns.items() if k not in ("era", "genre")}

        if _clap_steps:

            def _run_era_genre_async() -> None:
                """Hintergrund-Thread für Era+Genre (hat ROCm-Kontext)."""
                for _name in ("era", "genre"):
                    if _name not in _clap_steps:
                        continue
                    try:
                        setattr(result, _name, _clap_steps[_name]())
                        logger.info("pre_Analyse: step=%s done (async)", _name)
                    except Exception as _exc:
                        result.errors[_name] = str(_exc)
                        logger.warning("pre_Analyse: step=%s fehlgeschlagen (%s)", _name, _exc)
                    # Update progress via callback
                    nonlocal _done_steps
                    _done_steps += 1
                    if _done_steps <= _total_steps:
                        _pct = 75 + int((_done_steps / max(_total_steps, 1)) * 15)
                        _cb(_pct, f"Analyse: {_name} abgeschlossen ({_done_steps}/{_total_steps})…")

            _era_thread = threading.Thread(target=_run_era_genre_async, daemon=True, name="aurik-era-genre")
            _era_thread.start()
            # Markiere als "pending" — der Rest der Analyse läuft ohne Era/Genre weiter
            for _name in _clap_steps:
                if getattr(result, _name, None) is None and _name not in result.errors:
                    pass  # Will be set by async thread

        # Phase 1: Submit non-CLAP steps to pool → start immediately
        _pool = None
        _other_futs: dict[_cf.Future, str] = {}
        if _other_steps:
            _pool = _cf.ThreadPoolExecutor(max_workers=len(_other_steps))
            for name, fn in _other_steps.items():
                _other_futs[_pool.submit(fn)] = name

            # Phase 2: Collect pool results via as_completed
            try:
                _total_timeout = _SUBSTEP_TIMEOUT_S * len(_other_futs)
                for fut in _cf.as_completed(_other_futs, timeout=_total_timeout):
                    name = _other_futs[fut]
                    try:
                        setattr(result, name, fut.result(timeout=0.0))
                    except Exception as exc:
                        result.errors[name] = str(exc)
                        logger.warning("pre_Analyse: step=%s fehlgeschlagen (%s)", name, exc)
                    _done_steps += 1
                    _step_pct = 75 + int((_done_steps / max(_total_steps, 1)) * 15)
                    _cb(_step_pct, f"Analyse: {name} abgeschlossen ({_done_steps}/{_total_steps})…")
                    logger.info("pre_Analyse: step=%s done (%d/%d)", name, _done_steps, _total_steps)
            except (_cf.TimeoutError, TimeoutError):
                for fut, name in _other_futs.items():
                    if not fut.done():
                        result.errors[name] = f"timeout_after={_SUBSTEP_TIMEOUT_S:.1f}s"
                        fut.cancel()
                        logger.warning("pre_Analyse: step=%s timed out", name)
            finally:
                _pool.shutdown(wait=False, cancel_futures=True)
    else:
        logger.debug("pre_Analyse: steps 2-5 vollständig aus Zwischenspeicher geladen")

    _cb(90, "Analyse abgeschlossen — Ergebnisse werden gespeichert…")

    # ── Bidirektionale Genre↔Medium-Validierung (SOTA 2026) ──────
    # Nutzt die Knowledge Base des MediumDetectors um Genre und
    # Tonträgerkette gegenseitig zu validieren.  Schellack → kein
    # Hip-Hop.  Deutscher Schlager → kein Streaming-Only.
    if result.medium is not None and result.genre is not None:
        try:
            _md_val = result.medium
            _genre_label = str(getattr(result.genre, "genre_label", "") or "")
            _lang_code = str(getattr(result.genre, "language_code", "") or getattr(result.genre, "lang_code", "") or "")
            _chain = list(getattr(_md_val, "transfer_chain", []) or [])

            if _chain and _genre_label:
                _detector = cast(Callable[[], Any], _load_symbol("forensics.medium_detector", "get_medium_detector"))()

                # 1. Medium → Genre: Sind die erkannten Medien mit dem Genre vereinbar?
                _constraints = _detector.get_genre_constraints(_chain)
                _excluded = set(_constraints.get("excluded", []))
                _preferred = set(_constraints.get("preferred", []))
                _genre_key = _genre_label.lower().replace(" ", "_").replace("-", "_")

                if _genre_key in _excluded:
                    logger.warning(
                        "Bidirektionale Validierung: Genre '%s' ist auf Tonträgerkette %s "
                        "AUSGESCHLOSSEN. Medium→Genre-Konflikt.",
                        _genre_label,
                        " → ".join(_chain),
                    )
                elif _genre_key in _preferred:
                    logger.debug(
                        "Bidirektionale Validierung: Genre '%s' passt zur Kette %s (preferred)",
                        _genre_label,
                        " → ".join(_chain),
                    )
            _cb(93, "Tonträgerkette wird validiert…")

            # 2. Genre + Sprache → Kette: Chain mit erweiterten Parametern neu matchen
            # Sort detected materials chronologically (reel_tape before cassette etc.)
            _detected = sorted(set(_chain), key=lambda m: _detector._MEDIUM_ORDER.get(m, 99))
            _refined_chain = _detector._best_matching_chain(_detected, genre=_genre_label, language=_lang_code or None)
            if _refined_chain and _refined_chain != _chain:
                logger.info(
                    "Bidirektionale Validierung: Kette verfeinert — %s → %s (Genre=%s, Sprache=%s)",
                    " → ".join(_chain),
                    " → ".join(_refined_chain),
                    _genre_label,
                    _lang_code or "?",
                )
                _md_val.transfer_chain = _refined_chain  # type: ignore[attr-defined]
        except Exception as _bv_exc:
            logger.debug("Bidirektionale Validierung uebersprungen: %s", _bv_exc)

    # ── Cross-Validation: Multi-Factor Consensus Check ────────────
    # Vergleicht alle unabhängigen Evidenz-Quellen auf Konsistenz.
    # Konflikte → WARNING, Übereinstimmung → Confidence-Boost.
    if result.medium is not None:
        try:
            _cv_chain = list(getattr(result.medium, "transfer_chain", []) or [])
            _cv_confidence = float(getattr(result.medium, "confidence", 0.0) or 0.0)
            _cv_conflicts = []
            _cv_agreements = []

            # Factor 1: Era material_prior vs chain
            if result.era is not None:
                _era_mat = str(getattr(result.era, "material_prior", "") or "")
                if _era_mat and _era_mat != "unknown":
                    if _era_mat in _cv_chain:
                        _cv_agreements.append(f"Era({_era_mat})")
                    else:
                        _cv_conflicts.append(f"Era({_era_mat}) not in chain")

            # Factor 2: Genre-era compatibility vs chain-era
            if result.genre is not None and _cv_chain:
                _genre_label = str(getattr(result.genre, "genre_label", "") or "")
                if _genre_label:
                    try:
                        _detector_cv = cast(
                            Callable[[], Any], _load_symbol("forensics.medium_detector", "get_medium_detector")
                        )()
                        _constraints_cv = _detector_cv.get_genre_constraints(_cv_chain)
                        _excluded_cv = set(_constraints_cv.get("excluded", []))
                        _genre_key_cv = _genre_label.lower().replace(" ", "_").replace("-", "_")
                        if _genre_key_cv in _excluded_cv:
                            _cv_conflicts.append(f"Genre({_genre_label}) excluded by chain")
                        else:
                            _cv_agreements.append(f"Genre({_genre_label})")
                    except Exception as _cv_exc:
                        logger.debug(
                            "pre_Analyse: genre-chain cross-Validierung fehlgeschlagen (unkritisch): %s", _cv_exc
                        )

            # Factor 3: Defect scanner material vs chain
            if result.defects is not None:
                _def_mat_raw = getattr(result.defects, "material_type", None)
                # §v10.304.16: Enum.value statt str(Enum) → "mp3_high" statt "MaterialType.MP3_HIGH"
                if hasattr(_def_mat_raw, "value"):
                    _def_mat = str(_def_mat_raw.value)  # type: ignore[union-attr]
                else:
                    _def_mat = str(_def_mat_raw or getattr(result.defects, "auto_detected_material", "") or "")
                if _def_mat and _def_mat != "unknown":
                    if _def_mat in _cv_chain or any(_def_mat in m for m in _cv_chain):
                        _cv_agreements.append(f"Defect({_def_mat})")
                    else:
                        _cv_conflicts.append(f"Defect({_def_mat}) not in chain")

            # Report — §v10.370: Auf INFO gesenkt (vorher WARNING).
            # Cross-Validation läuft VOR der Chain-Injection — Konflikte
            # zwischen initialer Kette und Era/Defect sind ERWARTET.
            # Die Post-Injection-Cross-Validation (unten) prüft die finale Kette.
            if _cv_conflicts:
                logger.info(
                    "Cross-Validierung (pre-injection): %d Konflikt(e) — %s. Übereinstimmungen: %s. Confidence=%.2f",
                    len(_cv_conflicts),
                    ", ".join(_cv_conflicts),
                    ", ".join(_cv_agreements) if _cv_agreements else "keine",
                    _cv_confidence,
                )
            elif _cv_agreements:
                # Boost confidence when multiple independent factors agree
                _boost = min(0.15, len(_cv_agreements) * 0.05)
                logger.info(
                    "Cross-Validierung: %d Faktoren stimmen überein (%s). Confidence %.2f → %.2f",
                    len(_cv_agreements),
                    ", ".join(_cv_agreements),
                    _cv_confidence,
                    min(1.0, _cv_confidence + _boost),
                )
        except Exception as _cv_exc:
            logger.debug("Cross-Validierung uebersprungen: %s", _cv_exc)

    _cb(96, "Kette wird rekonstruiert…")

    # ── §2.46a Deep-Transfer-Chain-Injection [RELEASE_MUST] ───────────
    # Spec §2.46a: Importsongs mit 3+ Tonträgerstufen müssen vollständig
    # modelliert werden. Drei Quellen für die Ketten-Rekonstruktion:
    #   1. EraClassifier → inhaltsbasiertes Original-Medium
    #   2. DefectScanner → physikalische Defekte → Material
    #   3. MediumDetector → physical_analog_sources
    if result.medium is not None:
        try:
            _md = result.medium
            _chain = list(getattr(_md, "transfer_chain", []) or [])

            _era_material = None
            if result.era is not None:
                _era_material = str(getattr(result.era, "material_prior", "") or "")

            _defect_material = None
            if result.defects is not None and hasattr(result.defects, "material_type"):
                _dm = str(getattr(result.defects, "material_type", "")).lower()
                _defmap = {
                    "cassette": "cassette",
                    "vinyl": "vinyl",
                    "shellac": "shellac",
                    "tape": "tape",
                    "reel_tape": "reel_tape",
                    "reel": "reel_tape",
                    "cd_digital": "cd_digital",
                    "dat": "dat",
                }
                _defect_material = _defmap.get(_dm)
                # §2.46a: Wenn der DefectScanner ein anderes Material auto-detektiert
                # hat als der Hint, das auto-detektierte Material für die Kette verwenden.
                # §v10.14: ABER nur wenn der Hint "unknown" ist — MediumDetector ist
                # autoritativ für Material-Klassifikation, DefectScanner nur supplementär.
                # Bei digitalem Endformat (mp3_high etc.) ist die DefectScanner-Heuristik
                # unzuverlässig (Rumble/Noise vom Codec, nicht vom Träger).
                _chain_primary = _chain[0] if _chain else "unknown"
                _md_has_material = _chain_primary not in ("unknown", "")
                _auto_dm = getattr(result.defects, "auto_detected_material", None)
                if _auto_dm is not None and not _md_has_material:
                    _adm = str(_auto_dm).lower()
                    for _suffix in [
                        ".cassette",
                        ".vinyl",
                        ".reel_tape",
                        ".tape",
                        ".shellac",
                        ".lacquer_disc",
                        ".wire_recording",
                        ".wax_cylinder",
                    ]:
                        if _adm.endswith(_suffix):
                            _adm = _suffix[1:]
                            break
                    _adm_mapped = _defmap.get(_adm)
                    if _adm_mapped and _adm_mapped != _defect_material:
                        logger.info(
                            "pre_Analyse: DefectScanner auto-erkannt %s (overrides hint %s)",
                            _adm_mapped,
                            _defect_material or "none",
                        )
                        _defect_material = _adm_mapped

            # §v10.14 §fix: era/defect scores computed here (previously undefined).
            _era_decade = int(getattr(result.era, "decade", 0) or 0)
            _era_confidence = float(getattr(result.era, "confidence", 0.0) or 0.0)
            # Aggregate defect severity across all detected defects as score.
            _defect_score: float = 0.0
            if result.defects is not None and hasattr(result.defects, "scores"):
                _ds = getattr(result.defects, "scores", {})
                _sevs = [float(getattr(s, "severity", 0.0))
                         for s in (_ds.values() if isinstance(_ds, dict) else [])]
                _defect_score = sum(_sevs) if _sevs else 0.0

            _physical = list(getattr(_md, "physical_analog_sources", []) or [])
            _analog = {
                "shellac",
                "wax_cylinder",
                "vinyl",
                "tape",
                "reel_tape",
                "cassette",
                "lacquer_disc",
                "wire_recording",
            }

            # Kette bauen: neue Stufen VOR der digitalen Stufe einfügen.
            # §2.46a: _era_material ist das ORIGINAL-Aufnahmemedium und gehört
            # an den ANFANG der Kette. _defect_material ist ein Zwischenträger
            # und gehört VOR die digitale Stufe. physical_analog_sources werden
            # ebenfalls VOR der digitalen Stufe eingefügt.

            # §v10.304.14: Multi-Carrier-Inferenz aus Defect-Signaturen.
            # Der DefectScanner erkennt Defekte die spezifisch für bestimmte
            # Tonträger sind. Diese Signale werden genutzt um ZUSÄTZLICHE
            # Träger in der Kette zu inferieren — nicht nur den dominanten.
            _defect_inferred_carriers: list[str] = []
            # §v10.14: Per-Material Defekt-Severity-Aggregation für Material-Konsens.
            _defect_carrier_scores: dict[str, float] = {}
            if result.defects is not None and hasattr(result.defects, "scores"):
                _defect_scores = getattr(result.defects, "scores", {})
                # Defekt → Träger-Mapping mit Schwellwerten
                _DEFECT_CARRIER_MAP: dict[str, tuple[str, float]] = {
                    "crackle": ("vinyl", 0.35),
                    "groove_echo": ("vinyl", 0.30),
                    "inner_groove_distortion": ("vinyl", 0.40),
                    "riaa_curve_error": ("vinyl", 0.30),
                    "tape_hiss": ("cassette", 0.25),
                    "wow": ("cassette", 0.20),
                    "flutter": ("cassette", 0.20),
                    "multiband_wow_flutter": ("cassette", 0.25),
                    "print_through": ("reel_tape", 0.30),
                    "tape_head_level_dip": ("reel_tape", 0.25),
                    "low_freq_rumble": ("vinyl", 0.30),
                    "soft_saturation": ("reel_tape", 0.30),
                    "quantization_noise": ("cd_digital", 0.30),
                    "compression_artifacts": ("mp3_high", 0.20),
                }
                for _defect_name, (_carrier, _threshold) in _DEFECT_CARRIER_MAP.items():
                    for _score_key, _score_obj in _defect_scores.items():
                        _sk_name = _score_key.value if hasattr(_score_key, "value") else str(_score_key)
                        if _sk_name == _defect_name:
                            _sev = float(getattr(_score_obj, "severity", 0.0))
                            if (
                                _sev >= _threshold
                                and _carrier not in _chain
                                and _carrier not in _defect_inferred_carriers
                            ):
                                _defect_inferred_carriers.append(_carrier)
                                logger.debug(
                                    "§v10.304.14 Defect-Carrier: %s(sev=%.2f) → %s",
                                    _defect_name,
                                    _sev,
                                    _carrier,
                                )
                            # §v10.14: Aggregiere Severity pro Carrier (auch unterhalb der Schwelle,
                            # für gewichtete Material-Affinitäts-Berechnung im Konsens).
                            _defect_carrier_scores[_carrier] = _defect_carrier_scores.get(_carrier, 0.0) + _sev
                            break
            if _defect_inferred_carriers:
                logger.info(
                    "§v10.304.14 Multi-Carrier-Inferenz: %d zusätzliche Träger aus Defekt-Signaturen (%s)",
                    len(_defect_inferred_carriers),
                    " → ".join(_defect_inferred_carriers),
                )
                # Sortiere nach chronologischer Reihenfolge (älteste zuerst)
                _CARRIER_ORDER = {
                    "reel_tape": 0,
                    "vinyl": 1,
                    "cassette": 2,
                    "cd_digital": 3,
                    "mp3_high": 4,
                    "mp3_low": 5,
                }
                _defect_inferred_carriers.sort(key=lambda c: _CARRIER_ORDER.get(c, 99))

            _era_injected = None
            _chain_injected: list[str] = []
            # §v10.304.14: Defect-inferierte Träger ZUERST einfügen
            for _src in _defect_inferred_carriers:
                if _src and _src in _analog and _src not in _chain and _src not in _chain_injected:
                    _chain_injected.append(_src)
            # §v10.307: Digitale Träger aus Defekt-Signaturen (mp3_high etc.)
            # separat sammeln — ersetzen "unknown" am Kettenende.
            _digital_defect_carriers: list[str] = []
            for _src in _defect_inferred_carriers:
                if _src and _src not in _analog and _src not in _digital_defect_carriers:
                    _digital_defect_carriers.append(_src)
            for _src in [_defect_material]:  # type: ignore[assignment]
                if _src and _src in _analog and _src not in _chain and _src not in _chain_injected:
                    _chain_injected.append(_src)
            for _ps_mat, _ps_conf in _physical:
                _k = str(_ps_mat).lower().replace(" ", "_")
                if _k in _analog and _k not in _chain and _k not in _chain_injected:
                    _chain_injected.append(_k)
            # Era-Material separat: Original-Aufnahmemedium → Position 0
            if _era_material and _era_material in _analog and _era_material not in _chain:
                _era_injected = _era_material

            _any_injected = bool(_chain_injected) or _era_injected is not None
            if _any_injected:
                if _era_injected is not None:
                    _chain.insert(0, _era_injected)
                    # §v10.306: Era-Material aus chain_injected entfernen —
                    # verhindert Doppeleintrag wenn DefectScanner das gleiche
                    # Medium erkennt (z.B. vinyl als era_material UND
                    # defect_inferred_carrier). Prüft auf ALLE Vorkommen
                    # (list.remove() stoppt nach erstem Treffer).
                    _chain_injected = [c for c in _chain_injected if c != _era_injected]
                if _chain_injected:
                    _dpos = len(_chain)
                    for i, m in enumerate(_chain):
                        if m in {"mp3_low", "mp3_high", "cd_digital", "streaming", "aac", "unknown"}:
                            _dpos = i
                            break
                    for _m in reversed(_chain_injected):
                        _chain.insert(_dpos, _m)
                _injected = ([_era_injected] if _era_injected else []) + _chain_injected

                # ── §2.46a Vinyl-Inference ─────────────────────────
                # Wenn reel_tape + cassette in der Kette sind und die
                # Ära in der Vinyl-Ära liegt (1950–1990), war die
                # Veröffentlichung mit hoher Wahrscheinlichkeit auf
                # Vinyl. Kein physikalisches Risiko — nur logische
                # Inferenz ohne Audio-Veränderung.
                _has_reel = "reel_tape" in _chain
                _has_cassette = "cassette" in _chain
                _has_vinyl = "vinyl" in _chain
                _vinyl_era = result.era is not None and 1950 <= getattr(result.era, "decade", 0) <= 1990
                if _has_reel and _has_cassette and not _has_vinyl and _vinyl_era:
                    _vi = _chain.index("cassette")
                    _chain.insert(_vi, "vinyl")
                    logger.info("pre_Analyse: Vinyl-Inference — reel_tape+cassette+vinyl-era → vinyl eingefügt")

                # §v10.14 Lacquer-Disc-Inference: Jede Vinylpressung entsteht
                # aus einer Lackfolie (Schneidstichel → Lack/Alu → Galvanik → Stempel).
                # Wenn reel_tape UND vinyl in der Kette sind, MUSS lacquer_disc
                # dazwischen liegen — der physische Zwang des Pressprozesses.
                _has_lacquer = "lacquer_disc" in _chain
                if _has_reel and _has_vinyl and not _has_lacquer:
                    _ld_pos = _chain.index("vinyl")
                    _chain.insert(_ld_pos, "lacquer_disc")
                    logger.info("pre_Analyse: Lacquer-Disc-Inference — reel_tape+vinyl → lacquer_disc eingefügt")

                _md.is_multi_generation = len(_chain) > 1  # type: ignore[attr-defined]
                _analog_in = [m for m in _chain if m in _analog]
                if _analog_in:
                    _md.primary_material = _analog_in[-1]  # type: ignore[attr-defined]
                # Chronological sort after all injections (§v10.306: robust, kein Singleton-Try)
                # 1930+++++1950+++++1960+++++1980+++++1990+++++++++2000++
                # MUST run BEFORE _md.transfer_chain assignment — §v10.307 Bugfix:
                # transfer_chain wurde VOR dem Sort gesetzt, GUI zeigte unsortierte Kette.
                if len(_chain) > 1:
                    _TIMELINE: dict[str, int] = {
                        "wax_cylinder": 0,
                        "lacquer_disc": 5,  # §v10.440: 1→5 — Lacquer ist VINYL-Master, wird VOM Tape geschnitten
                        "shellac": 2,
                        "wire_recording": 3,
                        "reel_tape": 4,
                        "vinyl": 6,
                        "tape": 6,
                        "cassette": 7,
                        "cartridge_8track": 8,
                        "cd_digital": 9,
                        "dat": 10,
                        "minidisc": 11,
                        "mp3_high": 12,
                        "mp3_low": 13,
                        "aac": 13,
                        "streaming": 14,
                    }
                    _sorted = sorted(_chain, key=lambda m: _TIMELINE.get(m, 99))
                    if _sorted != _chain:
                        logger.info("pre_Analyse: chain sorted: %s → %s", " → ".join(_chain), " → ".join(_sorted))
                        _chain = _sorted

                # §v10.307: Digitale Defekt-Träger am Kettenende einsetzen.
                # "unknown" stammt vom MediumDetector wenn kein Codec erkannt wurde.
                # Der DefectScanner hat aber oft mp3_high/mp3_low erkannt → ersetzen.
                if _digital_defect_carriers and _chain and _chain[-1] == "unknown":
                    _digital = _digital_defect_carriers[0]
                    _chain[-1] = _digital
                    logger.info("pre_Analyse: 'unknown' → '%s' (aus DefectScanner)", _digital)

                _md.transfer_chain = _chain  # type: ignore  # §v10.307: NACH dem Sort setzen

                # §v10.712: Chain-Depth-Confidence-Guard.
                # Wenn der MediumDetector nur geringe Konfidenz hat (z.B. 0.41),
                # dürfen daraus abgeleitete Zusatzträger (defect_inferred_carriers)
                # die Kette nicht aufblähen. Ein unsicheres "lacquer_disc → vinyl → mp3_low"
                # wird sonst zu "reel_tape → lacquer_disc → vinyl → cassette → mp3_low"
                # aufgebläht — und die aggressive chain_depth=5 zerstört dann
                # tonal_center und timbre im eigentlich sauberen 320kbps MP3.
                # §v10.14 Era-Plausibilität: Materialien deren Ära VOR dem Song-Ende
                # liegt (z.B. Shellack 1890-1950 in einem 1977er Song) werden
                # ausgefiltert. Verhindert "shellac → reel_tape" bei 1977er Vinyl-Song.
                _MATERIAL_ERA_END: dict[str, int] = {
                    "wax_cylinder": 1920,
                    "shellac": 1955,
                    "lacquer_disc": 1990,  # §v10.14: Als PRESSWERK-Zwischenträger bis Ende Vinyl-Ära
                    "wire_recording": 1945,
                }
                if _era_decade and _era_decade > 0:
                    _pre_filter_chain = list(_chain)
                    _chain = [
                        m for m in _chain
                        if _MATERIAL_ERA_END.get(m, 9999) >= _era_decade - 10
                    ]
                    _removed = len(_pre_filter_chain) - len(_chain)
                    if _removed > 0:
                        logger.info(
                            "§v10.14 Era-Filter: %d anachronistische Materialien entfernt (Ära=%d): %s",
                            _removed,
                            _era_decade,
                            ", ".join(set(_pre_filter_chain) - set(_chain)),
                        )
                _md_confidence = float(getattr(_md, "confidence", 0.5) or 0.5)
                _max_chain_depth = 2 if _md_confidence < 0.50 else (3 if _md_confidence < 0.60 else 99)
                if len(_chain) > _max_chain_depth:
                    # §v10.14: Letzten Eintrag (Endformat, z.B. mp3_high) IMMER behalten.
                    # Aus den analogen Zwischenträgern den Ära-plausibelsten wählen.
                    if len(_chain) > 1 and _chain[-1] not in (
                        "shellac", "wax_cylinder", "vinyl", "cassette",
                        "reel_tape", "tape", "lacquer_disc", "wire_recording",
                    ):
                        _last = [_chain[-1]]
                        _analog_candidates = _chain[:-1]
                        # §v10.14: Ära-bewusste Auswahl — Material das zur Ära passt bevorzugen
                        _MATERIAL_ERA_PEAK: dict[str, int] = {
                            "shellac": 1930, "wax_cylinder": 1900, "lacquer_disc": 1965,
                            "wire_recording": 1940, "reel_tape": 1965, "vinyl": 1975,
                            "cassette": 1985, "tape": 1970,
                        }
                        if _era_decade and _era_decade > 0:
                            _analog_candidates.sort(
                                key=lambda m: abs(_MATERIAL_ERA_PEAK.get(m, 1970) - _era_decade)
                            )
                        _middle = _analog_candidates[:_max_chain_depth - 1]
                        _trimmed = _middle + _last
                    else:
                        _trimmed = _chain[:_max_chain_depth]
                    logger.info(
                        "§v10.712 Chain-Depth-Cap: confidence=%.2f → chain von %d auf %d Träger gekürzt (%s → %s)",
                        _md_confidence,
                        len(_chain),
                        len(_trimmed),
                        " → ".join(_chain),
                        " → ".join(_trimmed),
                    )
                    _chain = _trimmed
                    _md.transfer_chain = _chain  # type: ignore[attr-defined]

                logger.info(
                    "pre_Analyse: Deep-Transfer-Chain: %s (injected=%s, era=%s, defect=%s)",
                    " → ".join(_chain),
                    ",".join(_injected) if _injected else "none",
                    _era_material or "none",
                    _defect_material or "none",
                )

                # ── §v10.20 Material-Konsens: 3 Detektoren abgleichen + Kette korrigieren ──
                try:
                    from backend.core.material_consensus import resolve_material_consensus, validate_material_era_consistency

                    _consensus = resolve_material_consensus(
                        medium_result={"material": _chain[-1] if _chain else "unknown", "confidence": _md_confidence, "chain": " → ".join(_chain)},
                        era_result={"material": _era_material, "decade": _era_decade, "confidence": _era_confidence} if _era_material else None,
                        defect_result={"material": _defect_material, "score": _defect_score,
                                      "material_scores": _defect_carrier_scores} if _defect_material else None,
                    )

                    if _consensus["conflict_detected"]:
                        logger.warning("pre_Analyse: Material-KONFLIKT — gewählter Konsens: %s (%.2f)",
                                       _consensus["material"], _consensus["confidence"])
                        # Korrigiere die Kette mit allen Detektor-Ergebnissen
                        _all_materials = []
                        for _det, _info in _consensus["all_votes"].items():
                            _mat = _info.get("material", "unknown")
                            if _mat and _mat != "unknown" and _mat not in _all_materials:
                                _all_materials.append(_mat)
                        if len(_all_materials) > 1:
                            _era_order = ["shellac", "wax_cylinder", "vinyl", "lacquer_disc", "reel_tape",
                                         "cassette", "dat", "cd", "minidisc", "mp3", "mp3_low", "mp3_high", "streaming"]
                            _all_materials.sort(key=lambda m: _era_order.index(m) if m in _era_order else 99)
                            _chain = _all_materials
                            _md.transfer_chain = _chain  # type: ignore[attr-defined]
                            logger.info("pre_Analyse: Kette KORRIGIERT: %s", " → ".join(_chain))

                    # Ära und Kette sind KOMPLEMENTÄR, nicht widersprüchlich.
                    # Ära = Aufnahmedatum. Kette = gesamte Medien-Historie.
                    # Ein Song von 1960 kann selbstverständlich als MP3 vorliegen.
                    # Keine Ära-Korrektur nötig — die Kette enthält bereits alle Infos.
                except Exception:
                    pass
        except Exception as _inj_exc:
            logger.debug("Deep-Transfer-Chain-Injection uebersprungen: %s", _inj_exc)

    # ── §v10.304.13: Era-Re-Klassifikation mit angereicherter Chain ────────
    # Wenn die Deep-Transfer-Chain-Injection die Kette erweitert hat (depth≥3),
    # muss die Era-Klassifikation mit der VOLLEN Chain wiederholt werden.
    # Der originale EraClassifier-Lauf (asynchron) hatte nur die flache Chain.
    # Warte kurz auf den async Era-Thread, dann re-klassifiziere mit voller Chain.
    if result.medium is not None:
        _enriched_chain = list(getattr(result.medium, "transfer_chain", []) or [])
        if len(_enriched_chain) >= 3:
            # Warte bis zu 10s auf async Era-Thread
            _era_thread_waited = False
            try:
                _era_thread  # noqa: B018 — prüft ob Variable existiert
                if _era_thread is not None and _era_thread.is_alive():
                    _era_thread.join(timeout=10.0)
                    _era_thread_waited = True
            except NameError:
                pass  # _era_thread wurde nie erstellt (Era/Gerne aus Cache)
            if result.era is not None:
                _orig_confidence = float(getattr(result.era, "confidence", 0.0))
                if _orig_confidence < 0.65:
                    try:
                        _classify_era = cast(
                            Callable[..., Any],
                            _load_symbol("backend.api.bridge", "get_era_classifier_fn"),
                        )
                        _era_re = _classify_era()(audio_native, sr_native, transfer_chain=_enriched_chain)
                        _new_decade = int(getattr(_era_re, "decade", 0))
                        _old_decade = int(getattr(result.era, "decade", 0))
                        if _new_decade != _old_decade:
                            logger.info(
                                "§v10.304.13 Era-Re-Klassifikation: chain_depth=%d → "
                                "era %d→%d (confidence %.2f→%.2f, waited=%s)",
                                len(_enriched_chain),
                                _old_decade,
                                _new_decade,
                                _orig_confidence,
                                float(getattr(_era_re, "confidence", 0.0)),
                                "yes" if _era_thread_waited else "no",
                            )
                            result.era = _era_re
                    except Exception as _era_re_exc:
                        logger.debug("Era-Re-Klassifikation fehlgeschlagen: %s", _era_re_exc)
            else:
                # Era ist None (async noch nicht fertig oder fehlgeschlagen) →
                # direkt mit voller Chain klassifizieren
                try:
                    _classify_era = cast(
                        Callable[..., Any],
                        _load_symbol("backend.api.bridge", "get_era_classifier_fn"),
                    )
                    result.era = _classify_era()(audio_native, sr_native, transfer_chain=_enriched_chain)
                    _new_decade = int(getattr(result.era, "decade", 0))
                    logger.info(
                        "§v10.304.13 Era-Direktklassifikation: chain_depth=%d → era=%d (confidence=%.2f, era_was_None)",
                        len(_enriched_chain),
                        _new_decade,
                        float(getattr(result.era, "confidence", 0.0)),
                    )
                except Exception as _era_re_exc:
                    logger.debug("Era-Direktklassifikation fehlgeschlagen: %s", _era_re_exc)

    # ------------------------------------------------------------------
    # Store in bridge cache so UV3 never re-runs classifiers
    # ------------------------------------------------------------------
    _cb(99, "Ergebnisse werden gespeichert…")
    if store_in_bridge_cache and file_path:
        _store_in_cache(file_path, result)

    result.elapsed_seconds = time.monotonic() - t0
    # §v10.330: Post-Injection Cross-Validation.
    # Die erste Cross-Validation lief VOR der Chain-Injection (Zeile ~473) und
    # verglich die unvollständige Kette. Jetzt, nach Deep-Transfer-Chain-Injection
    # und Era-Re-Klassifikation, ist die finale Kette verfügbar. Erst JETZT
    # sind Konflikte zwischen Era/Defect/Genre und Chain aussagekräftig.
    if result.medium is not None and getattr(result.medium, "is_multi_generation", False):
        try:
            _final_chain = list(getattr(result.medium, "transfer_chain", []) or [])
            _era_material = str(getattr(result.era, "material_prior", "") or "")
            _genre_label = str(getattr(result.genre, "genre_label", "") or "")
            if _era_material and _era_material != "unknown" and _era_material not in _final_chain:
                logger.info(
                    "pre_Analyse: Post-Injection-Era-Pruefung — era=%s now in chain=%s ✓",
                    _era_material,
                    " → ".join(_final_chain),
                )
        except Exception:
            logger.debug("pre_Analyse.py:879: Silent exception absorbed", exc_info=True)
    logger.info("pre_Analyse: vollstaendig in %.1fs (errors=%s)", result.elapsed_seconds, list(result.errors))

    # Free DefectScanner STFT/spectral intermediate arrays (30 defect types × full audio).
    # Vollstaendiges GC ist hier sicher; malloc_trim(0) bleibt bewusst deaktiviert,
    # weil der Aufruf im Projekt bereits mehrfach als SIGABRT-Risiko unter
    # konkurrierenden Audio-/NumPy-Threads aufgefallen ist.
    gc.collect()

    _cb(100, "Voranalyse fertig.")
    return result


def _store_in_cache(file_path: str, result: PreAnalysisResult) -> None:
    """Store all sub-results in bridge LRU caches."""
    try:
        cache_defect_result = cast(
            Callable[[str, object], None],
            _load_symbol("backend.api.bridge", "cache_defect_result"),
        )
        cache_era_genre_result = cast(
            Callable[..., None],
            _load_symbol("backend.api.bridge", "cache_era_genre_result"),
        )
        cache_medium_result = cast(
            Callable[[str, object], None],
            _load_symbol("backend.api.bridge", "cache_medium_result"),
        )
        cache_restorability_result = cast(
            Callable[[str, object], None],
            _load_symbol("backend.api.bridge", "cache_restorability_result"),
        )

        if result.medium is not None:
            cache_medium_result(file_path, result.medium)

        if result.era is not None or result.genre is not None:
            cache_era_genre_result(
                file_path,
                era_result=result.era,
                genre_result=result.genre,
            )

        if result.defects is not None:
            cache_defect_result(file_path, result.defects)

        if result.restorability is not None:
            cache_restorability_result(file_path, result.restorability)

        logger.debug("pre_Analyse: bridge Zwischenspeicher updated for %s", file_path)
    except Exception as exc:
        logger.warning("pre_Analyse: bridge Zwischenspeicher store fehlgeschlagen (%s)", exc)


def _load_cached_parts(file_path: str) -> dict[str, object | None]:
    """Lädt verfügbare Bridge-Cache-Teilergebnisse best-effort."""
    try:
        get_cached_defect_result = cast(
            Callable[[str], object | None],
            _load_symbol("backend.api.bridge", "get_cached_defect_result"),
        )
        get_cached_era_genre_result = cast(
            Callable[[str], dict[str, object] | None],
            _load_symbol("backend.api.bridge", "get_cached_era_genre_result"),
        )
        get_cached_medium_result = cast(
            Callable[[str], object | None],
            _load_symbol("backend.api.bridge", "get_cached_medium_result"),
        )
        get_cached_restorability_result = cast(
            Callable[[str], object | None],
            _load_symbol("backend.api.bridge", "get_cached_restorability_result"),
        )

        era_genre = get_cached_era_genre_result(file_path)
        era_result = era_genre.get("era_result") if isinstance(era_genre, dict) else None
        genre_result = era_genre.get("genre_result") if isinstance(era_genre, dict) else None
        return {
            "medium": get_cached_medium_result(file_path),
            "era": era_result,
            "genre": genre_result,
            "defects": get_cached_defect_result(file_path),
            "restorability": get_cached_restorability_result(file_path),
        }
    except Exception as exc:
        logger.debug("pre_Analyse: Zwischenspeicher part laden nicht blockierend (%s)", exc)
        return {}


def _load_from_cache(file_path: str, sr_native: int) -> PreAnalysisResult | None:
    """Lädt ein vollständiges PreAnalysisResult aus Bridge-Caches, falls vorhanden.

    Returns ``None``, wenn mindestens ein Pflicht-Subresultat fehlt.
    """
    try:
        _parts = _load_cached_parts(file_path)
        return _build_result_from_cached_parts(_parts, sr_native=sr_native, file_path=file_path)
    except Exception as exc:
        logger.debug("pre_Analyse: Zwischenspeicher laden nicht blockierend (%s)", exc)
        return None


def _build_result_from_cached_parts(
    parts: dict[str, object | None],
    *,
    sr_native: int,
    file_path: str,
) -> PreAnalysisResult | None:
    """Erzeugt ein vollständiges PreAnalysisResult nur bei vollständig belegten Cache-Parts."""
    medium = parts.get("medium")
    era = parts.get("era")
    genre = parts.get("genre")
    defects = parts.get("defects")
    restorability = parts.get("restorability")
    if medium is None or era is None or genre is None or defects is None or restorability is None:
        return None

    return PreAnalysisResult(
        medium=medium,
        era=era,
        genre=genre,
        defects=defects,
        restorability=restorability,
        native_sr=sr_native,
        file_path=file_path,
    )
