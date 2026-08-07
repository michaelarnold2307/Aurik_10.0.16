"""Global ML Memory Budget — prevents OOM-kills by capping total ML model RAM.

Problem: 35+ GB model files on a 32 GB system. Without a global cap, individual
plugin loaders each check free RAM independently and all see "enough space", then
load their models sequentially until the kernel OOM-killer fires.

Solution: Centralized Thread-safe budget singleton. Every heavy ML model loader
calls ``try_allocate()`` before loading. Once the budget is exhausted, remaining
models fall back to DSP. Budget is set to ``ML_MAX_GB`` (default 16 GB), leaving
~16 GB free for OS + app + audio processing buffers on a 32 GB machine.

Usage in a plugin loader::

    from backend.core.ml_memory_budget import try_allocate, release

    if not try_allocate("AudioSR", size_gb=7.0):
        return None           # → DSP fallback
    try:
        model = load_heavy_model(...)
    except Exception:
        release("AudioSR", size_gb=7.0)   # refund on failure
        raise
"""

from __future__ import annotations

import logging
import threading
import time

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

logger = logging.getLogger(__name__)


def _tune_glibc_malloc_mmap_threshold() -> None:
    """§v10.307 1.1: Arena-Bloat-Fix — arrays > 2 MB via mmap statt Heap-Arena.

    Ohne diesen Tune hält glibc `malloc` freigegebene Pages großer numpy-Arrays im
    Prozess-Heap zurück (Default `M_MMAP_THRESHOLD=128KB` ist für Audio-Buffer im
    MB-Bereich ungeeignet) — Arena fragmentiert über Stunden bis VMS explodiert.
    Ab 2 MB werden Allokationen stattdessen per `mmap` bedient und geben bei `del`
    sofort via `munmap` an das OS zurück. Best-effort, nur Linux/glibc.
    """
    try:
        import ctypes
        import platform

        if platform.system() != "Linux":
            return
        _M_MMAP_THRESHOLD = -3
        _TWO_MB = 2 * 1024 * 1024
        libc = ctypes.CDLL("libc.so.6")
        if not libc.mallopt(ctypes.c_int(_M_MMAP_THRESHOLD), ctypes.c_int(_TWO_MB)):
            logger.debug("ml_memory_Grenze: mallopt(M_MMAP_Schwelle) nicht blockierend: glibc lehnte Wert ab")
    except Exception as _mallopt_exc:
        logger.debug("ml_memory_Grenze: mallopt-Tuning nicht blockierend: %s", _mallopt_exc)


_tune_glibc_malloc_mmap_threshold()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _auto_detect_budget() -> float:
    """Derive ML budget from system RAM: total_ram / 3, capped at [4, 12] GB.

    Ensures ~2/3 of RAM stays free for OS, Qt GUI, audio buffers, and numpy
    intermediate arrays.  On 32 GB → 10 GB; on 16 GB → 5 GB; on 64 GB → 12 GB.
    """
    if _psutil is not None:
        total_gb = _psutil.virtual_memory().total / (1024**3)
        budget = max(4.0, min(12.0, total_gb / 3.0))
        return round(budget, 1)  # type: ignore[no-any-return]
    return 10.0  # conservative default without psutil


# Maximum total RAM allowed for ALL ML models combined.
# Auto-detected from system RAM; override via set_budget() if needed.
ML_MAX_GB: float = _auto_detect_budget()
_SYSTEM_MEMORY_MARGIN_BASE: float = 1.35  # Basis-Margin für kleine Modelle (< 1 GB)
_SYSTEM_MEMORY_MARGIN_MIN: float = 1.10  # Minimale Margin für sehr große Modelle (>= 5 GB)
_PRESSURE_LIGHT_MODEL_MAX_GB: float = 0.12
_PRESSURE_LIGHT_MODEL_MIN_AVAIL_RATIO: float = 0.35
_PRESSURE_LIGHT_MODEL_MAX_SWAP_PCT: float = 95.0
_PRESSURE_LIGHT_MODEL_TOTAL_GB_CAP: float = 0.35
_PRESSURE_LIGHT_MODEL_ALLOWLIST: frozenset[str] = frozenset(
    {
        "SileroVAD",
        "SileroVAD_phase18",
        "FCPE",
        "BasicPitch",
    }
)


def _calibrate_guard_thresholds() -> dict[str, float]:
    """Calibrate all preemptive-guard thresholds to system RAM at import time.

    Larger systems get proportionally more permissive thresholds because
    absolute free RAM is still large even at lower percentages, and swap
    pressure is less critical when more physical RAM is available.

    Returns a dict of calibrated values (also stored as module-level constants).
    """
    if _psutil is None:
        _total_gb = 16.0
    else:
        _total_gb = float(_psutil.virtual_memory().total) / (1024.0**3)

    # ── Heavy-model preemptive guard thresholds ──────────────────────

    # Minimum model size that triggers the heavy-load guard.
    _heavy_min_gb: float = 1.0

    # Swap threshold for \"elevated\" (definite block with paging/low-headroom).
    # Larger RAM → more tolerant of swap (swap pressure less critical).
    _heavy_swap_pct: float = 70.0 + (_total_gb - 8.0) * 0.22
    _heavy_swap_pct = max(65.0, min(80.0, _heavy_swap_pct))

    # Swap threshold for \"early\" (block only with low headroom).
    _heavy_swap_early_pct: float = 45.0 + (_total_gb - 8.0) * 0.28
    _heavy_swap_early_pct = max(40.0, min(58.0, _heavy_swap_early_pct))

    # Active swap I/O threshold (MB/s).
    _heavy_swap_io_mb_s: float = 2.0

    # Available-RAM ratio below which headroom is considered \"low\".
    # Larger systems can tolerate lower ratios (more RAM = more buffer).
    _heavy_avail_ratio_max: float = 0.30 - (_total_gb - 8.0) * 0.0045
    _heavy_avail_ratio_max = max(0.10, min(0.32, _heavy_avail_ratio_max))

    # ── Hard minimum free RAM (systemd-oomd protection) ─────────────
    # Scales to 7-10% of total RAM, floor at 2 GB.
    _min_free_mb_hard: float = max(2048.0, _total_gb * 0.075 * 1024.0)

    # ── System memory margins ────────────────────────────────────────
    # Smaller systems need larger relative margins.
    if _total_gb >= 32.0:
        _margin_base = 1.25
        _margin_min = 1.06
    elif _total_gb >= 16.0:
        _margin_base = 1.30
        _margin_min = 1.08
    else:
        _margin_base = 1.35
        _margin_min = 1.10

    # ── Pressure recovery ────────────────────────────────────────────
    _pressure_recovery_attempts: int = 2
    _pressure_recovery_sleep_s: float = 0.35

    # ── Persist as module-level for fast access ──────────────────────
    globals().update(
        {
            "_HEAVY_MODEL_PREEMPTIVE_MIN_GB": _heavy_min_gb,
            "_HEAVY_MODEL_PREEMPTIVE_SWAP_PCT": round(_heavy_swap_pct, 1),
            "_HEAVY_MODEL_PREEMPTIVE_SWAP_EARLY_PCT": round(_heavy_swap_early_pct, 1),
            "_HEAVY_MODEL_PREEMPTIVE_SWAP_IO_MB_S": _heavy_swap_io_mb_s,
            "_HEAVY_MODEL_PREEMPTIVE_AVAIL_RATIO_MAX": round(_heavy_avail_ratio_max, 2),
            "_MIN_FREE_MB_HARD": round(_min_free_mb_hard, 0),
            "_PRESSURE_RECOVERY_ATTEMPTS": _pressure_recovery_attempts,
            "_PRESSURE_RECOVERY_SLEEP_S": _pressure_recovery_sleep_s,
            "_SYSTEM_MEMORY_MARGIN_BASE": _margin_base,
            "_SYSTEM_MEMORY_MARGIN_MIN": _margin_min,
            # For _preflight_system_memory load-peak factors
            "_CALIBRATED_TOTAL_RAM_GB": round(_total_gb, 1),
        }
    )

    logger.info(
        "ml_memory_Grenze: guard thresholds kalibriert for %.1f GB RAM — "
        "heavy_avail_Verhaeltnis=%.2f, heavy_swap_early=%.0f%%, heavy_swap=%.0f%%, "
        "min_free_mb=%.0f, margin_base=%.2f, margin_min=%.2f",
        _total_gb,
        _heavy_avail_ratio_max,
        _heavy_swap_early_pct,
        _heavy_swap_pct,
        _min_free_mb_hard,
        _margin_base,
        _margin_min,
    )

    return {
        "total_ram_gb": round(_total_gb, 1),
        "heavy_avail_ratio_max": round(_heavy_avail_ratio_max, 2),
        "heavy_swap_early_pct": round(_heavy_swap_early_pct, 1),
        "heavy_swap_pct": round(_heavy_swap_pct, 1),
        "min_free_mb_hard": round(_min_free_mb_hard, 0),
    }


# Run calibration once at module import.
_calibrated = _calibrate_guard_thresholds()


# ── §B2 Systemprofil beim Start loggen ─────────────────────────────
def _log_system_profile() -> None:
    """Log a one-time system profile at module import for diagnostics."""
    if _psutil is None:
        return
    try:
        vm = _psutil.virtual_memory()
        swap = _psutil.swap_memory()
        cpu_count = _psutil.cpu_count(logical=True)
        cpu_phys = _psutil.cpu_count(logical=False)
        logger.info(
            "ml_memory_Grenze: Systemprofil — RAM total=%.1f GB, verfuegbar=%.1f GB (%.0f%%), "
            "swap total=%.1f GB, used=%.0f%%, CPU=%d logical/%d physical",
            vm.total / (1024**3),
            vm.available / (1024**3),
            vm.available / max(vm.total, 1) * 100,
            swap.total / (1024**3),
            swap.percent,
            cpu_count or 0,
            cpu_phys or 0,
        )
    except Exception as _e:
        logger.debug("ml_memory_Grenze: unkritisch exception: %s", _e)


_log_system_profile()

# Provide defaults in case _calibrate_guard_thresholds fails (psutil missing, etc.).
# These are overridden by globals().update() above when calibration succeeds.
_HEAVY_MODEL_PREEMPTIVE_MIN_GB: float = 1.0
_HEAVY_MODEL_PREEMPTIVE_SWAP_PCT: float = 70.0
_HEAVY_MODEL_PREEMPTIVE_SWAP_EARLY_PCT: float = 45.0
_HEAVY_MODEL_PREEMPTIVE_SWAP_IO_MB_S: float = 2.0
_HEAVY_MODEL_PREEMPTIVE_AVAIL_RATIO_MAX: float = 0.20
_MIN_FREE_MB_HARD: float = 2048.0
_PRESSURE_RECOVERY_ATTEMPTS: int = 2
_PRESSURE_RECOVERY_SLEEP_S: float = 0.35
_CALIBRATED_TOTAL_RAM_GB: float = 8.0

# Cooldown for is_system_thrashing() log-spam guard (BUG G).
# Log WARNING at most once per 60 s; always return the correct bool.
_THRASH_WARN_COOLDOWN_S: float = 60.0
_last_thrash_warn_time: float = 0.0
_last_thrash_warn_lock = threading.Lock()


def _scaled_margin(size_gb: float) -> float:
    """Skalierte RAM-Margin: Große Modelle brauchen prozentual weniger Reserve.

    Kleine Modelle (<1 GB): 1.35× (35% Reserve für Overhead, Fragmentierung)
    Große Modelle (>=5 GB): 1.10× (10% Reserve — Modellgewichte sind kompakt, wenig Overhead)
    Dazwischen: linear interpoliert.

    Begründung: AudioSR (7 GB) × 1.35 = 9.45 GB — blockiert auf 16-GB-Systemen
    obwohl 8.8 GB frei sind. Mit skalierter Margin: 7 × 1.10 = 7.7 GB — passt.
    """
    if size_gb <= 1.0:
        return _SYSTEM_MEMORY_MARGIN_BASE
    if size_gb >= 5.0:
        return _SYSTEM_MEMORY_MARGIN_MIN
    # Linear interpolation between 1.0 GB and 5.0 GB
    t = (size_gb - 1.0) / 4.0
    return _SYSTEM_MEMORY_MARGIN_BASE + t * (_SYSTEM_MEMORY_MARGIN_MIN - _SYSTEM_MEMORY_MARGIN_BASE)


# ---------------------------------------------------------------------------
# Internal state (thread-safe)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_allocated: dict[str, float] = {}  # model_name → GB currently allocated
_total_gb: float = 0.0  # sum of _allocated.values()
_last_swap_poll_ts: float = 0.0
_last_swap_sin: int = 0
_last_swap_sout: int = 0


def _swap_io_rate_mb_per_s(swap_obj: object) -> float:
    """Schätzt recent swap I/O activity in MB/s.

    High swap usage alone does not always mean active thrashing. This helper uses
    swap sin/sout counters to distinguish stale high swap occupancy from ongoing
    paging pressure that can trigger freezes and OOM-kills.
    """
    global _last_swap_poll_ts, _last_swap_sin, _last_swap_sout  # pylint: disable=global-statement
    if _psutil is None:
        return 0.0
    try:
        now = time.monotonic()
        sin = int(getattr(swap_obj, "sin", 0) or 0)
        sout = int(getattr(swap_obj, "sout", 0) or 0)

        if _last_swap_poll_ts <= 0.0:
            _last_swap_poll_ts = float(now)
            _last_swap_sin = sin
            _last_swap_sout = sout
            return 0.0

        dt = max(1e-3, float(now - _last_swap_poll_ts))
        delta_bytes = max(0, (sin - _last_swap_sin)) + max(0, (sout - _last_swap_sout))
        rate_mb_s = (delta_bytes / (1024.0 * 1024.0)) / dt

        _last_swap_poll_ts = float(now)
        _last_swap_sin = sin
        _last_swap_sout = sout
        return float(rate_mb_s)
    except Exception as e:
        logger.warning("ml_memory_Grenze.py::_swap_io_rate_mb_per_s Ersatzpfad: %s", e)
        return 0.0


def _available_memory_mb() -> float:
    """Gibt available system memory in MB, or inf if psutil is unavailable zurück."""
    if _psutil is None:
        return float("inf")
    return float(_psutil.virtual_memory().available / (1024 * 1024))


def is_system_thrashing() -> bool:
    """Erkennt swap-thrashing: high swap usage or combined swap+RAM pressure.

        Conditions (any triggers):
            1. swap > 80 % AND active swap I/O > 8 MB/s (real thrashing)
            2. swap > 95 % AND available RAM < 25 % (critical emergency)
            3. swap > 30 % AND available RAM < 15 % (systemd-oomd warning zone)
            4. available RAM < 8 % (hard emergency regardless of swap)

    On Linux, systemd-oomd kills at ~50 % memory-pressure for > 20 s.
    We detect BEFORE that point to allow graceful degradation.
    """
    if _psutil is None:
        return False
    try:
        swap = _psutil.swap_memory()
        vm = _psutil.virtual_memory()
        swap_used_pct = swap.percent  # 0–100
        avail_ratio = vm.available / max(vm.total, 1)
        swap_io_rate_mb_s = _swap_io_rate_mb_per_s(swap)
        # Condition 1: critically high swap plus ongoing swap I/O.
        swap_critical_active = swap_used_pct > 80.0 and swap_io_rate_mb_s > 8.0
        # Condition 2: emergency when swap is nearly full and RAM headroom is shrinking.
        swap_critical_emergency = swap_used_pct > 95.0 and avail_ratio < 0.25
        # Condition 3: combined pressure (legacy heuristic, retained).
        combined_pressure = swap_used_pct > 30.0 and avail_ratio < 0.15
        # Condition 4: hard emergency regardless of swap stats.
        ram_emergency = avail_ratio < 0.08

        thrashing = swap_critical_active or swap_critical_emergency or combined_pressure or ram_emergency
        if thrashing:
            global _last_thrash_warn_time  # pylint: disable=global-statement
            _now = time.monotonic()
            with _last_thrash_warn_lock:
                _emit = (_now - _last_thrash_warn_time) >= _THRASH_WARN_COOLDOWN_S
                if _emit:
                    _last_thrash_warn_time = _now
            if _emit:
                logger.warning(
                    "ml_memory_Grenze: swap thrashing erkannt — swap %.0f %% used (%.1f GB), "
                    "swap I/O %.1f MB/s, RAM verfuegbar %.1f %% (%.1f GB) — ML loads will be blocked",
                    swap_used_pct,
                    swap.used / (1024**3),
                    swap_io_rate_mb_s,
                    avail_ratio * 100,
                    vm.available / (1024**3),
                )
        elif swap_used_pct > 80.0 and swap_io_rate_mb_s <= 8.0:
            logger.debug(
                "ml_memory_Grenze: high swap occupancy without active paging (swap %.0f %%, I/O %.1f MB/s) "
                "— no thrashing block",
                swap_used_pct,
                swap_io_rate_mb_s,
            )
        return thrashing
    except Exception as e:
        logger.warning("ml_memory_Grenze.py::is_system_thrashing Ersatzpfad: %s", e)
        return False


def _allow_lightweight_under_pressure(model_name: str, size_gb: float) -> bool:
    """Allow tiny models during pressure when memory headroom is still healthy.

    This prevents broad cascade fallbacks (e.g. VAD/pitch helper models) while
    keeping large model loads blocked under thrashing.
    """
    if _psutil is None:
        return False
    if size_gb > _PRESSURE_LIGHT_MODEL_MAX_GB:
        return False
    if model_name not in _PRESSURE_LIGHT_MODEL_ALLOWLIST:
        return False
    try:
        vm = _psutil.virtual_memory()
        swap = _psutil.swap_memory()
        avail_ratio = vm.available / max(vm.total, 1)
        swap_pct = float(getattr(swap, "percent", 100.0))
        with _lock:
            _tiny_allocated = float(
                sum(
                    float(_sz)
                    for _name, _sz in _allocated.items()
                    if _name in _PRESSURE_LIGHT_MODEL_ALLOWLIST and float(_sz) <= _PRESSURE_LIGHT_MODEL_MAX_GB
                )
            )
        if (_tiny_allocated + float(size_gb)) > _PRESSURE_LIGHT_MODEL_TOTAL_GB_CAP:
            logger.warning(
                "ML-Grenze: '%s' (%.2f GB) pressure soft-allow abgelehnt — tiny-Grenze cap erreicht (%.2f/%.2f GB)",
                model_name,
                size_gb,
                _tiny_allocated,
                _PRESSURE_LIGHT_MODEL_TOTAL_GB_CAP,
            )
            return False
        if avail_ratio >= _PRESSURE_LIGHT_MODEL_MIN_AVAIL_RATIO and swap_pct < _PRESSURE_LIGHT_MODEL_MAX_SWAP_PCT:
            logger.warning(
                "ML-Grenze: '%s' (%.2f GB) soft-allowed under pressure window "
                "(RAM %.1f %%, swap %.1f %%, tiny-cap %.2f/%.2f GB)",
                model_name,
                size_gb,
                avail_ratio * 100.0,
                swap_pct,
                _tiny_allocated + float(size_gb),
                _PRESSURE_LIGHT_MODEL_TOTAL_GB_CAP,
            )
            return True
    except Exception as e:
        logger.warning("ml_memory_Grenze.py::_allow_lightweight_under_pressure Ersatzpfad: %s", e)
        return False
    return False


def _estimate_load_peak_factor(model_size_gb: float) -> float:
    """Estimate the deserialization load-peak factor from model file size (§E2).

    Larger model files have proportionally less deserialization overhead
    because the tensor data dominates the file size, while metadata and
    graph structure overhead is roughly constant.

    Empirical basis (PyTorch torch.load / ONNX deserialization):
        < 500 MB  → 1.50×  (metadata overhead proportionally large)
        0.5–2 GB  → 1.35×  (moderate overhead)
        > 2 GB    → 1.20×  (tensor-dominated, compact)

    This complements the system-RAM-based factor in _preflight_system_memory
    and _should_block_heavy_ml_load by providing a model-specific estimate.
    The lower of the two factors is used (more optimistic = less blocking).
    """
    if model_size_gb >= 2.0:
        return 1.20
    if model_size_gb >= 0.5:
        return 1.35
    return 1.50


def _should_block_heavy_ml_load(size_gb: float) -> bool:
    """Gibt True when heavy model loads should be blocked preemptively zurück.

    Rationale:
    - Crash pattern: swap climbs from ~70 % to >85 % during a single heavy load wave
      even while RAM still looks healthy in absolute GB.
    - Thrashing detection alone can be too late if swap occupancy is already high and
      paging starts shortly after model deserialization begins.

    Policy:
    - Applies only to heavy models (>= 1.0 GB).
    - Blocks when swap is already elevated AND either active paging is visible
      or available RAM ratio is no longer high.
    """
    if _psutil is None:
        return False
    if size_gb < _HEAVY_MODEL_PREEMPTIVE_MIN_GB:
        return False
    try:
        vm = _psutil.virtual_memory()
        swap = _psutil.swap_memory()
        swap_pct = float(getattr(swap, "percent", 0.0) or 0.0)
        avail_bytes = float(vm.available)
        avail_ratio = avail_bytes / max(float(vm.total), 1.0)
        avail_gb = avail_bytes / float(1024**3)
        swap_io_rate_mb_s = _swap_io_rate_mb_per_s(swap)

        # ── Model-size-aware safe-free-RAM threshold ───────────────
        # Larger models need more free RAM (proportional to load-peak).
        # Uses the same adaptive peak factors as _preflight_system_memory
        # but at 80 % strictness for the preemptive gate.
        # §E2: Blend system-RAM-based peak with model-file-size-based peak.
        _total_ram_gb = _psutil.virtual_memory().total / (1024.0**3)
        if _total_ram_gb >= 24.0:
            _sys_peak = 1.30
            _oomd_frac = 0.08
        elif _total_ram_gb >= 16.0:
            _sys_peak = 1.45
            _oomd_frac = 0.10
        else:
            _sys_peak = 1.60
            _oomd_frac = 0.12
        _model_peak = _estimate_load_peak_factor(size_gb)
        _peak = min(_sys_peak, _model_peak)  # use the more optimistic estimate
        _preempt_factor = 0.80
        _model_safe_gb = (size_gb * _peak + _total_ram_gb * _oomd_frac) * _preempt_factor

        elevated_swap = swap_pct >= _HEAVY_MODEL_PREEMPTIVE_SWAP_PCT
        early_swap = swap_pct >= _HEAVY_MODEL_PREEMPTIVE_SWAP_EARLY_PCT
        active_paging = swap_io_rate_mb_s >= _HEAVY_MODEL_PREEMPTIVE_SWAP_IO_MB_S
        # Hard safety floor: below _MIN_FREE_MB_HARD, always block regardless of swap.
        # Protects against low-RAM + moderate-swap scenarios where neither swap
        # threshold triggers but physical RAM is critically low.
        _hard_floor_gb = _MIN_FREE_MB_HARD / 1024.0
        low_headroom = (
            avail_ratio <= _HEAVY_MODEL_PREEMPTIVE_AVAIL_RATIO_MAX
            or avail_gb <= _model_safe_gb
            or avail_gb <= _hard_floor_gb
        )
        should_block = (elevated_swap and (active_paging or low_headroom)) or (early_swap and low_headroom)
        if should_block:
            logger.warning(
                "ML-Grenze: preemptive heavy-laden block (model %.1f GB, safe-Schwelle %.1f GB) — "
                "swap %.0f %%, swap-I/O %.1f MB/s, RAM verfuegbar %.1f %% (%.1f GB) "
                "→ DSP Ersatzpfad before thrashing escalation",
                size_gb,
                _model_safe_gb,
                swap_pct,
                swap_io_rate_mb_s,
                avail_ratio * 100.0,
                avail_gb,
            )
        return should_block
    except Exception as e:
        logger.warning("ml_memory_Grenze.py::_should_block_heavy_ml_laden Ersatzpfad: %s", e)
        return False


def _preflight_system_memory(required_mb: float) -> bool:
    """Best-effort system RAM preflight before allocating ML budget.

    This complements the logical ML budget with a physical RAM check.
    On pressure, it asks PluginLifecycleManager to evict stale models first.

    For models >= 1 GB we apply a load-peak-aware check:
      PyTorch's torch.load() temporarily uses ~1.6× the model weight size while
      deserializing tensors (the original bytes and the parsed tensors coexist
      briefly in memory).  We must also keep at least 12 % of total system RAM
      free after the peak to stay below the systemd-oomd kill threshold (90 %).
      Without this check a 2 GB model on a system with 5.9 GB free PASSES the
      steady-state margin (1.29 GB needed) but CRASHES during load:
        peak usage ≈ 3.2 GB → RAM reaches 92 % → systemd-oomd fires SIGKILL.
    """
    if _psutil is None:
        return True

    _size_gb = required_mb / 1024.0
    available_mb = _available_memory_mb()

    if _size_gb >= 1.0:
        # Load-peak formula: adaptive to total system RAM (§E2).
        # ≥24 GB → 1.30× peak, 8 % oomd (more headroom → tighter margins safe)
        # ≥16 GB → 1.45× peak, 10 % oomd
        #  <16 GB → 1.60× peak, 12 % oomd (conservative)
        # §E2: Blend with model-file-size-based peak for more precision.
        _total_ram_gb = float(_psutil.virtual_memory().total) / (1024.0**3)
        if _total_ram_gb >= 24.0:
            _sys_peak = 1.30
            _oomd_pct = 0.08
        elif _total_ram_gb >= 16.0:
            _sys_peak = 1.45
            _oomd_pct = 0.10
        else:
            _sys_peak = 1.60
            _oomd_pct = 0.12
        _model_peak = _estimate_load_peak_factor(_size_gb)
        _load_peak = min(_sys_peak, _model_peak)
        _total_ram_mb = float(_psutil.virtual_memory().total) / (1024.0 * 1024.0)
        _oomd_safe_mb = max(2048.0, _total_ram_mb * _oomd_pct)
        _peak_required_mb = required_mb * _load_peak + _oomd_safe_mb
        required_with_margin = max(_peak_required_mb, required_mb * _scaled_margin(_size_gb), _MIN_FREE_MB_HARD)
    else:
        _margin = _scaled_margin(_size_gb)
        required_with_margin = max(required_mb * _margin, _MIN_FREE_MB_HARD)

    if available_mb >= required_with_margin:
        return True

    try:
        from backend.core.plugin_lifecycle_manager import evict_stale_plugins  # pylint: disable=import-outside-toplevel

        evict_stale_plugins(required_mb=required_with_margin)
    except Exception as _exc:
        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    available_after_evict_mb = _available_memory_mb()
    if available_after_evict_mb >= required_with_margin:
        return True

    logger.warning(
        "ml_memory_Grenze: physical RAM too low — required %.0f MB (incl. safety margin), "
        "verfuegbar %.0f MB. ML laden blocked, DSP Ersatzpfad active.",
        required_with_margin,
        available_after_evict_mb,
    )
    # §DEBUG: Stack-Trace für suspekt große Requests (> 10000 MB) automatisch loggen
    if required_with_margin > 10_000:
        import traceback as _tb  # pylint: disable=import-outside-toplevel

        logger.warning(
            "ml_memory_Grenze §DEBUG: suspekt große Anfrage (%.0f MB) — Stack-Trace:\n%s",
            required_with_margin,
            "".join(_tb.format_stack()),
        )
    return False


def _attempt_quality_preserving_pressure_recovery(model_name: str, size_gb: float) -> bool:
    """Try short pressure recovery before forcing DSP fallback.

    Strategy: evict stale plugins and wait briefly so the allocator can reclaim
    memory pages. This keeps ML-first behavior under transient pressure and only
    blocks when pressure remains critical after retries.
    """
    if _psutil is None:
        return False

    required_mb = max(float(size_gb), 0.0) * 1024.0
    attempts = max(0, int(_PRESSURE_RECOVERY_ATTEMPTS))
    if attempts <= 0:
        return False

    for attempt in range(1, attempts + 1):
        try:
            # pylint: disable=import-outside-toplevel
            from backend.core.plugin_lifecycle_manager import evict_stale_plugins
            # pylint: enable=import-outside-toplevel

            evicted = int(evict_stale_plugins(required_mb=required_mb))
        except Exception as _exc:
            logger.debug("ML-Grenze: pressure Wiederherstellung eviction fehlgeschlagen (non-fatal): %s", _exc)
            evicted = 0

        if _PRESSURE_RECOVERY_SLEEP_S > 0.0:
            time.sleep(_PRESSURE_RECOVERY_SLEEP_S)

        still_thrashing = is_system_thrashing()
        still_heavy_block = _should_block_heavy_ml_load(float(max(size_gb, 0.0)))
        if not still_thrashing and not still_heavy_block:
            # §v10.306: Nur akzeptieren wenn tatsächlich RAM freigeworden ist
            if evicted > 0:
                logger.warning(
                    "ML-Grenze: pressure Wiederherstellung succeeded for '%s' after %d/%d Versuch(s) "
                    "(evicted=%d) — ML laden Wiederholung statt DSP-Ersatzpfad.",
                    model_name,
                    attempt,
                    attempts,
                    evicted,
                )
                return True
            # Kein Plugin evicted → Druck nur transient gesunken → DSP-Fallback sicherer
            logger.warning(
                "ML-Grenze: pressure Wiederherstellung transient for '%s' (evicted=0) — DSP-Ersatzpfad statt Risiko.",
                model_name,
            )

        logger.warning(
            "ML-Grenze: pressure Wiederherstellung %d/%d for '%s' insufficient (evicted=%d, thrashing=%s, heavy_block=%s)",
            attempt,
            attempts,
            model_name,
            evicted,
            still_thrashing,
            still_heavy_block,
        )

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def try_allocate(model_name: str, size_gb: float) -> bool:
    """Reserve ``size_gb`` GB of ML budget for ``model_name``.
        _total_gb = 0  # Aurik v10: init before += (bugfix)

    Returns True if granted (proceed to load the model).
    Returns False if the budget would be exceeded (use DSP fallback instead).
    Idempotent: if ``model_name`` is already allocated, returns True immediately.
    """
    global _total_gb  # pylint: disable=global-statement
    with _lock:
        if model_name in _allocated:
            # Already loaded — no additional budget consumed.
            return True

    # Swap-Thrashing-Guard: Wenn system bereits thrashing, alle neuen
    # ML-Loads blockieren — DSP-Fallback statt Freeze/OOM.
    if is_system_thrashing():
        if _allow_lightweight_under_pressure(
            model_name, float(max(size_gb, 0.0))
        ) or _attempt_quality_preserving_pressure_recovery(model_name, size_gb):
            pass
        else:
            logger.warning(
                "ML-Grenze: '%s' (%.1f GB) blockiert — System-Thrashing erkannt, DSP-Ersatzpfad aktiv.",
                model_name,
                size_gb,
            )
            return False

    # Präventiver Heavy-Load-Guard: große Modell-Ladungen schon vor dem
    # harten Thrashing-Zustand abbrechen, wenn Swap-Druck bereits ansteigt.
    if _should_block_heavy_ml_load(float(max(size_gb, 0.0))):
        if not _attempt_quality_preserving_pressure_recovery(model_name, size_gb):
            logger.warning(
                "ML-Grenze: '%s' (%.1f GB) präventiv blockiert — erhöhter Swap-Druck, DSP-Ersatzpfad aktiv.",
                model_name,
                size_gb,
            )
            return False

    if not _preflight_system_memory(required_mb=max(size_gb, 0.0) * 1024.0):
        return False

    with _lock:
        if model_name in _allocated:
            return True
        remaining = ML_MAX_GB - _total_gb
        if size_gb > remaining:
            logger.warning(
                "ml_memory_Grenze: '%s' needs %.1f GB, only %.1f GB of %.1f GB free — DSP Ersatzpfad active.",
                model_name,
                size_gb,
                remaining,
                ML_MAX_GB,
            )
            return False
        _allocated[model_name] = size_gb
        _total_gb += size_gb
        logger.info(
            "ml_memory_Grenze: '%s' allocated %.1f GB  →  total %.1f / %.1f GB used.",
            model_name,
            size_gb,
            _total_gb,
            ML_MAX_GB,
        )
    # §C2: Invalidate ml_model_readiness failure cache so subsequent
    # readiness checks reflect the newly loaded state.
    # §v10.304.33: Import AUSSERHALB von _lock — ml_model_readiness
    # braucht ~5-10s (webrtcvad→pkg_resources). Lock während Import
    # gehalten → alle anderen try_allocate-Aufrufe blockieren.
    try:
        from backend.core.ml_model_readiness import invalidate_ml_readiness

        invalidate_ml_readiness(model_name)
    except Exception as _e:
        logger.debug("ml_memory_Grenze: unkritisch exception: %s", _e)
    return True


def release(model_name: str) -> None:
    """Release the budget slot for ``model_name`` (call when model is unloaded).

    Safe to call even if the model was never allocated.
    """
    global _total_gb  # pylint: disable=global-statement
    with _lock:
        freed = _allocated.pop(model_name, 0.0)
        _total_gb = max(0.0, _total_gb - freed)
        if freed:
            logger.info(
                "ml_memory_Grenze: '%s' released (%.1f GB)  →  total %.1f / %.1f GB used.",
                model_name,
                freed,
                _total_gb,
                ML_MAX_GB,
            )


def _aggressive_memory_reclaim() -> None:
    """§OOM-SwapGuard: Erzwingt sofortige Speicher-Freigabe an das Betriebssystem.

    Aufgerufen wenn Swap-Nutzung kritisch ist (§v10.307 Pre-Pipeline-Drain,
    §OOM-SwapGuard vor jeder Phase). Drei Stufen, jede best-effort:
        1. Python GC über alle Generationen (löst zyklische Referenzen z. B.
           in ML-Tensor-Graphen, die sonst erst spät kollektiert würden).
        2. Torch-GPU-Cache leeren (CUDA/ROCm) — PyTorch hält freigegebene
           GPU-Buffer standardmäßig in einem internen Cache-Allocator zurück.
        3. glibc `malloc_trim(0)` (Linux) — gibt vom Python-Heap freigegebene,
           aber vom Allocator zurückgehaltene Pages tatsächlich an das OS zurück;
           ohne diesen Schritt bleibt RSS trotz gc.collect() oft hoch.
    """
    import gc

    for _gen in range(3):
        gc.collect(_gen)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as _torch_exc:
        logger.debug("ml_memory_Grenze: Torch-Zwischenspeicher-Reclaim nicht blockierend: %s", _torch_exc)

    try:
        import ctypes
        import platform

        if platform.system() == "Linux":
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as _trim_exc:
        logger.debug("ml_memory_Grenze: malloc_trim nicht blockierend: %s", _trim_exc)


def get_status() -> dict:
    """Gibt current budget status (for logging/debug) zurück."""
    with _lock:
        return {
            "max_gb": ML_MAX_GB,
            "allocated_gb": round(_total_gb, 2),
            "free_gb": round(ML_MAX_GB - _total_gb, 2),
            "models": dict(_allocated),
        }


def set_budget(max_gb: float, guard_overrides: dict[str, float] | None = None) -> None:
    """Override the default budget and optionally guard thresholds.

    Args:
        max_gb: New ML memory budget in GB.
        guard_overrides: Optional dict of guard threshold overrides.
            Supported keys: 'heavy_swap_early_pct', 'heavy_swap_pct',
            'heavy_avail_ratio_max', 'min_free_mb_hard'.
            Values are applied after calibration; use for per-system tuning.

    Example:
        set_budget(12.0, {'heavy_swap_early_pct': 55.0, 'min_free_mb_hard': 4096})
    """
    global ML_MAX_GB  # pylint: disable=global-statement
    with _lock:
        ML_MAX_GB = float(max_gb)
        logger.info("ml_memory_Grenze: max Grenze set to %.1f GB.", ML_MAX_GB)

    if guard_overrides:
        _valid_keys = {
            "_HEAVY_MODEL_PREEMPTIVE_SWAP_EARLY_PCT",
            "_HEAVY_MODEL_PREEMPTIVE_SWAP_PCT",
            "_HEAVY_MODEL_PREEMPTIVE_AVAIL_RATIO_MAX",
            "_MIN_FREE_MB_HARD",
        }
        applied: list[str] = []
        for key, val in guard_overrides.items():
            _mod_key = f"_HEAVY_MODEL_PREEMPTIVE_{key.upper()}" if not key.startswith("_") else key
            if _mod_key in _valid_keys or key in _valid_keys:
                target = _mod_key if _mod_key in _valid_keys else key
                globals()[target] = float(val)
                applied.append(f"{target}={val}")
        if applied:
            logger.info(
                "ml_memory_Grenze: guard thresholds overridden — %s",
                ", ".join(applied),
            )


# ---------------------------------------------------------------------------
# §3.9.5  Startup reconciliation
# ---------------------------------------------------------------------------


def _reconcile_on_startup() -> None:
    """Setzt zurück: allocated budget to 0 on fresh process start (§3.9.5).

    Rationale: All allocations from a previous process are gone after OS
    cleanup (SIGKILL / crash). Each module re-registers via try_allocate()
    when it actually loads its model. No stale allocation persists across
    process boundaries.  Called once at module import time.
    """
    global _total_gb  # pylint: disable=global-statement
    with _lock:
        _allocated.clear()
        _total_gb = 0.0
    logger.info("ml_memory_Grenze: Start reconciliation — Grenze zurueckgesetzt to 0.0 GB")


# Run reconciliation exactly once at module import (= new process start).
_reconcile_on_startup()


# ---------------------------------------------------------------------------
# §3.9.5  Thin OO wrapper — returned by get_ml_memory_budget()
# ---------------------------------------------------------------------------


class _MLMemoryBudgetProxy:
    """Thin proxy so callers can use get_ml_memory_budget().try_allocate().

    All work is delegated to the module-level functions; no additional state.
    """

    # Lock-order: Priority 1 (MLMemoryBudget) — see §3.9.8
    def try_allocate(self, model_name: str, size_gb: float) -> bool:
        """Delegiert an das modulare try_allocate."""
        return try_allocate(model_name, size_gb)

    def release(self, model_name: str) -> None:
        """Delegiert an das modulare release."""
        release(model_name)

    def get_status(self) -> dict:
        """Delegiert an das modulare get_status."""
        return get_status()


_proxy_instance = _MLMemoryBudgetProxy()


def get_ml_memory_budget() -> _MLMemoryBudgetProxy:
    """Gibt the global ML-memory-budget proxy (singleton-safe) zurück.

    Provides an OO API (.try_allocate / .release) in addition to the
    module-level functions, so both usage styles work.
    """
    return _proxy_instance
