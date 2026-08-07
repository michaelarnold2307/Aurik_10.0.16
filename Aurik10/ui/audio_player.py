"""Unterbrechungsfreier Audio-Player für Aurik (v10.0.0).

Verwendet einen persistenten ``sounddevice.OutputStream`` mit Callback-basierter
Audioausgabe.  Quellwechsel (Original ↔ Restauriert ↔ Live-Preview) erfolgen
**lückenlos** durch ein additives Crossfade-Overlay im Audio-Thread — kein
Stop/Restart des Streams nötig.

Design-Prinzipien:
  * Der Lock im Callback schützt nur 3 Referenz-Reads + 1 Int-Write (~µs).
  * Teure Operationen (Resampling) finden **vor** Lock-Acquire statt.
  * Das Crossfade-Overlay ist maximal 480 Samples (10 ms @ 48 kHz, ~4 KB)
    — Berechnung dauert <50 µs, kein Underrun-Risiko.
  * Sample-genaues Position-Tracking (kein ``time.monotonic()``-Drift).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd

    _SD_AVAILABLE = True
except Exception:
    sd = None  # type: ignore[assignment]
    _SD_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CROSSFADE_SAMPLES = 384  # ~8 ms @ 48 kHz — smooth enough, imperceptible latency


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class StreamingAudioPlayer:
    """Gapless audio player with source-hot-swap capability.

    The player owns a single ``OutputStream``.  Audio sources can be swapped
    at any time without audible interruption (crossfade overlay).

    Thread-safety
    -------------
    * ``play()``, ``stop()``, ``seek()``, ``shutdown()`` — safe from any thread.
    * Properties (``is_playing``, ``position_frac``, …) — safe from any thread.
    * The ``on_finished`` callback runs in the **PortAudio callback thread**;
      use ``QMetaObject.invokeMethod`` or similar to marshal to the GUI thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Stream
        self._stream: sd.OutputStream | None = None  # type: ignore[name-defined]
        self._output_sr: int = 0
        self._channels: int = 2

        # Playback state — the callback reads these WITHOUT the lock (see
        # _audio_callback docstring).  Writers MUST hold self._lock and
        # assign self._buf LAST (after _pos, _cf_overlay, _active etc.)
        # so the callback never sees a stale buf/pos combination.
        self._buf: np.ndarray | None = None  # float32, shape [N, channels]
        self._pos: int = 0
        self._active: bool = False
        self._stop_after_fade: bool = False

        # Crossfade overlay (additive, small — written under lock)
        self._cf_overlay: np.ndarray | None = None
        self._cf_pos: int = 0

        # Metadata for UI queries
        self._source_duration_s: float = 0.0
        self._label: str = ""

        # Finished callback (called in PortAudio thread when playback ends naturally)
        self._on_finished: Callable[[], None] | None = None
        self._fire_finished: bool = False

        # Resample cache: list of (fingerprint, prepared_audio)
        # fingerprint = (shape, hash_first_4KB, sr, output_sr)
        # Keeps last 3 entries — enough for original + restored + preview.
        self._resample_cache: list[tuple[tuple, np.ndarray]] = []
        self._MAX_CACHE = 3

    # ------------------------------------------------------------------
    # Properties (all thread-safe, read without lock for atomicity)
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if sounddevice is importable and an output device exists."""
        if not _SD_AVAILABLE or sd is None:
            return False
        try:
            dev = sd.query_devices(kind="output")
        except Exception as exc:
            logger.debug("StreamingAudioPlayer: Ausgabe device query fehlgeschlagen: %s", exc)
            return False
        if not isinstance(dev, dict):
            return False
        max_output_channels = int(dev.get("max_output_channels", 0) or 0)
        default_samplerate = float(dev.get("default_samplerate", 0.0) or 0.0)
        return max_output_channels > 0 and default_samplerate > 0.0

    @property
    def is_playing(self) -> bool:
        return self._active and not self._stop_after_fade

    @property
    def position_frac(self) -> float:
        """Current position as fraction [0.0, 1.0]. Returns -1.0 when idle."""
        buf = self._buf
        if buf is None or not self._active:
            return -1.0
        n = buf.shape[0]
        if n == 0:
            return 0.0
        return min(self._pos / n, 1.0)

    @property
    def elapsed_seconds(self) -> float:
        sr = self._output_sr
        if sr <= 0 or self._buf is None:
            return 0.0
        return self._pos / sr

    @property
    def duration_seconds(self) -> float:
        return self._source_duration_s

    @property
    def label(self) -> str:
        return self._label

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def play(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        start_frac: float = 0.0,
        label: str = "",
        on_finished: Callable[[], None] | None = None,
    ) -> bool:
        """Start or switch playback.  Crossfades if already playing.

        Parameters
        ----------
        audio : ndarray
            Source audio (any shape/dtype — will be normalised).
        sr : int
            Sample rate of *audio*.
        start_frac : float
            Start position as fraction [0.0, 1.0].
        label : str
            Human-readable label (e.g. "Original", "Restauriert").
        on_finished : callable or None
            Called (in PortAudio thread!) when this source finishes naturally.

        Returns True on success.
        """
        if not _SD_AVAILABLE:
            return False

        # 1. Ensure stream exists (reads self._output_sr).
        with self._lock:
            if not self._ensure_stream():
                return False
            output_sr = self._output_sr

        # 2. Prepare audio data (slow: resampling) — outside lock!
        prepared = self._prepare(audio, sr, output_sr)
        if prepared is None or prepared.shape[0] == 0:
            return False

        n = prepared.shape[0]
        start_idx = int(max(0.0, min(1.0, float(start_frac))) * n)

        # 3. Swap source under lock (fast: only small overlay computation).
        with self._lock:
            self._swap_source(prepared, start_idx, label, on_finished)
            return True

    def stop(self) -> None:
        """Stop playback with a short fade-out (~8 ms)."""
        with self._lock:
            if not self._active or self._buf is None:
                self._active = False
                return

            pos = self._pos
            buf = self._buf
            remain = min(_CROSSFADE_SAMPLES, buf.shape[0] - pos)
            if remain > 1:
                # Build fade-out overlay: gradually subtracts the signal to zero.
                seg = buf[pos : pos + remain]
                # factor goes 0 → −1  ⇒  output = signal + signal*factor = signal*(1+factor)
                # At start: output = signal*(1+0) = signal   (full)
                # At end:   output = signal*(1−1) = 0        (silent)
                factor = np.linspace(0.0, -1.0, remain, dtype=np.float32)
                if seg.ndim == 2:
                    factor = factor[:, np.newaxis]
                self._cf_overlay = np.ascontiguousarray(seg * factor, dtype=np.float32)
                self._cf_pos = 0
                self._stop_after_fade = True
            else:
                self._active = False

    def seek(self, frac: float) -> None:
        """Seek to *frac* [0.0, 1.0] with crossfade from current position."""
        with self._lock:
            buf = self._buf
            if buf is None:
                return
            n = buf.shape[0]
            new_pos = int(max(0.0, min(1.0, float(frac))) * n)

            if self._active and self._pos < n:
                old_pos = self._pos
                cf_len = min(_CROSSFADE_SAMPLES, n - new_pos, n - old_pos)
                if cf_len > 1:
                    old_seg = buf[old_pos : old_pos + cf_len]
                    new_seg = buf[new_pos : new_pos + cf_len]
                    fade = np.linspace(1.0, 0.0, cf_len, dtype=np.float32)
                    if old_seg.ndim == 2:
                        fade = fade[:, np.newaxis]
                    self._cf_overlay = np.ascontiguousarray((old_seg - new_seg) * fade, dtype=np.float32)
                    self._cf_pos = 0

            self._pos = new_pos
            self._active = True
            self._stop_after_fade = False

    def invalidate_cache(self) -> None:
        """Löscht the resample cache (call after loading a new file)."""
        self._resample_cache.clear()

    def shutdown(self) -> None:
        """Release all resources.  Call from ``closeEvent``."""
        with self._lock:
            self._active = False
            self._buf = None
            self._cf_overlay = None
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    logger.warning("audio_player.py::Herunterfahren Ersatzpfad", exc_info=True)
                self._stream = None
            self._resample_cache.clear()

    # ------------------------------------------------------------------
    # PortAudio callback (runs in dedicated audio thread)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Fill *outdata* with next *frames* of audio.

        **Lock-free design** — this callback runs in the PortAudio real-time
        thread and must NEVER block on a Python lock.  Under heavy ML/DSP
        load the GIL is already contended; adding a lock here causes
        buffer underruns → audible stuttering.

        Safety contract with writers (play/stop/seek):
          * Writers hold ``self._lock`` among themselves.
          * Writers assign ``self._buf`` LAST — so the callback either sees
            the old consistent (buf, pos) pair or the new one.
          * ``self._pos`` is a Python int (atomic read/write under GIL).
          * ``self._cf_overlay`` is either None or a valid ndarray ref.
        """
        _should_fire = False

        # --- snapshot read (no lock) ---
        buf = self._buf
        if not self._active or buf is None:
            outdata[:] = 0.0
            _should_fire = self._fire_finished
            self._fire_finished = False
        else:
            pos = self._pos
            n = buf.shape[0]

            if pos >= n:
                outdata[:] = 0.0
                self._active = False
                _should_fire = True
            else:
                end = min(pos + frames, n)
                valid = end - pos
                outdata[:valid] = buf[pos:end]
                if valid < frames:
                    outdata[valid:] = 0.0
                self._pos = end

                # --- Crossfade overlay (additive) ---
                cf = self._cf_overlay
                if cf is not None:
                    cf_pos = self._cf_pos
                    cf_end = min(cf_pos + valid, cf.shape[0])
                    cf_valid = cf_end - cf_pos
                    if cf_valid > 0:
                        outdata[:cf_valid] += cf[cf_pos:cf_end]
                        np.clip(outdata[:cf_valid], -1.0, 1.0, out=outdata[:cf_valid])
                    self._cf_pos = cf_end
                    if cf_end >= cf.shape[0]:
                        self._cf_overlay = None
                        if self._stop_after_fade:
                            self._active = False
                            self._stop_after_fade = False
                            _should_fire = True

                # Natural end of source
                if valid < frames and not self._stop_after_fade:
                    self._active = False
                    _should_fire = True

        # Fire finished callback outside critical section to avoid deadlock
        if _should_fire:
            self._fire_finished = False
            cb = self._on_finished
            if cb is not None:
                try:
                    cb()
                except Exception:
                    logger.warning("audio_player.py::_audio_callback Ersatzpfad", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _swap_source(
        self,
        prepared: np.ndarray,
        start_idx: int,
        label: str,
        on_finished: Callable[[], None] | None,
    ) -> None:
        """Replace current source with crossfade.  Lock MUST be held.

        IMPORTANT: ``self._buf`` is assigned **last** so the lock-free
        callback never sees a stale (buf, pos) combination.
        """
        was_active = self._active and not self._stop_after_fade
        old_buf = self._buf
        old_pos = self._pos

        cf_overlay: np.ndarray | None = None

        if was_active and old_buf is not None and old_pos < old_buf.shape[0]:
            # Compute crossfade overlay: (old − new) × fade_out
            # Result gets *added* to the new source in the callback.
            # At t=0: out = new + (old−new)·1 = old   (continuity)
            # At t=N: out = new + (old−new)·0 = new   (converged)
            cf_len = min(
                _CROSSFADE_SAMPLES,
                old_buf.shape[0] - old_pos,
                prepared.shape[0] - start_idx,
            )
            if cf_len > 1:
                old_seg = old_buf[old_pos : old_pos + cf_len]
                new_seg = prepared[start_idx : start_idx + cf_len]
                fade = np.linspace(1.0, 0.0, cf_len, dtype=np.float32)
                if old_seg.ndim == 2:
                    fade = fade[:, np.newaxis]
                cf_overlay = np.ascontiguousarray((old_seg - new_seg) * fade, dtype=np.float32)

        # --- Assign _buf LAST (lock-free callback safety) ---
        self._pos = start_idx
        self._active = True
        self._stop_after_fade = False
        self._cf_overlay = cf_overlay
        self._cf_pos = 0
        self._source_duration_s = prepared.shape[0] / max(1, self._output_sr)
        self._label = label
        self._on_finished = on_finished
        self._fire_finished = False
        self._buf = prepared  # LAST — makes (buf, pos) visible atomically

    def _ensure_stream(self) -> bool:
        """Erstellt output stream if not yet open.  Lock MUST be held."""
        if self._stream is not None:
            try:
                if self._stream.active or not self._stream.closed:
                    return True
            except Exception:
                logger.warning("audio_player.py::_ensure_stream Ersatzpfad", exc_info=True)
            # Stream dead — recreate
            self._stream = None

        if sd is None:
            return False

        # Detect device SR
        try:
            dev = sd.query_devices(kind="output")
            dev_sr = int(round(float(dev.get("default_samplerate", 48000.0)))) if isinstance(dev, dict) else 48000
        except Exception:
            dev_sr = 48000
        if dev_sr <= 0:
            dev_sr = 48000

        self._output_sr = dev_sr
        self._channels = 2

        try:
            self._stream = sd.OutputStream(
                samplerate=dev_sr,
                channels=2,
                dtype="float32",
                callback=self._audio_callback,
                latency=0.150,  # 150 ms explicit — PulseAudio/PipeWire L/R sync (no "high")
                blocksize=512,  # 512 frames (~10.7 ms @ 48 kHz) — GIL-tolerant, L/R phase-safe
            )
            self._stream.start()
            logger.debug("StreamingAudioPlayer: stream opened @ %d Hz, latency=150ms", dev_sr)
            return True
        except Exception as exc:
            logger.warning("StreamingAudioPlayer: stream creation fehlgeschlagen: %s", exc)
            self._stream = None
            return False

    def _prepare(self, audio: np.ndarray, sr: int, output_sr: int) -> np.ndarray | None:
        """Normalise + resample audio to output SR.  NOT locked (slow!)."""
        # Cache lookup — uses content fingerprint (first 4 KB) + shape to avoid
        # false-positive hits from GC-reused memory addresses (§v10.131 BUG-012).
        try:
            _fp = (audio.shape, hash(audio.data.tobytes()[:4096]) if audio.nbytes else 0, sr, output_sr)
        except Exception:
            _fp = None
        if _fp is not None:
            for entry in self._resample_cache:
                if entry[0] == _fp:
                    return entry[1]

        try:
            data = np.asarray(audio, dtype=np.float32)
        except Exception:
            logger.warning("audio_player.py::_prepare Ersatzpfad", exc_info=True)
            return None

        # Shape normalisation: (channels, samples) → (samples, channels)
        if data.ndim == 2 and data.shape[0] <= 2 and data.shape[1] > data.shape[0]:
            data = data.T
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim == 2 and data.shape[1] == 1:
            # Mono → duplicate to stereo for consistent channel count
            data = np.concatenate([data, data], axis=1)
        if data.ndim == 2 and data.shape[1] > 2:
            data = data[:, :2]

        # Sanitise
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -1.0, 1.0)

        # Resample to device SR if needed.
        # soxr: pure-C, ~1s for 4 min stereo at 48→44.1 kHz.
        # scipy.signal.resample_poly(160, 147) hangs 6+ min → VERBOTEN here.
        if output_sr > 0 and abs(sr - output_sr) >= 2000:
            try:
                import soxr as _soxr_player

                data = _soxr_player.resample(data, sr, output_sr, quality="HQ").astype(np.float32)
            except Exception as exc:
                logger.debug("soxr resample fehlgeschlagen (%s), trying resample_poly", exc)
                try:
                    from scipy.signal import resample_poly

                    g = math.gcd(sr, output_sr)
                    data = resample_poly(data, output_sr // g, sr // g, axis=0).astype(np.float32)
                except Exception as exc2:
                    logger.debug("resample_poly also fehlgeschlagen: %s — playing at source SR", exc2)
                    # Fallback: play at wrong SR (slight pitch shift, better than silence)

        data = np.ascontiguousarray(data, dtype=np.float32)

        # Update cache (FIFO eviction) — stores (fingerprint, data) pairs
        if len(self._resample_cache) >= self._MAX_CACHE:
            self._resample_cache.pop(0)
        if _fp is not None:
            self._resample_cache.append((_fp, data))

        return data


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_instance: StreamingAudioPlayer | None = None
_instance_lock = threading.Lock()


def get_streaming_player() -> StreamingAudioPlayer:
    """Gibt the global ``StreamingAudioPlayer`` singleton (thread-safe) zurück."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = StreamingAudioPlayer()
    return _instance
