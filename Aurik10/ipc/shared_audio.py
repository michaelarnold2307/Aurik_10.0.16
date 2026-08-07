"""Shared-Audio-Ringpuffer für echte Live-Preview während der Verarbeitung.

Lock-freier Single-Producer (Pipeline-Prozess/Thread) / Single-Consumer (GUI) Ringpuffer.
Nutzt multiprocessing.shared_memory für prozessübergreifenden Zugriff ohne Kopien.

Design:
  - Feste Anzahl von Frames (RING_SIZE), jeder mit fester Länge (FRAME_SAMPLES)
  - write_idx / read_idx als atomare 32-Bit-Counter im Shared-Memory-Header
  - Metadata-Block: Sample-Rate, aktuelle Phase-ID, Status-Flags
  - Lock-frei: Producer updated write_idx nach vollständigem Frame-Schreibvorgang als Commit
"""

from __future__ import annotations

import logging
import struct
import threading
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Layout-Konstanten ──────────────────────────────────────────────────────────
SAMPLE_RATE_DEFAULT: int = 48000
CHANNELS: int = 2
FRAME_DURATION_S: float = 2.0
FRAME_SAMPLES: int = int(FRAME_DURATION_S * SAMPLE_RATE_DEFAULT * CHANNELS)  # 192000
FRAME_BYTES: int = FRAME_SAMPLES * 4  # float32 = 4 bytes → 768,000
RING_SIZE: int = 16
META_SIZE: int = 128
RING_OFFSET: int = META_SIZE
TOTAL_SIZE: int = META_SIZE + RING_SIZE * FRAME_BYTES  # ~12.3 MB

MAGIC: int = 0x4155524B  # "AURK"

# Metadata-Layout (little-endian, 128 bytes total):
#   0:  magic       (I, 4 bytes)  — 0x4155524B
#   4:  sample_rate (I, 4 bytes)
#   8:  write_idx   (I, 4 bytes)  — monotonisch steigend, % RING_SIZE für Slot
#  12:  read_idx    (I, 4 bytes)  — monotonisch steigend
#  16:  status      (I, 4 bytes)  — 0=idle, 1=running, 2=finished, 3=error
#  20:  frame_count (I, 4 bytes)
#  24:  channels    (I, 4 bytes)
#  28:  phase_id    (64s, 64 bytes) — UTF-8, null-terminated
#  92:  reserved    (36 bytes)

META_STRUCT = struct.Struct("<I I I I I I I 64s")
META_PHASE_ID_LEN = 64


@dataclass
class AudioFrame:
    """Audio-Frame aus dem Ringpuffer (Consumer-seitig gelesen)."""

    audio: np.ndarray  # float32 [samples, channels]
    sample_rate: int
    phase_id: str
    frame_index: int  # monotonisch steigend, ab 0


@dataclass
class RingInfo:
    """Statische Layout-Information für Debug/Diagnose."""

    magic: int = MAGIC
    meta_size: int = META_SIZE
    ring_offset: int = RING_OFFSET
    frame_bytes: int = FRAME_BYTES
    ring_size: int = RING_SIZE
    total_bytes: int = TOTAL_SIZE
    channels: int = CHANNELS
    frame_samples: int = FRAME_SAMPLES
    sample_rate_default: int = SAMPLE_RATE_DEFAULT


class SharedAudioRing:
    """Lock-freier Ringpuffer für Audio-Daten zwischen Pipeline und GUI.

    Producer (Pipeline) ruft write(audio, sr, phase_id).
    Consumer (GUI) ruft poll() → list[AudioFrame] neu seit letztem poll(), oder read_latest().

    Thread/prozess-sicher: write_idx und read_idx sind im Shared-Memory und
    werden nur vom jeweiligen Owner geschrieben (SPSC-Pattern).
    """

    def __init__(self, name: str = "aurik_audio_ring", create: bool = False):
        self._name = name
        self._create = create
        self._shm: SharedMemory | None = None
        self._buf: memoryview | None = None
        self._owned: bool = create

        if create:
            self._create_segment()
        else:
            self._open_segment()

        # Lokale Caches
        self._local_read_idx: int = -1  # was wir zuletzt gelesen haben

    # ── Segment-Verwaltung ─────────────────────────────────────────────────────

    def _create_segment(self) -> None:
        try:
            self._shm = SharedMemory(name=self._name, create=False)
            self._shm.close()
            self._shm = SharedMemory(name=self._name, create=False)
            self._buf = memoryview(self._shm.buf)  # type: ignore[arg-type]
            self._owned = False
            logger.info("SharedAudioRing: bestehendes Segment '%s' geöffnet", self._name)
            self._validate()
            return
        except FileNotFoundError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        self._shm = SharedMemory(name=self._name, create=True, size=TOTAL_SIZE)
        self._buf = memoryview(self._shm.buf)  # type: ignore[arg-type]
        self._init_header()
        logger.info("SharedAudioRing: Segment '%s' erstellt (%d bytes, %d frames)", self._name, TOTAL_SIZE, RING_SIZE)

    def _open_segment(self) -> None:
        self._shm = SharedMemory(name=self._name, create=False)
        self._buf = memoryview(self._shm.buf)  # type: ignore[arg-type]
        self._validate()
        logger.info("SharedAudioRing: Segment '%s' geöffnet", self._name)

    def _validate(self) -> None:
        magic = struct.unpack_from("<I", self._buf, 0)[0]  # type: ignore[arg-type]
        if magic != MAGIC:
            raise RuntimeError(f"SharedAudioRing: Magic 0x{magic:08X} != 0x{MAGIC:08X}. Segment korrupt?")

    def _init_header(self) -> None:
        self._pack(0, MAGIC, SAMPLE_RATE_DEFAULT, 0, 0, 0, 0, CHANNELS, b"\x00" * META_PHASE_ID_LEN)

    def _pack(
        self,
        offset: int,
        magic: int,
        sr: int,
        write_idx: int,
        read_idx: int,
        status: int,
        frame_count: int,
        channels: int,
        phase_id: bytes,
    ) -> None:
        META_STRUCT.pack_into(
            self._buf,
            offset,
            magic,
            sr,
            write_idx,
            read_idx,
            status,
            frame_count,
            channels,
            phase_id,  # type: ignore[arg-type]
        )

    def _unpack(self, offset: int = 0):
        return META_STRUCT.unpack_from(self._buf, offset)  # type: ignore[arg-type]

    # ── Feld-Accessoren (direkt im Shared-Memory) ──────────────────────────────

    @property
    def sample_rate(self) -> int:
        if self._buf is None:
            return SAMPLE_RATE_DEFAULT
        return struct.unpack_from("<I", self._buf, 4)[0]  # type: ignore[no-any-return]

    @sample_rate.setter
    def sample_rate(self, v: int) -> None:
        if self._buf:
            struct.pack_into("<I", self._buf, 4, v)

    @property
    def write_idx(self) -> int:
        if self._buf is None:
            return 0
        return struct.unpack_from("<I", self._buf, 8)[0]  # type: ignore[no-any-return]

    @write_idx.setter
    def write_idx(self, v: int) -> None:
        if self._buf:
            struct.pack_into("<I", self._buf, 8, v)

    @property
    def read_idx(self) -> int:
        if self._buf is None:
            return 0
        return struct.unpack_from("<I", self._buf, 12)[0]  # type: ignore[no-any-return]

    @read_idx.setter
    def read_idx(self, v: int) -> None:
        if self._buf:
            struct.pack_into("<I", self._buf, 12, v)

    @property
    def status(self) -> int:
        if self._buf is None:
            return 0
        return struct.unpack_from("<I", self._buf, 16)[0]  # type: ignore[no-any-return]

    @status.setter
    def status(self, v: int) -> None:
        if self._buf:
            struct.pack_into("<I", self._buf, 16, v)

    @property
    def frame_count(self) -> int:
        if self._buf is None:
            return 0
        return struct.unpack_from("<I", self._buf, 20)[0]  # type: ignore[no-any-return]

    @frame_count.setter
    def frame_count(self, v: int) -> None:
        if self._buf:
            struct.pack_into("<I", self._buf, 20, v)

    @property
    def channels_field(self) -> int:
        if self._buf is None:
            return CHANNELS
        return struct.unpack_from("<I", self._buf, 24)[0]  # type: ignore[no-any-return]

    @property
    def phase_id(self) -> str:
        if self._buf is None:
            return ""
        raw = bytes(self._buf[28 : 28 + META_PHASE_ID_LEN])
        end = raw.find(b"\x00")
        if end >= 0:
            raw = raw[:end]
        return raw.decode("utf-8", errors="replace")

    @phase_id.setter
    def phase_id(self, v: str) -> None:
        if self._buf is None:
            return
        encoded = v.encode("utf-8")[: META_PHASE_ID_LEN - 1]
        self._buf[28 : 28 + META_PHASE_ID_LEN] = b"\x00" * META_PHASE_ID_LEN
        self._buf[28 : 28 + len(encoded)] = encoded

    @property
    def info(self) -> RingInfo:
        return RingInfo()

    # ── Producer API ──────────────────────────────────────────────────────────

    def write(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE_DEFAULT, phase_id: str = "") -> int:
        """Schreibt einen Audio-Frame in den Ringpuffer.

        Args:
            audio: float32 ndarray [samples] oder [samples, channels]
            sample_rate: Hz
            phase_id: Aktueller Phasenname für UI

        Returns:
            Frame-Index des geschriebenen Frames (monotonisch)
        """
        if self._buf is None:
            return -1

        audio_f32 = np.asarray(audio, dtype=np.float32)
        if audio_f32.ndim == 1:
            audio_f32 = np.column_stack([audio_f32, audio_f32])
        elif audio_f32.ndim == 2 and audio_f32.shape[1] == 1:
            audio_f32 = np.column_stack([audio_f32[:, 0], audio_f32[:, 0]])
        # §v10.31: Normalize zu channels-last (N,2) für korrektes
        # Interleaving via ravel(). Pipeline-intern ist channels-first
        # (2,N), aber .ravel() auf (2,N) ergibt [L0…LN, R0…RN] statt
        # [L0,R0,L1,R1,…]. Transponiere bei Bedarf.
        if audio_f32.ndim == 2 and audio_f32.shape[0] == 2 and audio_f32.shape[1] > 2:
            audio_f32 = audio_f32.T

        flat = audio_f32.ravel()
        n_samples = min(len(flat), FRAME_SAMPLES)
        fc = self.frame_count
        slot = fc % RING_SIZE
        frame_offset = RING_OFFSET + slot * FRAME_BYTES

        self._buf[frame_offset : frame_offset + n_samples * 4] = flat[:n_samples].tobytes()
        if n_samples < FRAME_SAMPLES:
            self._buf[frame_offset + n_samples * 4 : frame_offset + FRAME_BYTES] = b"\x00" * (
                FRAME_BYTES - n_samples * 4
            )

        self.sample_rate = sample_rate
        self.phase_id = phase_id
        if self.status == 0:
            self.status = 1  # running

        # Commit: frame_count und write_idx atomar inkrementieren
        # SPSC: nur Producer schreibt diese Felder
        self.write_idx = fc + 1  # redundanter Spiegel für has_new
        self.frame_count = fc + 1  # primärer Counter

        return self.frame_count

    def signal_finished(self) -> None:
        self.status = 2

    def signal_error(self) -> None:
        self.status = 3

    def reset(self) -> None:
        """Reset für nächste Datei."""
        self.write_idx = 0
        self.read_idx = 0
        self.frame_count = 0
        self.status = 0
        self.phase_id = ""
        self._local_read_idx = -1

    # ── Consumer API ──────────────────────────────────────────────────────────

    def poll(self) -> list[AudioFrame]:
        """Holt alle neuen Frames seit dem letzten poll()-Aufruf. Non-blocking."""
        frames: list[AudioFrame] = []
        if self._buf is None:
            return frames

        fc = self.frame_count  # total frames written
        ri = self._local_read_idx  # last frame index we've seen (-1 = none)

        # Lese alle ungesehenen Frames (0-indexed: frames 0..fc-1 exist)
        for idx in range(ri + 1, fc):
            slot = idx % RING_SIZE
            frame_offset = RING_OFFSET + slot * FRAME_BYTES
            raw = bytes(self._buf[frame_offset : frame_offset + FRAME_BYTES])
            audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, CHANNELS)
            frames.append(
                AudioFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    phase_id=self.phase_id,
                    frame_index=idx,
                )
            )

        if frames:
            self._local_read_idx = frames[-1].frame_index
        return frames

    def read_latest(self) -> AudioFrame | None:
        """Liest nur den neuesten Frame. Non-blocking."""
        if self._buf is None:
            return None
        fc = self.frame_count
        if fc == 0:
            return None

        idx = fc - 1  # 0-indexed, letzter geschriebener Frame
        slot = idx % RING_SIZE
        frame_offset = RING_OFFSET + slot * FRAME_BYTES
        raw = bytes(self._buf[frame_offset : frame_offset + FRAME_BYTES])
        audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, CHANNELS)
        frame = AudioFrame(
            audio=audio,
            sample_rate=self.sample_rate,
            phase_id=self.phase_id,
            frame_index=idx,
        )
        self._local_read_idx = idx
        return frame

    @property
    def has_new(self) -> bool:
        if self._buf is None:
            return False
        return self.frame_count > (self._local_read_idx + 1)

    @property
    def is_running(self) -> bool:
        return self.status == 1

    @property
    def is_finished(self) -> bool:
        return self.status == 2

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Gibt alle lokalen Referenzen frei. Danach ist das Objekt unbrauchbar."""
        if self._buf is not None:
            try:
                self._buf.release()
            except (BufferError, ValueError):
                pass  # bereits freigegeben
            self._buf = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def unlink(self) -> None:
        """Entfernt das Shared-Memory-Segment dauerhaft (nur Owner)."""
        # Zuerst schließen falls noch offen
        self.close()
        # Dann neu öffnen zum Unlinken
        try:
            shm = SharedMemory(name=self._name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)


# ── Testing ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)

    print("=== SharedAudioRing Test Suite ===")

    # Test 1: Erstellen
    ring = SharedAudioRing(name="test_aurik_ring", create=True)
    assert ring.write_idx == 0
    assert ring.status == 0
    assert ring.sample_rate == SAMPLE_RATE_DEFAULT
    print("  ✓ create")

    # Test 2: Schreiben
    for i in range(5):
        audio = np.sin(np.linspace(0, (i + 1) * 440 * 2 * np.pi, 48000, dtype=np.float32))
        idx = ring.write(audio, phase_id=f"phase_{i}")
        assert idx == i + 1, f"Expected {i + 1}, got {idx}"
        time.sleep(0.001)
    assert ring.frame_count == 5
    assert ring.is_running
    print("  ✓ write × 5")

    # Test 3: Lesen via poll()
    frames = ring.poll()
    assert len(frames) == 5, f"Expected 5, got {len(frames)}"
    for i, f in enumerate(frames):
        assert f.frame_index == i
    print("  ✓ poll() → 5 frames")

    # Test 4: Keine doppelten Frames
    frames2 = ring.poll()
    assert len(frames2) == 0, f"Expected 0, got {len(frames2)}"
    print("  ✓ poll() → 0 (no duplicates)")

    # Test 5: read_latest
    latest = ring.read_latest()
    assert latest is not None
    assert latest.frame_index == 4
    print("  ✓ read_latest()")

    # Test 6: has_new
    assert not ring.has_new
    ring.write(np.zeros(1000, dtype=np.float32), phase_id="new")
    assert ring.has_new
    print("  ✓ has_new")

    # Test 7: signal_finished
    ring.signal_finished()
    assert ring.is_finished
    print("  ✓ signal_finished")

    # Test 8: reset
    ring.reset()
    assert ring.write_idx == 0
    assert ring.read_idx == 0
    assert ring.frame_count == 0
    assert ring.status == 0
    print("  ✓ reset")

    # Test 9: Zweiter Consumer (separates SharedMemory-Objekt)
    consumer = SharedAudioRing(name="test_aurik_ring", create=False)
    consumer.write(np.ones(5000, dtype=np.float32), phase_id="from_consumer_test")
    frames_c = consumer.poll()
    assert len(frames_c) == 1
    assert frames_c[0].phase_id == "from_consumer_test"
    consumer.close()
    print("  ✓ consumer on existing segment")

    # Cleanup
    ring.close()
    ring.unlink()

    print("\n=== ALL TESTS PASSED ===")
