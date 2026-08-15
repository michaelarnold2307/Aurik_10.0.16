"""
backend/core/musical_structure_analyzer.py — MusicalStructureAnalyzer (Aurik 10.0.0 §2.17)
===========================================================================
SSM-gestützte Segmentstruktur-Erkennung (Intro/Verse/Chorus/Bridge/Outro).
Implementierung gemäß Spec §2.17: Self-Similarity-Matrix, Novelty-Kurve, Foote 2000.
"""

from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from backend.core.core_utils import fft_autocorr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class SegmentInfo:
    """Einzelnes musikalisches Segment."""

    label: str  # "intro" | "verse" | "chorus" | "bridge" | "outro" | "unknown"
    start_sample: int = 0
    end_sample: int = 0
    start_s: float = 0.0
    end_s: float = 0.0
    repeat_count: int = 0
    ssm_similarity: float = 0.0

    @property
    def duration_s(self) -> float:
        """Gibt non-negative segment duration in seconds zurück."""
        return max(0.0, self.end_s - self.start_s)

    @property
    def start_time_s(self) -> float:
        """Alias für start_s (Test-Kompatibilität)."""
        return self.start_s

    @property
    def end_time_s(self) -> float:
        """Alias für end_s (Test-Kompatibilität)."""
        return self.end_s


@dataclass
class MusicalStructure:
    """Vollständige Segmentstruktur einer Aufnahme (§2.17)."""

    boundaries_samples: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
    # Erweiterte Felder — werden vom Analyzer befüllt (§2.17)
    segments: list[SegmentInfo] = field(default_factory=list)
    total_duration_s: float = 0.0
    bpm: float = 0.0
    # Direkt setzbare Segment-Listen für Tests und externe Aufrufe
    chorus_segments: list[SegmentInfo] = field(default_factory=list)
    verse_segments: list[SegmentInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class MusicalStructureAnalyzer:
    """SSM-basierte musikalische Struktur-Erkennung (§2.17).

    Für Dateien < 20 s: leere Segment-Liste (kein Phrasen-Prior).
    Für längere Dateien: Segmentierung alle 8 s mit Novelty-basierter Klassifikation.
    """

    MIN_DURATION_S: float = 20.0  # Mindestlänge für Segmentierung
    MAX_SEGMENTS: int = 200  # Spec §2.17: maximal 200 Segmente

    # SSM analysis parameters (Foote 2000, Müller FMP §4.4)
    _SSM_HOP_S: float = 0.5  # chroma hop: 0.5 s per frame
    _SSM_KERNEL_SIZE: int = 8  # checkerboard kernel half-size (in frames)
    _SSM_NOVELTY_SIGMA: float = 3.0  # Gaussian smoothing of novelty curve (frames)
    _MIN_SEG_S: float = 4.0  # minimum segment duration in seconds

    def analyze(self, audio: np.ndarray, sr: int) -> MusicalStructure:
        """Analysiert Audio und gibt die musikalische Segmentstruktur zurück."""
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = self._to_mono(arr)
        n = mono.shape[0]
        sr = max(1, sr)
        duration_s = n / sr

        base = MusicalStructure(
            boundaries_samples=[],
            labels=[],
            confidence=0.0,
            metadata=self._structure_metadata([]),
            segments=[],
            total_duration_s=float(duration_s),
            bpm=0.0,
        )

        if n == 0 or duration_s < self.MIN_DURATION_S:
            return base

        # BPM estimation (independent of segmentation)
        bpm = self._estimate_bpm(mono, sr)

        # Attempt SSM-based segmentation (Foote 2000)
        bounds, conf = self._ssm_segment(mono, sr, duration_s)

        # Fall back to uniform 8 s slicing if SSM yields too few / too many bounds
        if len(bounds) < 3 or len(bounds) - 1 > self.MAX_SEGMENTS:
            bounds, conf = self._uniform_segment(n, sr, duration_s)
        bounds = self._normalize_boundaries(bounds, n)

        labels = self._classify_segments(bounds, mono, sr)

        # Build SegmentInfo objects
        segs: list[SegmentInfo] = []
        for i, label in enumerate(labels):
            s_samp = bounds[i]
            e_samp = bounds[i + 1]
            segs.append(
                SegmentInfo(
                    label=label,
                    start_s=float(s_samp) / sr,
                    end_s=float(e_samp) / sr,
                    start_sample=s_samp,
                    end_sample=e_samp,
                )
            )
        self._annotate_segment_similarity(segs, mono, sr)
        self._refine_labels_with_similarity(segs)
        labels = [segment.label for segment in segs]

        chorus_segs = [s for s in segs if s.label == "chorus"]
        verse_segs = [s for s in segs if s.label == "verse"]

        return MusicalStructure(
            boundaries_samples=bounds,
            labels=labels,
            confidence=conf,
            metadata=self._structure_metadata(segs),
            segments=segs,
            total_duration_s=float(duration_s),
            bpm=float(bpm),
            chorus_segments=chorus_segs,
            verse_segments=verse_segs,
        )

    # ------------------------------------------------------------------
    # SSM-based segmentation (Foote 2000 / Müller FMP §4.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Gibt mono audio from channel-first or channel-last arrays zurück."""
        if audio.ndim != 2:
            return audio.reshape(-1)
        rows, cols = audio.shape
        if rows <= 8 and cols > rows:
            return audio.mean(axis=0)  # type: ignore[no-any-return]
        if cols <= 8 and rows > cols:
            return audio.mean(axis=1)  # type: ignore[no-any-return]
        return audio.mean(axis=0)  # type: ignore[no-any-return]

    def _ssm_segment(self, mono: np.ndarray, sr: int, duration_s: float) -> tuple[list[int], float]:
        """Self-Similarity-Matrix novelty segmentation.

        Returns:
            (boundary_sample_list, confidence)
        """
        hop = max(1, int(sr * self._SSM_HOP_S))
        n = len(mono)

        # 1. Chroma features (STFT-based, no librosa dependency required)
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=hop, bins_per_octave=36).astype(
                np.float32
            )  # (12, T)
        except Exception:
            try:
                chroma = self._stft_chroma(mono, sr, hop)
            except Exception as e:
                logger.warning("musical_structure_analyzer.py::_ssm_segment Ersatzpfad: %s", e)
                return self._uniform_segment(n, sr, duration_s)

        n_frames = chroma.shape[1]
        if n_frames < 2 * self._SSM_KERNEL_SIZE + 4:
            return self._uniform_segment(n, sr, duration_s)

        # 2. Normalise each chroma frame to unit length
        col_norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
        chroma_n = chroma / col_norms  # (12, T)

        # 3. Self-Similarity Matrix (cosine)
        ssm = chroma_n.T @ chroma_n  # (T, T)  values in [-1, 1]
        ssm = np.clip((ssm + 1.0) / 2.0, 0.0, 1.0)  # rescale to [0, 1]

        # 4. Checkerboard kernel novelty (Foote 2000)
        novelty = self._checkerboard_novelty(ssm, self._SSM_KERNEL_SIZE)

        # 5. Gaussian smooth
        novelty = self._gauss_smooth(novelty, self._SSM_NOVELTY_SIGMA)

        # 6. Peak picking — frame positions of segment boundaries
        min_seg_frames = max(2, int(self._MIN_SEG_S / self._SSM_HOP_S))
        boundary_frames = self._pick_peaks(novelty, min_dist=min_seg_frames)

        # Always include start and end
        boundary_frames = sorted({0, *boundary_frames, n_frames})
        boundary_frames = [b for b in boundary_frames if 0 <= b <= n_frames]

        # Convert frames → samples
        bounds_samples: list[int] = []
        for f in boundary_frames:
            samp = min(n, max(0, f * hop))
            bounds_samples.append(samp)
        if not bounds_samples or bounds_samples[-1] != n:
            bounds_samples.append(n)
        if bounds_samples[0] != 0:
            bounds_samples.insert(0, 0)

        # Deduplicate while preserving order
        seen: set[int] = set()
        bounds_unique: list[int] = []
        for b in bounds_samples:
            if b not in seen:
                seen.add(b)
                bounds_unique.append(b)
        bounds_unique.sort()

        # Confidence: ratio of novelty peak variance (higher = clearer structure)
        if novelty.std() > 0:
            conf = float(np.clip(novelty.std() * 4.0 + 0.5, 0.0, 1.0))
        else:
            conf = 0.4
        conf = float(np.clip(conf + min(0.3, duration_s / 180.0), 0.0, 1.0))
        return bounds_unique, conf

    @classmethod
    def _normalize_boundaries(cls, bounds: list[int], n_samples: int) -> list[int]:
        """Gibt sorted unique boundaries including 0 and n_samples zurück."""
        n_samples = max(0, int(n_samples))
        cleaned = sorted({max(0, min(n_samples, int(bound))) for bound in bounds})
        if not cleaned or cleaned[0] != 0:
            cleaned.insert(0, 0)
        if cleaned[-1] != n_samples:
            cleaned.append(n_samples)
        cleaned = sorted(set(cleaned))
        if len(cleaned) - 1 <= cls.MAX_SEGMENTS:
            return cleaned
        indexes = np.linspace(0, len(cleaned) - 1, cls.MAX_SEGMENTS + 1)
        reduced = [cleaned[int(round(index))] for index in indexes]
        reduced[0] = 0
        reduced[-1] = n_samples
        return sorted(set(reduced))

    @staticmethod
    def _checkerboard_novelty(ssm: np.ndarray, kernel_half_size: int = 8) -> np.ndarray:
        """Wendet Schachbrett-Kernel auf SSM an und gibt Novelty-Kurve zurück.

        Foote (2000): kernel is +1 on the diagonal blocks, -1 on off-diagonal.
        """
        n = ssm.shape[0]
        k = max(1, int(kernel_half_size))
        novelty = np.zeros(n, dtype=np.float32)

        # Build +1/-1 Gaussian-tapered checkerboard kernel
        g = np.exp(-(np.arange(-k, k + 1) ** 2) / (2.0 * (k / 2.0) ** 2))
        kernel = np.outer(g, g)
        mask = np.ones((2 * k + 1, 2 * k + 1), dtype=np.float32)
        mask[:k, k + 1 :] = -1.0
        mask[k + 1 :, :k] = -1.0
        mask[:k, :k] = 1.0
        mask[k + 1 :, k + 1 :] = 1.0
        mask[k, :] = 0.0
        mask[:, k] = 0.0
        kernel = kernel * mask

        for t in range(k, n - k):
            block = ssm[t - k : t + k + 1, t - k : t + k + 1]
            novelty[t] = float(np.sum(block * kernel))

        # Clip negative values (only peaks matter)
        novelty = np.clip(novelty, 0.0, None)
        return novelty  # type: ignore[no-any-return]

    @staticmethod
    def _gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
        """1-D Gaussian smoothing via convolution."""
        if sigma <= 0:
            return x
        half = max(1, int(3 * sigma))
        t = np.arange(-half, half + 1)
        kernel = np.exp(-(t**2) / (2 * sigma**2)).astype(np.float32)
        kernel /= kernel.sum()
        return np.convolve(x, kernel, mode="same").astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _pick_peaks(novelty: np.ndarray, min_dist: int = 8) -> list[int]:
        """Einfaches peak picking with minimum distance constraint."""
        n = len(novelty)
        if n == 0:
            return []
        threshold = novelty.mean() + 0.5 * novelty.std()
        peaks: list[int] = []
        last_peak = -min_dist - 1
        for i in range(1, n - 1):
            if (
                novelty[i] > threshold
                and novelty[i] >= novelty[i - 1]
                and novelty[i] >= novelty[i + 1]
                and (i - last_peak) >= min_dist
            ):
                peaks.append(i)
                last_peak = i
        return peaks

    @staticmethod
    def _stft_chroma(mono: np.ndarray, sr: int, hop: int) -> np.ndarray:
        """Minimal STFT-based chroma (fallback, no librosa)."""
        from scipy.signal import stft  # pylint: disable=import-outside-toplevel

        n_fft = max(4096, min(max(1, mono.size), hop * 2))
        noverlap = max(0, min(n_fft - 1, n_fft - hop))
        _, _, Zxx = stft(mono, sr, nperseg=n_fft, noverlap=noverlap, boundary="even")
        mag = np.abs(Zxx).astype(np.float32)
        n_freqs, n_frames = mag.shape
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)
        chroma = np.zeros((12, n_frames), dtype=np.float32)
        for k, f in enumerate(freqs[:n_freqs]):
            if f < 27.5:
                continue
            pc = int(round(12 * np.log2(f / 440.0 + 1e-12))) % 12
            chroma[pc] += mag[k]
        col_max = np.max(chroma, axis=0, keepdims=True)
        return chroma / (col_max + 1e-8)  # type: ignore[no-any-return]

    @classmethod
    def _uniform_segment(cls, n: int, sr: int, duration_s: float) -> tuple[list[int], float]:
        """Fallback: uniform 8 s slicing."""
        hop = max(1, int(sr * 8.0))
        if n > hop * cls.MAX_SEGMENTS:
            hop = max(1, -(-n // cls.MAX_SEGMENTS))
        bounds = [*range(0, n, hop), n]
        conf = float(np.clip(0.4 + min(0.3, duration_s / 180.0), 0.0, 1.0))
        return bounds, conf

    @staticmethod
    def _classify_segments(bounds: list[int], mono: np.ndarray, sr: int) -> list[str]:
        """Classify segments using energy + position heuristics.

        Rules (priority order):
            1. First segment                 → "intro"
            2. Last segment (< 20 s)         → "outro"
            3. High energy + repeated chroma → "chorus"
            4. Lower energy                  → "verse"
            5. Very short + low energy       → "bridge"
        """
        n_segs = max(0, len(bounds) - 1)
        if n_segs == 0:
            return []

        # Per-segment RMS energy
        energies = np.zeros(n_segs, dtype=np.float32)
        for i in range(n_segs):
            seg = mono[bounds[i] : bounds[i + 1]]
            if len(seg) > 0:
                energies[i] = float(np.sqrt(np.mean(seg**2)))

        energy_mean = float(energies.mean()) if energies.any() else 0.0
        energy_high = float(energy_mean * 1.15)

        labels: list[str] = []
        for i in range(n_segs):
            seg_dur = (bounds[i + 1] - bounds[i]) / max(1, sr)
            if i == 0:
                labels.append("intro")
            elif i == n_segs - 1 and seg_dur < 20.0:
                labels.append("outro")
            elif energies[i] >= energy_high:
                labels.append("chorus")
            elif seg_dur < 6.0:
                labels.append("bridge")
            else:
                labels.append("verse")

        return labels

    @classmethod
    def _annotate_segment_similarity(cls, segments: list[SegmentInfo], mono: np.ndarray, sr: int) -> None:
        """Populate per-segment SSM similarity and repetition count."""
        if len(segments) < 2:
            return
        features = np.vstack([cls._segment_chroma_feature(segment, mono, sr) for segment in segments])
        sim = np.clip(features @ features.T, 0.0, 1.0)
        feature_energy = np.linalg.norm(features, axis=1)
        usable = np.array(
            [cls._is_usable_reference_segment(segment) for segment in segments],
            dtype=bool,
        ) & (feature_energy > 1e-6)
        sim[~usable, :] = 0.0
        sim[:, ~usable] = 0.0
        np.fill_diagonal(sim, 0.0)
        if sim.shape[0] > 1:
            adjacent = np.eye(sim.shape[0], k=1, dtype=bool)
            sim[adjacent | adjacent.T] = 0.0
        for index, segment in enumerate(segments):
            row = sim[index]
            segment.ssm_similarity = float(np.max(row)) if row.size else 0.0
            segment.repeat_count = int(np.count_nonzero(row >= 0.82))

    @staticmethod
    def _refine_labels_with_similarity(segments: list[SegmentInfo]) -> None:
        """Promote stable repeated inner segments to chorus labels."""
        if len(segments) < 3:
            return
        for index, segment in enumerate(segments):
            if index in (0, len(segments) - 1):
                continue
            if segment.label in {"intro", "outro"}:
                continue
            if segment.repeat_count >= 1 and segment.ssm_similarity >= 0.82:
                segment.label = "chorus"

    @staticmethod
    def _segment_chroma_feature(segment: SegmentInfo, mono: np.ndarray, sr: int) -> np.ndarray:
        """Gibt a NaN-safe 12-bin chroma fingerprint for one segment zurück."""
        start = max(0, min(int(segment.start_sample), mono.size))
        end = max(start, min(int(segment.end_sample), mono.size))
        frame = np.nan_to_num(mono[start:end].astype(np.float32, copy=False))
        if frame.size < max(32, sr // 20):
            return np.zeros(12, dtype=np.float32)  # type: ignore[no-any-return]

        max_samples = max(1024, min(frame.size, int(sr * 4.0)))
        if frame.size > max_samples:
            offset = (frame.size - max_samples) // 2
            frame = frame[offset : offset + max_samples]

        window = np.hanning(frame.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
        freqs = np.fft.rfftfreq(frame.size, d=1.0 / max(1, sr)).astype(np.float32)
        valid = (freqs >= 27.5) & (freqs <= min(float(sr) * 0.5, 5000.0))
        chroma = np.zeros(12, dtype=np.float32)
        if np.any(valid):
            pitch_classes = np.rint(12.0 * np.log2(freqs[valid] / 440.0 + 1e-12)).astype(np.int32) % 12  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
            np.add.at(chroma, pitch_classes, spectrum[valid])
        norm = float(np.linalg.norm(chroma))
        if norm < 1e-8:
            return np.zeros(12, dtype=np.float32)  # type: ignore[no-any-return]
        return (chroma / norm).astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _structure_metadata(segments: list[SegmentInfo]) -> dict[str, object]:
        """Gibt compact structural summary metadata zurück."""
        similarities = [s.ssm_similarity for s in segments]
        return {
            "segment_count": len(segments),
            "repeated_segment_count": sum(1 for s in segments if s.repeat_count > 0),
            "mean_ssm_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "max_ssm_similarity": float(np.max(similarities)) if similarities else 0.0,
        }

    CHORUS_CONFIDENCE_MIN: float = 0.75

    def get_reference_segment(
        self,
        gap_start: int,
        structure: MusicalStructure,
    ) -> tuple[int, int] | None:
        """Bestes Referenzsegment für Inpainting (§2.12).

        Gibt None zurück wenn:
        - Keine Segmente vorhanden
        - Konfidenz < CHORUS_CONFIDENCE_MIN (0.75)
        """
        if structure.confidence < self.CHORUS_CONFIDENCE_MIN:
            return None
        chorus = structure.chorus_segments or [s for s in structure.segments if s.label == "chorus"]
        if chorus:
            seg = self._select_reference_candidate(gap_start, chorus, structure)
            if seg is not None:
                return seg.start_sample, seg.end_sample
        # Fallback: nearest segment by start_sample
        if structure.segments:
            s = self._select_reference_candidate(gap_start, structure.segments, structure)
            if s is None:
                return None
            return s.start_sample, s.end_sample
        if structure.boundaries_samples and len(structure.boundaries_samples) >= 2:
            return self._select_boundary_reference(gap_start, structure.boundaries_samples)
        return None

    @staticmethod
    def _select_reference_candidate(
        gap_start: int,
        candidates: list[SegmentInfo],
        structure: MusicalStructure,
    ) -> SegmentInfo | None:
        """Prefer repeated, SSM-stable candidates over plain nearest segments."""
        valid_candidates = [
            segment
            for segment in candidates
            if (
                MusicalStructureAnalyzer._is_usable_reference_segment(segment)
                and (gap_start < segment.start_sample or gap_start >= segment.end_sample)
            )
        ]
        if not valid_candidates:
            return None
        end_hint = max((segment.end_sample for segment in structure.segments), default=0)
        if structure.boundaries_samples:
            end_hint = max(end_hint, structure.boundaries_samples[-1])
        program_len = max(1, end_hint)

        def score(segment: SegmentInfo) -> float:
            distance = abs(segment.start_sample - gap_start) / program_len
            repetition = min(float(segment.repeat_count), 4.0) * 0.5
            similarity = float(np.clip(segment.ssm_similarity, 0.0, 1.0))
            return repetition + similarity - distance

        return max(valid_candidates, key=score)

    @staticmethod
    def _is_usable_reference_segment(segment: SegmentInfo) -> bool:
        """Gibt True when a segment can safely act as repair reference zurück."""
        if segment.start_sample < 0 or segment.end_sample <= segment.start_sample:
            return False
        if segment.duration_s > 0.0 and segment.duration_s < 0.25:
            return False
        return True

    @staticmethod
    def _select_boundary_reference(gap_start: int, boundaries: list[int]) -> tuple[int, int] | None:
        """Wählt aus: a valid boundary interval that does not contain gap_start."""
        intervals = [
            (start, end)
            for start, end in itertools.pairwise(boundaries)
            if MusicalStructureAnalyzer._is_boundary_reference_valid(gap_start, start, end)
        ]
        if not intervals:
            return None
        return min(intervals, key=lambda interval: abs(interval[0] - gap_start))

    @staticmethod
    def _is_boundary_reference_valid(gap_start: int, start: int, end: int) -> bool:
        """Gibt True for positive boundary intervals outside the gap point zurück."""
        if start < 0 or end <= start:
            return False
        return gap_start < start or gap_start >= end

    @staticmethod
    def _estimate_bpm(mono: np.ndarray, sr: int) -> float:
        """Einfache BPM-Schätzung via Energie-Onset-Autokorrelation."""
        if mono.size < sr:
            return 120.0
        try:
            hop = 512
            frame_e = np.array(
                [float(np.sum(mono[i : i + hop] ** 2)) for i in range(0, len(mono) - hop, hop)], dtype=np.float32
            )
            if frame_e.size < 4:
                return 120.0
            frame_e_centered = frame_e - frame_e.mean()
            ac = fft_autocorr(frame_e_centered)
            min_lag = max(1, int(sr * 60 / (200 * hop)))
            max_lag = min(ac.size - 1, int(sr * 60 / (60 * hop)))
            if max_lag <= min_lag:
                return 120.0
            peak_lag = int(np.argmax(ac[min_lag:max_lag])) + min_lag
            bpm = 60.0 * sr / (peak_lag * hop)
            return float(np.clip(bpm, 40.0, 240.0))
        except Exception as e:
            logger.warning("musical_structure_analyzer.py::_estimate_bpm Ersatzpfad: %s", e)
            return 120.0


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: MusicalStructureAnalyzer | None = None
_lock = threading.Lock()


def get_musical_structure_analyzer() -> MusicalStructureAnalyzer:
    """Thread-sicherer Singleton (Double-Checked Locking, §3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MusicalStructureAnalyzer()
    return _instance


def analyze_musical_structure(audio: np.ndarray, sr: int) -> MusicalStructure:
    """Convenience-Wrapper (§3.2)."""
    return get_musical_structure_analyzer().analyze(audio, sr)


__all__ = [
    "MusicalStructure",
    "MusicalStructureAnalyzer",
    "SegmentInfo",
    "analyze_musical_structure",
    "get_musical_structure_analyzer",
]
