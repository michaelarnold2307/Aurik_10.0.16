"""
breath_intelligence.py - Artistic Breath Noise Intelligence (Phase 2.2)

Kunstvoller Umgang mit Atemngeräuschen:
- Context-Aware Detection (Phrase Boundaries, Pauses)
- Artistic Intent Scoring (Genre, Style, Era)
- Musical Processing (Remove, Reduce, Preserve, Enhance)
- Quality Gates (No Voice Damage)

Author: AURIK Development Team
Version: 1.0.0
Date: 9. Februar 2026
"""

import logging
import warnings

import numpy as np
from scipy.signal import butter, hilbert, sosfilt

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class BreathEvent:
    """Represents a breath noise event in audio."""

    def __init__(
        self,
        start_sample: int,
        end_sample: int,
        confidence: float,
        breath_type: str,
        energy: float,
        spectral_centroid: float,
    ):
        """
        Parameters
        ----------
        start_sample : int
            Start position in samples
        end_sample : int
            End position in samples
        confidence : float
            Detection confidence (0.0-1.0)
        breath_type : str
            'inhale', 'exhale', 'gasp', 'sigh'
        energy : float
            RMS energy of breath event
        spectral_centroid : float
            Spectral centroid in Hz
        """
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.confidence = confidence
        self.breath_type = breath_type
        self.energy = energy
        self.spectral_centroid = spectral_centroid

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def duration_seconds(self, sr: int) -> float:
        return self.duration_samples / sr

    def __repr__(self):
        return (
            f"BreathEvent(type={self.breath_type}, "
            f"start={self.start_sample}, end={self.end_sample}, "
            f"confidence={self.confidence:.2f})"
        )


class BreathDetector:
    """
    Advanced breath noise detection using spectral + temporal features.
    """

    def __init__(self, sensitivity: float = 0.7, min_duration_ms: float = 50.0, max_duration_ms: float = 800.0):
        """
        Parameters
        ----------
        sensitivity : float
            Detection sensitivity (0.0-1.0), default 0.7
        min_duration_ms : float
            Minimum breath duration in milliseconds
        max_duration_ms : float
            Maximum breath duration in milliseconds
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms

    def detect(self, audio: np.ndarray, sr: int, vocal_mask: np.ndarray | None = None) -> list[BreathEvent]:
        """
        Erkennt breath events in audio.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sr : int
            Sample rate in Hz
        vocal_mask : np.ndarray, optional
            Boolean mask indicating vocal regions

        Returns
        -------
        events : List[BreathEvent]
            Detected breath events
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute spectral features
        breath_band_energy = self._compute_breath_energy(audio, sr)

        # Compute temporal features
        envelope = self._compute_envelope(audio, sr)

        # Combine features for detection
        detection_signal = breath_band_energy * envelope

        # Adaptive threshold
        threshold = np.percentile(detection_signal, 90) * (1.0 - self.sensitivity)

        # Find breath candidates
        candidates = detection_signal > threshold

        # Apply vocal mask if provided
        if vocal_mask is not None:
            # Breath events should NOT overlap with strong vocal content
            candidates = candidates & (~vocal_mask)

        # Convert to events
        events = self._extract_events(candidates, audio, sr, detection_signal)

        return events

    def _compute_breath_energy(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Berechnet energy in breath frequency band (500-3000 Hz).

        Breath noise is typically noisy, broadband, low-energy, and
        concentrated in the 500-3000 Hz range (less harmonic structure).
        """
        # Bandpass filter for breath band
        nyquist = sr / 2
        low = 500 / nyquist
        high = min(3000 / nyquist, 0.99)

        sos = butter(4, [low, high], btype="bandpass", output="sos")
        breath_filtered = sosfilt(sos, audio)

        # Short-term energy
        window_samples = int(0.02 * sr)  # 20ms windows
        hop_samples = window_samples // 2

        energy = []
        for i in range(0, len(breath_filtered) - window_samples, hop_samples):
            frame = breath_filtered[i : i + window_samples]
            frame_energy = np.sqrt(np.mean(frame**2))
            energy.append(frame_energy)

        energy = np.array(energy)  # type: ignore[assignment]

        # Interpolate to match audio length
        energy = np.interp(np.arange(len(audio)), np.linspace(0, len(audio), len(energy)), energy)  # type: ignore[assignment]

        return energy  # type: ignore[return-value]

    def _compute_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Berechnet amplitude envelope using Hilbert transform.
        Chunked to prevent OOM on long audio (hilbert allocates complex128 FFT).
        """
        max_chunk = 30 * sr  # 30 s chunks — ~46 MB complex128 per chunk
        n = len(audio)
        if n <= max_chunk:
            envelope = np.abs(np.asarray(hilbert(audio), dtype=np.complex128))
        else:
            overlap = int(0.01 * sr)  # 10 ms overlap for smooth joins
            envelope = np.empty(n, dtype=np.float64)
            pos = 0
            while pos < n:
                end = min(pos + max_chunk, n)
                chunk_env = np.abs(hilbert(audio[pos:end]))
                if pos == 0:
                    envelope[pos:end] = chunk_env
                else:
                    # Cross-fade overlap region
                    fade = np.linspace(0.0, 1.0, overlap)
                    envelope[pos : pos + overlap] = (
                        envelope[pos : pos + overlap] * (1.0 - fade) + chunk_env[:overlap] * fade
                    )
                    envelope[pos + overlap : end] = chunk_env[overlap:]
                pos = end - overlap if end < n else n

        # Smooth envelope
        window_samples = int(0.01 * sr)  # 10ms smoothing
        kernel = np.ones(window_samples) / window_samples
        envelope = np.convolve(envelope, kernel, mode="same")

        return envelope

    def _extract_events(
        self, candidates: np.ndarray, audio: np.ndarray, sr: int, detection_signal: np.ndarray
    ) -> list[BreathEvent]:
        """
        Extrahiert discrete breath events from candidate mask.
        """
        events = []

        min_duration_samples = int(self.min_duration_ms * sr / 1000)
        max_duration_samples = int(self.max_duration_ms * sr / 1000)

        # Find connected regions
        labeled, num_regions = self._label_regions(candidates)

        for region_id in range(1, num_regions + 1):
            region_mask = labeled == region_id
            indices = np.where(region_mask)[0]

            if len(indices) == 0:
                continue

            start_sample = indices[0]
            end_sample = indices[-1]
            duration = end_sample - start_sample

            # Duration filter
            if duration < min_duration_samples or duration > max_duration_samples:
                continue

            # Extract features
            breath_segment = audio[start_sample:end_sample]

            # Confidence (mean detection signal strength)
            confidence = np.mean(detection_signal[start_sample:end_sample])
            confidence = np.clip(confidence / np.max(detection_signal), 0.0, 1.0)

            # Energy
            energy = np.sqrt(np.mean(breath_segment**2))

            # Spectral centroid
            spectral_centroid = self._compute_spectral_centroid(breath_segment, sr)

            # Breath type classification (simple heuristic)
            breath_type = self._classify_breath_type(breath_segment, sr, spectral_centroid, energy)

            event = BreathEvent(
                start_sample=start_sample,
                end_sample=end_sample,
                confidence=confidence,
                breath_type=breath_type,
                energy=energy,
                spectral_centroid=spectral_centroid,
            )

            events.append(event)

        return events

    def _label_regions(self, mask: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Label connected regions in binary mask (vectorised).
        """
        mask_bool = mask.astype(bool)
        # Detect region starts: mask[i] is True and either i==0 or mask[i-1] is False
        starts = mask_bool.copy()
        starts[1:] &= ~mask_bool[:-1]
        # Cumulative sum of starts gives region IDs
        labeled = np.cumsum(starts) * mask_bool
        region_id = int(labeled.max()) if len(labeled) > 0 else 0
        return labeled, region_id

    def _compute_spectral_centroid(self, audio: np.ndarray, sr: int) -> float:
        """
        Berechnet spectral centroid.
        """
        # Zero-pad to power of 2
        n_fft = 2 ** int(np.ceil(np.log2(len(audio))))

        # FFT
        spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)

        # Weighted mean
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0.0

        return centroid

    def _classify_breath_type(self, audio: np.ndarray, sr: int, spectral_centroid: float, energy: float) -> str:
        """
        Einfaches breath type classification.

        - Inhale: Higher spectral centroid (>2 kHz), lower energy
        - Exhale: Lower spectral centroid (<1.5 kHz), moderate energy
        - Gasp: High energy, short duration
        - Sigh: Lower energy, longer duration
        """
        duration = len(audio) / sr

        # Simple heuristic classification
        if energy > 0.1 and duration < 0.2:
            return "gasp"
        elif spectral_centroid > 2000:
            return "inhale"
        elif spectral_centroid < 1500 and duration > 0.3:
            return "sigh"
        else:
            return "exhale"


class ArtisticIntentScorer:
    """
    Bewertet artistic intent for breath preservation based on genre, era, and context.
    """

    # Genre-based breath preservation scores (0.0 = remove, 1.0 = preserve)
    GENRE_SCORES = {
        "classical": 0.7,  # Natural breathing is part of interpretation
        "jazz": 0.8,  # Breath is part of phrasing
        "blues": 0.9,  # Expressive breathing is stylistic
        "opera": 0.6,  # Depends on context, moderate preservation
        "pop": 0.3,  # Clean production, less breath
        "rock": 0.4,  # Some breath OK, but not excessive
        "electronic": 0.1,  # Clean, no breath
        "hip-hop": 0.5,  # Depends on style
        "folk": 0.7,  # Natural, organic sound
        "ambient": 0.9,  # Breath can be atmospheric
        "default": 0.5,  # Moderate preservation
    }

    # Era-based scores
    ERA_SCORES = {
        "1920s-1950s": 0.9,  # Historical recordings: preserve authenticity
        "1960s-1970s": 0.7,  # Natural sound was valued
        "1980s-1990s": 0.4,  # Cleaner production
        "2000s-": 0.2,  # Modern: very clean
        "default": 0.5,
    }

    def score_artistic_intent(
        self, genre: str = "default", era: str = "default", context: str = "phrase_boundary"
    ) -> float:
        """
        Bewertet artistic intent for breath preservation.

        Parameters
        ----------
        genre : str
            Musical genre
        era : str
            Recording era
        context : str
            Context: 'phrase_boundary', 'mid_phrase', 'intro', 'outro'

        Returns
        -------
        score : float
            Artistic intent score (0.0-1.0)
            - 0.0 = remove completely
            - 0.5 = reduce moderately
            - 1.0 = preserve/enhance
        """
        # Normalize genre & era
        genre = genre.lower()

        # Base scores
        genre_score = self.GENRE_SCORES.get(genre, self.GENRE_SCORES["default"])
        era_score = self.ERA_SCORES.get(era, self.ERA_SCORES["default"])

        # Context modulation
        context_multiplier = {
            "phrase_boundary": 1.0,  # Breaths between phrases are natural
            "mid_phrase": 0.5,  # Mid-phrase breaths are less desirable
            "intro": 0.3,  # Clean intro
            "outro": 0.7,  # Outro can be more expressive
        }.get(context, 1.0)

        # Weighted average
        score = (genre_score * 0.6 + era_score * 0.4) * context_multiplier

        return np.clip(score, 0.0, 1.0)  # type: ignore[no-any-return]


class BreathProcessor:
    """
    Musical breath processing: Remove, Reduce, Preserve, or Enhance.
    """

    def __init__(self, artistic_intent_scorer: ArtisticIntentScorer | None = None):
        """
        Parameters
        ----------
        artistic_intent_scorer : ArtisticIntentScorer, optional
            Scorer for artistic intent
        """
        self.scorer = artistic_intent_scorer or ArtisticIntentScorer()

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        events: list[BreathEvent],
        genre: str = "default",
        era: str = "default",
        aggressive: float = 0.5,
    ) -> tuple[np.ndarray, dict]:
        """
        Verarbeitet breath events with artistic intent.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz
        events : List[BreathEvent]
            Detected breath events
        genre : str
            Musical genre
        era : str
            Recording era
        aggressive : float
            Processing aggressiveness (0.0-1.0)

        Returns
        -------
        audio_processed : np.ndarray
            Processed audio
        metrics : Dict
            Processing metrics
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle stereo
        if audio.ndim == 2:
            # Auto-detect format: (channels, samples) vs (samples, channels)
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples)
                left = self.process(audio[0], sr, events, genre, era, aggressive)[0]
                right = self.process(audio[1], sr, events, genre, era, aggressive)[0]
                return np.vstack([left, right]), {"stereo": True}
            else:
                # Format: (samples, channels) - transpose for processing
                audio_T = audio.T
                left = self.process(audio_T[0], sr, events, genre, era, aggressive)[0]
                right = self.process(audio_T[1], sr, events, genre, era, aggressive)[0]
                # Return in original format
                return np.column_stack([left, right]), {"stereo": True}

        audio_processed = audio.copy()
        processed_events = 0
        total_reduction_db = 0.0

        for event in events:
            # Score artistic intent
            context = self._determine_context(event, audio, sr)
            intent_score = self.scorer.score_artistic_intent(genre, era, context)

            # Determine processing action
            # intent_score: 0.0=remove, 0.5=reduce, 1.0=preserve
            # aggressive: 0.0=gentle, 1.0=aggressive

            if intent_score < 0.3:
                # Remove (strong reduction)
                reduction_factor = 0.1 + (1.0 - aggressive) * 0.1
            elif intent_score < 0.7:
                # Reduce moderately
                reduction_factor = 0.4 + (1.0 - aggressive) * 0.3
            else:
                # Preserve (minimal reduction or none)
                reduction_factor = 0.7 + (1.0 - aggressive) * 0.3

            # Apply reduction with fade
            start = event.start_sample
            end = event.end_sample

            # Fade in/out (10ms)
            fade_samples = min(int(0.01 * sr), (end - start) // 4)

            # Create reduction envelope
            envelope = np.ones(end - start)
            envelope[:fade_samples] = np.linspace(1.0, reduction_factor, fade_samples)
            envelope[-fade_samples:] = np.linspace(reduction_factor, 1.0, fade_samples)
            envelope[fade_samples:-fade_samples] = reduction_factor

            # Apply
            audio_processed[start:end] *= envelope

            # Metrics
            processed_events += 1
            reduction_db = -20 * np.log10(reduction_factor) if reduction_factor > 0 else 60
            total_reduction_db += reduction_db

        metrics = {
            "events_detected": len(events),
            "events_processed": processed_events,
            "average_reduction_db": total_reduction_db / max(processed_events, 1),
        }

        # NaN/Inf-Guard and clipping
        audio_processed = np.nan_to_num(audio_processed, nan=0.0, posinf=0.0, neginf=0.0)
        audio_processed = np.clip(audio_processed, -1.0, 1.0)

        return audio_processed, metrics

    def _determine_context(self, event: BreathEvent, audio: np.ndarray, sr: int) -> str:
        """
        Bestimmt context of breath event (phrase boundary, mid-phrase, etc.).
        """
        # Simple heuristic: check energy before and after
        context_window = int(0.5 * sr)  # 500ms context

        start = max(0, event.start_sample - context_window)
        end = min(len(audio), event.end_sample + context_window)

        before = audio[start : event.start_sample]
        after = audio[event.end_sample : end]

        energy_before = np.sqrt(np.mean(before**2)) if len(before) > 0 else 0
        energy_after = np.sqrt(np.mean(after**2)) if len(after) > 0 else 0

        # If low energy before AND after → phrase boundary
        if energy_before < 0.01 and energy_after < 0.01:
            return "phrase_boundary"

        # If high energy before AND after → mid-phrase
        if energy_before > 0.05 and energy_after > 0.05:
            return "mid_phrase"

        # If low energy before, high after → intro/start
        if energy_before < 0.01 and energy_after > 0.05:
            return "intro"

        # If high energy before, low after → outro/end
        if energy_before > 0.05 and energy_after < 0.01:
            return "outro"

        return "phrase_boundary"  # Default


class BreathIntelligence:
    """
    Einheitliche API für Atemlärm-Intelligenz.
    """

    def __init__(self, sensitivity: float = 0.7, genre: str = "default", era: str = "default", aggressive: float = 0.5):
        """
        Parameters
        ----------
        sensitivity : float
            Detection sensitivity (0.0-1.0)
        genre : str
            Musical genre
        era : str
            Recording era
        aggressive : float
            Processing aggressiveness (0.0-1.0)
        """
        self.detector = BreathDetector(sensitivity=sensitivity)
        self.processor = BreathProcessor()
        self.genre = genre
        self.era = era
        self.aggressive = aggressive

    def process(self, audio: np.ndarray, sr: int, vocal_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """
        Full breath intelligence pipeline.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz
        vocal_mask : np.ndarray, optional
            Boolean mask for vocal regions

        Returns
        -------
        audio_processed : np.ndarray
            Processed audio
        report : Dict
            Processing report with detection + processing metrics
        """
        # Handle stereo
        if audio.ndim == 2:
            logger.debug("BreathIntelligence.verarbeiten: Eingabe shape %s", audio.shape)
            # Auto-detect format: (channels, samples) vs (samples, channels)
            # Heuristic: If first dimension is small and < second dimension, likely channels
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples) - average over channels (axis 0)
                logger.debug("BreathIntelligence: format (channels, samples), axis=0")
                audio_mono = np.mean(audio, axis=0)
            else:
                # Format: (samples, channels) - average over channels (axis 1)
                logger.debug("BreathIntelligence: format (samples, channels), axis=1")
                audio_mono = np.mean(audio, axis=1)
            logger.debug("BreathIntelligence: mono shape %s", audio_mono.shape)
        else:
            audio_mono = audio

        # Detect breath events
        events = self.detector.detect(audio_mono, sr, vocal_mask)

        # Process breath events
        audio_processed, metrics = self.processor.process(audio, sr, events, self.genre, self.era, self.aggressive)

        report = {
            "events_detected": len(events),
            "events_processed": metrics.get("events_processed", 0),
            "average_reduction_db": metrics.get("average_reduction_db", 0.0),
            "genre": self.genre,
            "era": self.era,
            "aggressive": self.aggressive,
        }

        return audio_processed, report


# CLI interface
if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Breath Intelligence - Artistic breath noise processing")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", help="Output audio file")
    parser.add_argument("--genre", default="default", help="Musical genre")
    parser.add_argument("--era", default="default", help="Recording era")
    parser.add_argument("--sensitivity", type=float, default=0.7, help="Detection sensitivity (0.0-1.0)")
    parser.add_argument("--aggressive", type=float, default=0.5, help="Processing aggressiveness (0.0-1.0)")

    args = parser.parse_args()

    # Load audio
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Process
    breath_intel = BreathIntelligence(
        sensitivity=args.sensitivity, genre=args.genre, era=args.era, aggressive=args.aggressive
    )

    audio_processed, report = breath_intel.process(audio, sr)

    # Print report
    logger.info(str("\n" + "=" * 70))
    logger.info("BREATH INTELLIGENCE REPORT")
    logger.info(str("=" * 70))
    logger.info("Events erkannt:     %s", report["events_detected"])
    logger.info("Events verarbeitet:    %s", report["events_processed"])
    logger.info("Average Reduction:   %.1f dB", report["average_reduction_db"])
    logger.info("Genre:               %s", report["genre"])
    logger.info("Era:                 %s", report["era"])
    logger.info("Aggressiveness:      %.2f", report["aggressive"])
    logger.info(str("=" * 70))

    # Save
    if args.output:
        sf.write(args.output, audio_processed, sr)
        logger.info("\n✅ gespeichert to: %s", args.output)
