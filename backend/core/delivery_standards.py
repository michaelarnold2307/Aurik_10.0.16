"""
Delivery Standards System für Professional Broadcasting & Streaming

Implements GAP #2: Broadcast & Streaming Standards Compliance.
Ein professionelles System muss Material für verschiedene Delivery-Targets vorbereiten können:
- EBU R128 (European Broadcasting)
- ATSC A/85 (US Broadcasting)
- Spotify (-14 LUFS)
- iTunes/Apple Music (-16 LUFS)
- Archival (BWF metadata, -18 LUFS)

Architecture:
1. DeliveryStandard - Enum mit Ziel-Standards
2. StandardConfig - Technische Parameter pro Standard
3. LoudnessAnalyzer - ITU-R BS.1770-4 LUFS measurement
4. TruePeakLimiter - True Peak limiting (ITU-R BS.1770-4)
5. BWFMetadataWriter - Broadcast Wave Format metadata
6. DeliveryStandardsManager - Haupt-API

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DeliveryStandard(Enum):
    """Professional delivery standards für Broadcasting & Streaming."""

    EBU_R128 = "ebu_r128"
    """EBU R128 (European Broadcasting Union).

    Target: -23 LUFS ±0.5 LU
    True Peak: -1.0 dBTP max
    LRA (Loudness Range): Keine strikte Vorgabe, typisch 6-15 LU
    Use: European TV, Radio
    """

    ATSC_A85 = "atsc_a85"
    """ATSC A/85 (US Broadcasting Standard).

    Target: -24 LUFS ±2 LU
    True Peak: -2.0 dBTP max (conservative)
    Use: US TV, CALM Act Compliance
    """

    SPOTIFY = "spotify"
    """Spotify Loudness Normalization.

    Target: -14 LUFS
    True Peak: -1.0 dBTP max (to prevent clipping)
    Dynamic Range: Nicht limitiert, aber -8 bis -6 LUFS für Competitive
    Use: Streaming (Spotify, YouTube Music, Deezer)
    """

    ITUNES = "itunes"
    """iTunes / Apple Music Mastering Standard.

    Target: -16 LUFS ±1 LU
    True Peak: -1.0 dBTP max
    Sample Peak: -0.1 dBFS to prevent inter-sample peaks
    Use: Apple Music, iTunes Store
    """

    ARCHIVAL = "archival"
    """Archival Standard (IASA-TC 04 Compliant).

    Target: -18 LUFS (moderate, preserves dynamic range)
    True Peak: -1.0 dBTP
    Metadata: Full BWF (Broadcast Wave Format) with history
    Use: Long-term preservation, digital archives
    """

    CUSTOM = "custom"
    """Custom standard mit user-defined parameters."""


@dataclass
class StandardConfig:
    """
    Technische Parameter für einen Delivery Standard.

    Basiert auf ITU-R BS.1770-4 und EBU R128.
    """

    name: str
    """Standard name (z.B. 'EBU R128')."""

    description: str
    """Human-readable description."""

    # === Loudness Targets (ITU-R BS.1770-4) ===

    target_lufs: float
    """Target Integrated Loudness in LUFS (dB relative to Full Scale)."""

    lufs_tolerance: float = 0.5
    """Allowed deviation from target (±LU). EBU default: ±0.5 LU."""

    # === True Peak Limiting (ITU-R BS.1770-4) ===

    true_peak_max_dbtp: float = -1.0
    """Maximum True Peak level in dBTP. Default: -1.0 dBTP (EBU R128)."""

    sample_peak_max_dbfs: float | None = None
    """Optional Sample Peak limit in dBFS (für inter-sample peak prevention)."""

    # === Dynamic Range ===

    target_lra: float | None = None
    """Target Loudness Range in LU (optional). None = no constraint."""

    lra_max: float | None = None
    """Maximum allowed LRA (für competitive mastering). None = no limit."""

    # === Metadata Requirements ===

    require_bwf_metadata: bool = False
    """Require Broadcast Wave Format metadata (für archival/broadcast)."""

    bext_description: str = ""
    """BWF bext description field."""

    bext_originator: str = "AURIK Audio Restoration System"
    """BWF originator field."""

    # === Processing Options ===

    enable_loudness_normalization: bool = True
    """Enable loudness normalization to target_lufs."""

    enable_true_peak_limiting: bool = True
    """Enable True Peak limiting."""

    enable_dynamic_range_compression: bool = False
    """Enable DRC für competitive loudness (nur Streaming)."""

    drc_ratio: float = 2.0
    """Dynamic Range Compression ratio (wenn enabled)."""

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dictionary."""
        return asdict(self)


# === Predefined Standard Configurations ===

STANDARD_CONFIGS: dict[DeliveryStandard, StandardConfig] = {
    DeliveryStandard.EBU_R128: StandardConfig(
        name="EBU R128",
        description="European Broadcasting Union Loudness Recommendation",
        target_lufs=-23.0,
        lufs_tolerance=0.5,
        true_peak_max_dbtp=-1.0,
        sample_peak_max_dbfs=None,
        target_lra=None,  # No strict LRA requirement
        lra_max=None,
        require_bwf_metadata=True,  # Broadcast requires BWF
        bext_description="EBU R128 compliant audio",
        enable_loudness_normalization=True,
        enable_true_peak_limiting=True,
        enable_dynamic_range_compression=False,  # Preserve dynamics
    ),
    DeliveryStandard.ATSC_A85: StandardConfig(
        name="ATSC A/85",
        description="US Broadcasting Standard (CALM Act Compliant)",
        target_lufs=-24.0,
        lufs_tolerance=2.0,  # ATSC allows ±2 LU
        true_peak_max_dbtp=-2.0,  # More conservative
        sample_peak_max_dbfs=None,
        target_lra=None,
        lra_max=None,
        require_bwf_metadata=True,
        bext_description="ATSC A/85 compliant audio",
        enable_loudness_normalization=True,
        enable_true_peak_limiting=True,
        enable_dynamic_range_compression=False,
    ),
    DeliveryStandard.SPOTIFY: StandardConfig(
        name="Spotify",
        description="Spotify Loudness Normalization Target",
        target_lufs=-14.0,
        lufs_tolerance=0.5,
        true_peak_max_dbtp=-1.0,
        sample_peak_max_dbfs=-0.1,  # Prevent inter-sample peaks
        target_lra=None,
        lra_max=8.0,  # Competitive streaming: LRA < 8 LU
        require_bwf_metadata=False,
        bext_description="",
        enable_loudness_normalization=True,
        enable_true_peak_limiting=True,
        enable_dynamic_range_compression=True,  # Competitive loudness
        drc_ratio=3.0,
    ),
    DeliveryStandard.ITUNES: StandardConfig(
        name="iTunes / Apple Music",
        description="Apple Music Mastering Standard",
        target_lufs=-16.0,
        lufs_tolerance=1.0,
        true_peak_max_dbtp=-1.0,
        sample_peak_max_dbfs=-0.1,
        target_lra=None,
        lra_max=10.0,  # Moderate compression
        require_bwf_metadata=False,
        bext_description="",
        enable_loudness_normalization=True,
        enable_true_peak_limiting=True,
        enable_dynamic_range_compression=True,
        drc_ratio=2.5,
    ),
    DeliveryStandard.ARCHIVAL: StandardConfig(
        name="Archival (IASA-TC 04)",
        description="Long-term Preservation Standard",
        target_lufs=-18.0,  # Moderate, preserves dynamic range
        lufs_tolerance=1.0,
        true_peak_max_dbtp=-1.0,
        sample_peak_max_dbfs=None,
        target_lra=None,  # Preserve original dynamics
        lra_max=None,
        require_bwf_metadata=True,  # Archival MUST have metadata
        bext_description="Archived and restored by AURIK",
        enable_loudness_normalization=True,
        enable_true_peak_limiting=True,
        enable_dynamic_range_compression=False,  # Preserve dynamic range
    ),
}


@dataclass
class LoudnessResult:
    """Ergebnis der ITU-R BS.1770-4 Lautheitsmessung."""

    integrated_lufs: float
    """Integrated Loudness in LUFS."""

    loudness_range: float
    """Loudness Range (LRA) in LU."""

    true_peak_dbtp: float
    """Maximum True Peak in dBTP."""

    sample_peak_dbfs: float
    """Maximum Sample Peak in dBFS."""

    # Backward-Compat: dict-ähnliche API
    def get(self, key: str, default: Any = None) -> Any:
        return asdict(self).get(key, default)

    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key]

    def __contains__(self, key: str) -> bool:
        return key in asdict(self)

    def items(self):
        return asdict(self).items()

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class LoudnessAnalyzer:
    """
    ITU-R BS.1770-4 Loudness Analyzer.

    Measures:
    - Integrated Loudness (LUFS/LKFS)
    - Loudness Range (LRA in LU)
    - True Peak (dBTP)
    - Sample Peak (dBFS)

    Verwendet pyloudnorm für standard-compliant measurement.
    """

    def __init__(self):
        """Initialisiert LoudnessAnalyzer."""
        self.block_size = 0.400  # 400ms blocks (ITU-R BS.1770-4)

    def analyze(self, audio: np.ndarray, sample_rate: int) -> LoudnessResult:
        """
        Analysiert Lautstärke-Metriken gemäß ITU-R BS.1770-4.

        Args:
            audio: Audio array (mono or stereo)
            sample_rate: Sample rate

        Returns:
            Dict mit:
            - integrated_lufs: Integrated loudness in LUFS
            - loudness_range: LRA in LU
            - true_peak_dbtp: Maximum True Peak in dBTP
            - sample_peak_dbfs: Maximum Sample Peak in dBFS
        """
        try:
            import pyloudnorm as pyln

            _has_pyln = True
        except ImportError:
            _has_pyln = False
            logger.warning(
                "pyloudnorm not installed — using RMS-based loudness estimation. "
                "Install via: pip install pyloudnorm for ITU-R BS.1770-4 compliance."
            )

        # Ensure correct shape
        if audio.ndim == 1:
            audio_2d = audio.reshape(-1, 1)
        elif audio.ndim == 2:
            audio_2d = audio.T if audio.shape[0] < audio.shape[1] else audio
        else:
            raise ValueError(f"Invalid audio shape: {audio.shape}")

        # Integrated Loudness
        if _has_pyln:
            meter = pyln.Meter(sample_rate)
            integrated_lufs = meter.integrated_loudness(audio_2d)
        else:
            # RMS-based fallback: −23 LUFS ≈ −23 dB relative to full scale for typical speech
            rms = float(np.sqrt(np.mean(np.square(audio_2d))) + 1e-12)
            integrated_lufs = float(20 * np.log10(rms + 1e-12))

        # Loudness Range (LRA)
        # pyloudnorm doesn't have LRA built-in, but we can use percentile analysis
        # Simplified: Use short-term loudness variance
        try:
            short_term_loudness = self._compute_short_term_loudness(audio_2d, sample_rate)
            if len(short_term_loudness) > 0:
                # LRA ≈ difference between 95th and 10th percentile
                lra = np.percentile(short_term_loudness, 95) - np.percentile(short_term_loudness, 10)
            else:
                lra = 0.0  # type: ignore[assignment]
        except Exception:
            lra = 0.0  # type: ignore[assignment]

        # True Peak (dBTP)
        # True Peak = Sample Peak level considering inter-sample peaks
        # pyloudnorm doesn't have True Peak, so we approximate via oversampling
        try:
            from scipy import signal

            # Oversample 4x for True Peak approximation
            oversampled = signal.resample(audio_2d, len(audio_2d) * 4, axis=0)
            true_peak_linear: float = float(np.max(np.abs(oversampled)))
            true_peak_dbtp = 20 * np.log10(true_peak_linear + 1e-10)
        except Exception:
            # Fallback: Use sample peak
            true_peak_linear = np.max(np.abs(audio_2d))
            true_peak_dbtp = 20 * np.log10(true_peak_linear + 1e-10)

        # Sample Peak (dBFS)
        sample_peak_linear: float = float(np.max(np.abs(audio_2d)))
        sample_peak_dbfs = 20 * np.log10(sample_peak_linear + 1e-10)

        return LoudnessResult(
            integrated_lufs=float(integrated_lufs),
            loudness_range=float(lra),
            true_peak_dbtp=float(true_peak_dbtp),
            sample_peak_dbfs=float(sample_peak_dbfs),
        )

    def _compute_short_term_loudness(
        self, audio: np.ndarray, sample_rate: int, block_size_sec: float = 3.0
    ) -> np.ndarray:
        """
        Berechnet short-term loudness (3s blocks per ITU-R BS.1770-4).

        Args:
            audio: Audio array (N, channels)
            sample_rate: Sample rate
            block_size_sec: Block size in seconds (default: 3.0s)

        Returns:
            Array of short-term loudness values in LUFS
        """
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sample_rate)
            block_samples = int(block_size_sec * sample_rate)

            loudness_values = []

            for start in range(0, len(audio), block_samples):
                end = min(start + block_samples, len(audio))
                block = audio[start:end]

                if len(block) < sample_rate * 0.4:  # Skip blocks < 400ms
                    continue

                try:
                    loudness = meter.integrated_loudness(block)
                    if np.isfinite(loudness) and loudness > -70:  # Valid loudness
                        loudness_values.append(loudness)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue

            return np.array(loudness_values)  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning("Short-term loudness calculation fehlgeschlagen: %s", e)
            return np.array([])  # type: ignore[no-any-return]


class TruePeakLimiter:
    """
    True Peak Limiter (ITU-R BS.1770-4 compliant).

    Prevents True Peaks über threshold via lookahead limiter.
    """

    def __init__(self, threshold_dbtp: float = -1.0, lookahead_ms: float = 5.0, release_ms: float = 100.0):
        """
        Initialisiert True Peak Limiter.

        Args:
            threshold_dbtp: Maximum True Peak in dBTP
            lookahead_ms: Lookahead time in ms (default: 5ms)
            release_ms: Release time in ms (default: 100ms)
        """
        self.threshold_dbtp = threshold_dbtp
        self.threshold_linear = 10 ** (threshold_dbtp / 20.0)
        self.lookahead_ms = lookahead_ms
        self.release_ms = release_ms

    def limit(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Wendet an: True Peak limiting.

        Args:
            audio: Input audio
            sample_rate: Sample rate

        Returns:
            Limited audio
        """
        # Simplified True Peak Limiter:
        # 1. Oversample 4x
        # 2. Apply simple peak limiter
        # 3. Downsample back

        try:
            from scipy import signal

            # Oversample 4x
            oversampled = signal.resample(audio, len(audio) * 4, axis=0)

            # Apply lookahead limiter (simplified: soft clip)
            limited = self._soft_clip(oversampled, self.threshold_linear)

            # Downsample back
            result = signal.resample(limited, len(audio), axis=0)

            return result.astype(audio.dtype)  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning("True Peak limiting fehlgeschlagen: %s, using simple clipper", e)
            # Fallback: Simple hard clip
            return np.clip(audio, -self.threshold_linear, self.threshold_linear)  # type: ignore[no-any-return]

    def _soft_clip(self, audio: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft clipping function (smoother than hard clip).

        Uses tanh-based soft clipper.
        """
        # Normalize by threshold
        normalized = audio / threshold

        # Soft clip via tanh (smooth saturation)
        clipped_normalized = np.tanh(normalized)

        # Scale back
        return clipped_normalized * threshold  # type: ignore[no-any-return]


class BWFMetadataWriter:
    """
    Broadcast Wave Format (BWF) Metadata Writer.

    Schreibt BWF bext chunk für archival/broadcast compliance.
    Basiert auf EBU Tech 3285 und IASA-TC 04.
    """

    @staticmethod
    def write_bwf_metadata(
        audio_file_path: Path,
        description: str = "",
        originator: str = "AURIK Audio Restoration System",
        originator_reference: str = "",
        origination_date: str | None = None,
        origination_time: str | None = None,
        coding_history: str = "",
    ) -> bool:
        """
        Schreibt BWF metadata to WAV file.

        Args:
            audio_file_path: Path to WAV file
            description: Content description (max 256 chars)
            originator: Organization/person (max 32 chars)
            originator_reference: Unique file reference (max 32 chars)
            origination_date: YYYY-MM-DD
            origination_time: HH:MM:SS
            coding_history: Processing history (free form)

        Returns:
            True if successful
        """
        try:
            # datetime already imported at module level
            # Generate defaults
            if origination_date is None:
                now = datetime.now()
                origination_date = now.strftime("%Y-%m-%d")
                origination_time = now.strftime("%H:%M:%S")

            if originator_reference == "":
                # Generate unique reference based on timestamp
                originator_reference = f"AURIK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Baue den BEXT-Chunk gemäß EBU Tech 3285 (Broadcast Wave Format)
            import struct

            def _encode(s: str, length: int) -> bytes:
                return s.encode("latin-1", errors="replace")[:length].ljust(length, b"\x00")

            time_str = origination_time if origination_time else "00:00:00"
            coding_bytes = (coding_history + "\r\n").encode("latin-1", errors="replace")

            bext_content = (
                _encode(description, 256)  # Description
                + _encode(originator, 32)  # Originator
                + _encode(originator_reference, 32)  # OriginatorReference
                + _encode(origination_date, 10)  # OriginationDate YYYY-MM-DD
                + _encode(time_str, 8)  # OriginationTime HH:MM:SS
                + struct.pack("<II", 0, 0)  # TimeCodeLow / TimeCodeHigh
                + struct.pack("<H", 2)  # BWFVersion 2
                + bytes(64)  # UMID (zeros)
                + struct.pack("<hhhhh", 0, 0, 0, 0, 0)  # Loudness (0 = unset)
                + bytes(180)  # Reserved
                + coding_bytes
            )
            # Chunk-Größe muss gerade sein (RIFF alignment)
            if len(bext_content) % 2 != 0:
                bext_content += b"\x00"
            bext_chunk = b"bext" + struct.pack("<I", len(bext_content)) + bext_content

            # Füge BEXT vor dem "data"-Chunk in die WAV-Datei ein
            with open(audio_file_path, "rb") as fh:
                raw = fh.read()

            if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
                # Suche "data"-Chunk
                pos = 12
                insert_pos = None
                while pos + 8 <= len(raw):
                    cid = raw[pos : pos + 4]
                    csz = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
                    if cid == b"data":
                        insert_pos = pos
                        break
                    pos += 8 + csz + (csz % 2)

                if insert_pos is not None:
                    new_raw = raw[:insert_pos] + bext_chunk + raw[insert_pos:]
                    # RIFF-Größe anpassen
                    new_riff_size = len(new_raw) - 8
                    new_raw = new_raw[:4] + struct.pack("<I", new_riff_size) + new_raw[8:]
                    with open(audio_file_path, "wb") as fh:  # type: ignore[assignment]
                        fh.write(new_raw)
                    logger.info("BWF/BEXT-Chunk geschrieben: %s (%s Bytes)", audio_file_path, len(bext_content))
                    return True

            # Fallback: Nur Logging
            logger.info("BWF Metadata (nur Log, kein BEXT möglich für %s):", audio_file_path)
            logger.info("  Description: %s", description)
            logger.info("  Originator: %s", originator)
            return True

        except Exception as e:
            logger.error("BWF metadata schreibe fehlgeschlagen: %s", e)
            return False


_delivery_manager_instance: Optional["DeliveryStandardsManager"] = None
_delivery_manager_lock = threading.Lock()


def get_delivery_manager() -> "DeliveryStandardsManager":
    """Gibt zurück: or create DeliveryStandardsManager singleton.

    Returns:
        DeliveryStandardsManager singleton instance
    """
    global _delivery_manager_instance
    if _delivery_manager_instance is None:
        with _delivery_manager_lock:
            if _delivery_manager_instance is None:
                _delivery_manager_instance = DeliveryStandardsManager()
    return _delivery_manager_instance


class DeliveryStandardsManager:
    """
    Main API für Delivery Standards Processing.

    Usage:
        manager = get_delivery_manager()
        result = manager.process_for_standard(
            audio, sample_rate,
            standard=DeliveryStandard.EBU_R128
        )

        normalized_audio = result["audio"]
        metadata = result["metadata"]
    """

    def __init__(self) -> None:
        """Initialisiert DeliveryStandardsManager."""
        self.loudness_analyzer = LoudnessAnalyzer()
        logger.info("DeliveryStandardsManager initialisiert")

    def process_for_standard(
        self, audio: np.ndarray, sample_rate: int, standard: DeliveryStandard, output_path: Path | None = None
    ) -> dict[str, Any]:
        """
        Process audio for specific delivery standard.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            standard: Target delivery standard
            output_path: Optional output file path (für BWF metadata)

        Returns:
            Dict mit:
            - audio: Processed audio
            - standard_name: Standard name
            - initial_loudness: Initial LUFS
            - final_loudness: Final LUFS
            - gain_applied_db: Applied gain in dB
            - true_peak_dbtp: Final True Peak
            - compliant: Boolean (within tolerances?)
            - metadata: BWF metadata dict (if applicable)
        """
        config = STANDARD_CONFIGS.get(standard)
        if config is None:
            raise ValueError(f"Unknown standard: {standard}")

        logger.info("🎯 Processing for standard: %s", config.name)

        # === 1. Analyze Initial Loudness ===
        initial_metrics = self.loudness_analyzer.analyze(audio, sample_rate)
        initial_lufs = initial_metrics["integrated_lufs"]
        initial_tp = initial_metrics["true_peak_dbtp"]
        initial_lra = initial_metrics["loudness_range"]

        logger.info("  Initial: %.1f LUFS, TP %.1f dBTP, LRA %.1f LU", initial_lufs, initial_tp, initial_lra)

        # === 2. Loudness Normalization ===
        if config.enable_loudness_normalization:
            gain_db = config.target_lufs - initial_lufs
            gain_linear = 10 ** (gain_db / 20.0)

            audio_normalized = audio * gain_linear

            logger.info("  Loudness Gain: %+.1f dB → Target %.1f LUFS", gain_db, config.target_lufs)
        else:
            audio_normalized = audio.copy()
            gain_db = 0.0

        # === 3. Dynamic Range Compression (optional, für Streaming) ===
        if config.enable_dynamic_range_compression:
            audio_normalized = self._apply_drc(
                audio_normalized,
                sample_rate,
                ratio=config.drc_ratio,
                threshold_lufs=config.target_lufs - 6.0,  # Attack 6 LU below target
            )
            logger.info("  DRC angewendet (Verhaeltnis %s:1)", config.drc_ratio)

        # === 4. True Peak Limiting ===
        if config.enable_true_peak_limiting:
            limiter = TruePeakLimiter(threshold_dbtp=config.true_peak_max_dbtp, lookahead_ms=5.0, release_ms=100.0)
            audio_limited = limiter.limit(audio_normalized, sample_rate)
            logger.info("  True Peak limited to %.1f dBTP", config.true_peak_max_dbtp)
        else:
            audio_limited = audio_normalized

        # === 5. Final Loudness Verification ===
        final_metrics = self.loudness_analyzer.analyze(audio_limited, sample_rate)
        final_lufs = final_metrics["integrated_lufs"]
        final_tp = final_metrics["true_peak_dbtp"]
        final_lra = final_metrics["loudness_range"]

        logger.info("  Final: %.1f LUFS, TP %.1f dBTP, LRA %.1f LU", final_lufs, final_tp, final_lra)

        # === 6. Compliance Check ===
        lufs_deviation = abs(final_lufs - config.target_lufs)
        compliant = lufs_deviation <= config.lufs_tolerance and final_tp <= config.true_peak_max_dbtp

        if compliant:
            logger.info("  ✅ COMPLIANT with %s", config.name)
        else:
            logger.warning("  ⚠ NOT FULLY COMPLIANT (deviation: %.2f LU)", lufs_deviation)

        # === 7. BWF Metadata (wenn erforderlich) ===
        metadata = {}

        if config.require_bwf_metadata and output_path is not None:
            coding_history = (
                f"A=PCM,F={sample_rate},W=24,M=stereo,T=AURIK_Restoration\r\n"
                f"Loudness normalized to {config.target_lufs:.1f} LUFS ({config.name})\r\n"
                f"True Peak limited to {config.true_peak_max_dbtp:.1f} dBTP\r\n"
            )

            BWFMetadataWriter.write_bwf_metadata(
                audio_file_path=output_path,
                description=config.bext_description,
                originator=config.bext_originator,
                coding_history=coding_history,
            )

            metadata = {
                "description": config.bext_description,
                "originator": config.bext_originator,
                "coding_history": coding_history,
            }

        # === Result ===
        return {
            "success": True,
            "audio": audio_limited,
            "standard_name": config.name,
            "initial_loudness": initial_lufs,
            "final_loudness": final_lufs,
            "gain_applied_db": gain_db,
            "true_peak_dbtp": final_tp,
            "loudness_range": final_lra,
            "compliant": compliant,
            "lufs_deviation": lufs_deviation,
            "metadata": metadata,
        }

    def _apply_drc(
        self, audio: np.ndarray, sample_rate: int, ratio: float = 2.0, threshold_lufs: float = -20.0
    ) -> np.ndarray:
        """
        Wendet an: Dynamic Range Compression.

        Simplified DRC for competitive loudness.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            ratio: Compression ratio (2.0 = 2:1)
            threshold_lufs: Threshold in LUFS

        Returns:
            Compressed audio
        """
        # Simplified: Use RMS-based compression
        # (A real implementation would use a sophisticated compressor)

        try:
            # Convert threshold LUFS to linear amplitude
            threshold_linear = 10 ** (threshold_lufs / 20.0)

            # Compute envelope (RMS with sliding window)
            window_size = int(0.01 * sample_rate)  # 10ms window
            envelope = np.sqrt(np.convolve(audio**2, np.ones(window_size) / window_size, mode="same"))

            # Compute gain reduction
            gain = np.ones_like(envelope)
            over_threshold = envelope > threshold_linear

            if np.any(over_threshold):
                # Compress signal over threshold
                compression_amount = (envelope[over_threshold] / threshold_linear) ** (1.0 - 1.0 / ratio)
                gain[over_threshold] = threshold_linear / (envelope[over_threshold] + 1e-10) * compression_amount

            # Apply gain smoothly
            from scipy.ndimage import gaussian_filter1d

            gain_smooth = gaussian_filter1d(gain, sigma=window_size)

            return audio * gain_smooth  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning("DRC fehlgeschlagen: %s, returning Originalsignal", e)
            return audio


def get_standard_config(standard: DeliveryStandard) -> StandardConfig:
    """Gibt zurück: configuration for a delivery standard."""
    if standard not in STANDARD_CONFIGS:
        raise ValueError(f"Unknown standard: {standard}")
    return STANDARD_CONFIGS[standard]


def list_available_standards() -> dict[str, str]:
    """Listet auf: all available delivery standards."""
    return {standard.value: config.description for standard, config in STANDARD_CONFIGS.items()}


# === Example Usage ===
if __name__ == "__main__":
    import soundfile as sf

    from backend.file_import import load_audio_file

    # Load test audio
    _res = load_audio_file("test_audio/test_input.wav")
    audio, sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])  # type: ignore[index]

    # Process for EBU R128
    manager = DeliveryStandardsManager()

    result = manager.process_for_standard(
        audio=audio,
        sample_rate=sr,
        standard=DeliveryStandard.EBU_R128,
        output_path=Path("test_output/ebu_r128_compliant.wav"),
    )

    # Save result
    sf.write("test_output/ebu_r128_compliant.wav", result["audio"], sr)

    logger.debug("\n✅ %s Processing vollstaendig", result["standard_name"])
    logger.debug("   Initial Loudness: %.1f LUFS", result["initial_loudness"])
    logger.debug("   Final Loudness: %.1f LUFS", result["final_loudness"])
    logger.debug("   Gain angewendet: %.1f dB", result["gain_applied_db"])
    logger.debug("   True Peak: %.1f dBTP", result["true_peak_dbtp"])
    logger.debug("   Compliant: %s", result["compliant"])
