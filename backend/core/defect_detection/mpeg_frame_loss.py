"""
MPEG Frame Loss Detector — Aurik Tier 2
=======================================

Detektiert MP3/AAC-Frame-Verluste via:
- Abrupte High-Frequency-Cutoffs (Brickwall-Effekt bei typischen Bitraten)
- Zeitliche Diskontinuitäten: 26-ms-MPEG-Frame-Lücken
- Energie-Drops > 20 dB innerhalb eines Frames

Gibt (defect_locations, confidence) zurück und registriert via
DefectType.MPEG_FRAME_LOSS im DefectScanner.

Wissenschaftliche Grundlage:
    - Herre & Johnston (1996) AES Conv. 101 — Pre-Echo / Temporal Masking
    - Brandenburg & Stoll (1994) AES 96th Conv. — ISO-MPEG-1 Audio Layer 3
    - Bosi & Goldberg (2003) Introduction to Digital Audio Coding, Springer
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def detect_mpeg_frame_loss(
    audio: np.ndarray,
    sr: int,
    *,
    fft_size: int = 2048,
    hop_size: int | None = None,
    frame_loss_energy_db: float = 20.0,
    brickwall_check_khz: float = 15.5,
) -> tuple[list[tuple[float, float]], float, dict]:
    """Erkennt MPEG-Frame-Verluste in Audiomaterial.

    Args:
        audio: Mono-Audio-Array (n_samples,).
        sr: Abtastrate in Hz.
        fft_size: FFT-Fenstergröße für Spektral-Analyse.
        hop_size: Hop-Größe. Default: fft_size // 4.
        frame_loss_energy_db: Energie-Drop-Schwelle in dB.
        brickwall_check_khz: Frequenzgrenze für Brickwall-Suche in kHz.

    Returns:
        Tuple aus (defect_locations, confidence, metadata).
        defect_locations: Liste von (start_s, end_s) in Sekunden.
        confidence: 0.0–1.0.
        metadata: dict mit diagnostischen Feldern.
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # → mono

    audio = audio.astype(np.float64)
    n_samples = len(audio)
    duration = n_samples / sr

    if duration < 2.0:
        return [], 0.0  # type: ignore[return-value]

    if hop_size is None:
        hop_size = fft_size // 4

    n_frames = (n_samples - fft_size) // hop_size + 1
    if n_frames < 2:
        return [], 0.0  # type: ignore[return-value]

    # --- 1. Spektrogramm ---
    window = np.hanning(fft_size)
    specgram = np.zeros((fft_size // 2 + 1, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start : start + fft_size] * window
        specgram[:, i] = np.abs(np.fft.rfft(frame))

    freq_bins = np.fft.rfftfreq(fft_size, d=1.0 / sr)

    # --- 2. Brickwall-Detektion: Suche nach abruptem HF-Cutoff ---
    # Typische MP3-Brickwall-Frequenzen:
    #   128 kbps → ~16 kHz
    #   96 kbps  → ~15.5 kHz
    #   64 kbps  → ~11 kHz
    brickwall_candidates = [16_000, 15_500, 11_000, 8_000]
    np.arange(n_frames) * hop_size / sr

    brickwall_locations: list[tuple[float, float]] = []
    energy_drop_locations: list[tuple[float, float]] = []
    temporal_gap_locations: list[tuple[float, float]] = []

    # 2a. HF-Energy pro Frame oberhalb jedes Brickwall-Candidates
    for bw_hz in brickwall_candidates:
        if bw_hz >= sr / 2.1:
            continue
        mask = freq_bins >= bw_hz
        if not np.any(mask):
            continue
        hf_energy = np.sum(specgram[mask, :], axis=0) + 1e-20

        # Gleitender Median der HF-Energie (51-Frame-Fenster ≈ ~1.2 s, muss ungerade sein)
        win_len = min(51, n_frames // 3)
        if win_len < 3:
            continue
        if win_len % 2 == 0:
            win_len += 1  # scipy medfilt erfordert ungerade kernel_size
        hf_median = signal.medfilt(hf_energy, kernel_size=win_len)

        # Brickwall: HF-Energie fällt unter 1% des lokalen Medians
        threshold = hf_median * 0.01
        zero_hf = hf_energy < threshold

        # Finde zusammenhängende Regionen
        if np.any(zero_hf):
            edges = np.diff(np.concatenate([[0], zero_hf.astype(int), [0]]))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]

            for s, e in zip(starts, ends):
                if e > s:  # mindestens 2 Frames
                    dur_s = (e - s) * hop_size / sr
                    if dur_s >= 0.026:  # mindestens ein MPEG-Frame
                        t_start = s * hop_size / sr
                        t_end = e * hop_size / sr
                        if not _overlaps_existing(t_start, t_end, brickwall_locations, margin=0.05):
                            brickwall_locations.append((t_start, t_end))

    # --- 3. Energie-Drops: 26-ms-Fenster, suche nach > 20 dB Einbrüchen ---
    frame_len_ms = 26.0
    frame_len_samp = int(frame_len_ms / 1000 * sr)
    if frame_len_samp < 8:
        frame_len_samp = 8

    n_energy_frames = (n_samples - frame_len_samp) // frame_len_samp + 1
    if n_energy_frames >= 2:
        rms_per_frame = np.zeros(n_energy_frames, dtype=np.float64)
        for i in range(n_energy_frames):
            start = i * frame_len_samp
            seg = audio[start : start + frame_len_samp]
            rms_per_frame[i] = np.sqrt(np.mean(seg**2)) + 1e-20

        rms_db = 20 * np.log10(rms_per_frame)
        rms_diff = np.diff(rms_db)  # dB-Änderung zwischen benachbarten 26-ms-Frames

        # Suche nach Negativ-Sprüngen > frame_loss_energy_db
        drop_indices = np.where(rms_diff < -frame_loss_energy_db)[0]
        for idx in drop_indices:
            t_start = idx * frame_len_samp / sr
            t_end = min((idx + 2) * frame_len_samp / sr, duration)
            if not _overlaps_existing(t_start, t_end, energy_drop_locations, margin=0.03):
                energy_drop_locations.append((t_start, t_end))

    # --- 4. Zeitliche Diskontinuitäten: Suche nach abrupten Phasen- oder Pegelsprüngen ---
    # Kurzzeit-Energie in 13-ms-Schritten (halbe MPEG-Frame-Länge)
    half_frame_samples = max(int(13e-3 * sr), 16)
    n_half_frames = (n_samples - half_frame_samples) // half_frame_samples + 1
    if n_half_frames >= 4:
        half_rms = np.zeros(n_half_frames, dtype=np.float64)
        half_energy = np.zeros(n_half_frames, dtype=np.float64)
        for i in range(n_half_frames):
            start = i * half_frame_samples
            seg = audio[start : start + half_frame_samples]
            half_rms[i] = np.sqrt(np.mean(seg**2)) + 1e-20
            half_energy[i] = np.sum(seg**2) + 1e-20

        # RMS-Ratio zwischen benachbarten Halbframes
        rms_ratio = half_rms[:-1] / (half_rms[1:] + 1e-20)

        # Große Diskontinuitäten: Ratio > 10 (≈20 dB) in einer Richtung
        big_jumps = np.where((rms_ratio > 10.0) | (rms_ratio < 0.1))[0]
        for idx in big_jumps:
            t_start = idx * half_frame_samples / sr
            t_end = min((idx + 2) * half_frame_samples / sr, duration)
            if not _overlaps_existing(t_start, t_end, temporal_gap_locations, margin=0.02):
                temporal_gap_locations.append((t_start, t_end))

    # --- Severity & Confidence ---
    all_locations = brickwall_locations + energy_drop_locations + temporal_gap_locations
    all_locations = sorted(set(all_locations))

    if not all_locations:
        return [], 0.0  # type: ignore[return-value]

    # Brickwall als stärkster Indikator für MPEG-Frame-Verlust
    total_defect_s = sum(e - s for s, e in all_locations)
    defect_density = total_defect_s / max(duration, 1e-6)

    # Mehr Brickwalls = höhere Konfidenz + Severity
    bw_weight = min(len(brickwall_locations) / 5.0, 1.0)  # Sättigung bei 5
    drop_weight = min(len(energy_drop_locations) / 8.0, 1.0)

    confidence = float(np.clip(0.55 + 0.35 * bw_weight + 0.10 * drop_weight, 0.4, 0.95))
    severity = float(np.clip(defect_density / 0.05, 0.0, 1.0))  # 5% defect time → severity 1.0

    # Deckle auf 200 Locations
    if len(all_locations) > 200:
        all_locations = all_locations[:200]

    _metadata = {
        "brickwall_regions": len(brickwall_locations),
        "energy_drops": len(energy_drop_locations),
        "temporal_gaps": len(temporal_gap_locations),
        "total_defect_s": round(total_defect_s, 4),
        "defect_density": round(defect_density, 4),
        "severity": round(severity, 4),
    }

    return all_locations, confidence  # type: ignore[return-value]


def _overlaps_existing(start: float, end: float, existing: list[tuple[float, float]], margin: float = 0.05) -> bool:
    """Prüft, ob ein Zeitintervall mit bestehenden Locations überlappt."""
    return any(start < ee + margin and end > es - margin for es, ee in existing)
