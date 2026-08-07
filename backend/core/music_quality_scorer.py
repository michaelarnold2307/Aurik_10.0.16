"""
core/music_quality_scorer.py — Music-spezifischer Qualitäts-Scorer
===================================================================

DNSMOS (Deep Noise Suppression MOS) wurde für Sprache trainiert und
unterschätzt musikalische Qualität systematisch. Dieser Modul ersetzt
DNSMOS für Musik durch ein DSP-basiertes Music-MOS-Äquivalent, das auf
Eigenschaften archivalischer Musik kalibriert ist.

MUSIC-MOS DIMENSIONEN (analog DNSMOS P.835):
  - MUSIC_SIG  (Signal Quality, 1–5): Wie klar ist das Musiksignal?
  - MUSIC_BAK  (Background Quality, 1–5): Wie sauber ist der Hintergrund?
  - MUSIC_OVR  (Overall Quality, 1–5): Gesamtbewertung
  - MUSIC_NAT  (Naturalness, 1–5): Klingt es nach echter Musik?

ALGORITHMUS (reine DSP-Merkmale, kein Neuronales Netz nötig):
  MUSIC_SIG: Harmonizität × Tonalität × Dynamikerhaltung
  MUSIC_BAK:  1 - f(Rauschpegel, Hum-Energie, Klick-Dichte)
  MUSIC_OVR:  Gewichtetes Mittel aller Dimensionen
  MUSIC_NAT:  Spectral Centroid-Stabilität × Envelope-Smoothness × Artikulation

KALIBRIERUNG gegen archivalische Referenzen:
  - Score 5.0 = restauriertes LP-Original (Referenz-Digitalisierung)
  - Score 4.0 = typisches iZotope-RX-Ergebnis
  - Score 3.0 = Standard-FFT-Denoise ohne Musikkalibrierung
  - Score 2.0 = überprozessiertes, artefaktreiches Signal
  - Score 1.0 = stark beschädigtes Originalmaterial

Plugin-Erweiterungspunkt:
  Wenn `plugins/music_mos_plugin.py` vorhanden und `score()` implementiert,
  wird das als Primärpfad genutzt (z.B. MERT-Embedding-basierter Scorer).

Author: Aurik Development Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ─── Konstanten ──────────────────────────────────────────────────────────
_TARGET_SR = 16000  # Interne Analyse-Samplerate
_FRAME_SIZE = 1024
_HOP_SIZE = 256
_MIN_SIGNAL_RMS = 1e-6  # Untergrenze für sinnvolle Analyse


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────


def _resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Einfaches Decimation-basiertes Resampling auf 16 kHz (kein scipy-Dep.)."""
    if sample_rate == _TARGET_SR:
        return audio
    mono = audio.flatten() if audio.ndim > 1 else audio
    ratio = sample_rate / _TARGET_SR
    target_len = max(1, int(len(mono) / ratio))
    indices = np.linspace(0, len(mono) - 1, target_len)
    resampled = np.interp(indices, np.arange(len(mono)), mono).astype(np.float32)
    # NaN/Inf-Guard (§3.1)
    resampled = np.nan_to_num(resampled, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(resampled, -1.0, 1.0)  # type: ignore[no-any-return]


def _frame_audio(audio: np.ndarray) -> np.ndarray:
    """Zerlegt Audio in überlappende Frames (shape: [n_frames, frame_size])."""
    n_frames = max(1, (len(audio) - _FRAME_SIZE) // _HOP_SIZE + 1)
    frames = np.array(
        [
            audio[i * _HOP_SIZE : i * _HOP_SIZE + _FRAME_SIZE]
            for i in range(n_frames)
            if i * _HOP_SIZE + _FRAME_SIZE <= len(audio)
        ]
    )
    return frames if len(frames) > 0 else audio[:_FRAME_SIZE][None, :]


def _harmonicity(frames: np.ndarray) -> float:
    """
    Harmonizität = mittleres Verhältnis Harmonische/Gesamtenergie.
    Hoher Wert = tonales Signal (Musik), niedriger = Rauschen/Sprache.
    """
    scores = []
    for frame in frames[:100]:
        window = frame * np.hanning(len(frame))
        mag = np.abs(np.fft.rfft(window))
        if mag.max() < 1e-10:
            continue
        # Fundament-Kandidat: stärkste Komponente unter 2000 Hz
        freqs = np.fft.rfftfreq(len(frame), 1 / _TARGET_SR)
        low_mask = freqs < 2000
        if not np.any(low_mask):
            continue
        f0_idx = np.argmax(mag * low_mask)
        if f0_idx == 0:
            continue
        # Harmonische (2f0, 3f0, …, 8f0) Energie
        harmonic_energy = 0.0
        for k in range(2, 9):
            h_idx = f0_idx * k
            if h_idx < len(mag):
                w = max(1, f0_idx // 4)
                harmonic_energy += float(np.sum(mag[max(0, h_idx - w) : h_idx + w] ** 2))  # type: ignore
        total_energy = np.sum(mag**2) + 1e-10
        scores.append(harmonic_energy / total_energy)

    result = float(np.mean(scores)) if scores else 0.3
    # §v10.93 NaN-Guard: np.mean([]) → nan bereits durch else 0.3 abgefangen.
    # nan_to_num hier defensiv gegen hypothetisches NaN in scores (kein bekannter Pfad).
    return float(np.clip(np.nan_to_num(result, nan=0.3, posinf=1.0, neginf=0.0), 0.0, 1.0))


def _noise_floor_db(audio: np.ndarray) -> float:
    """Schätzt den Rauschpegel in dBFS (Minimum-Statistics-Näherung)."""
    frames = _frame_audio(audio)
    rms_vals = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
    result = float(20 * np.log10(np.percentile(rms_vals, 10)))
    # NaN/Inf-Guard (§3.1)
    result = np.nan_to_num(result, nan=-60.0, posinf=0.0, neginf=-80.0)
    return float(result)


def _click_density(audio: np.ndarray) -> float:
    """Klick-Dichte: Anteil der Samples, deren Energie extreme Ausreißer sind."""
    if len(audio) < 256:
        return 0.0
    abs_audio = np.abs(audio)
    threshold = np.percentile(abs_audio, 99.5)
    if threshold < 1e-10:
        return 0.0
    n_clicks: int = int(np.sum(abs_audio > threshold * 3))
    result = float(n_clicks / len(audio))
    # NaN/Inf-Guard (§3.1)
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return float(np.clip(result, 0.0, 1.0))


def _hum_energy_ratio(audio: np.ndarray) -> float:
    """Verhältnis der Energie in typischen Hum-Frequenzen (50/60 Hz ± 5 Hz)."""
    if len(audio) < 1024:
        return 0.0
    mag = np.abs(np.fft.rfft(audio[:4096] if len(audio) >= 4096 else audio))
    freqs = np.fft.rfftfreq(min(4096, len(audio)), 1 / _TARGET_SR)
    total_energy = np.sum(mag**2) + 1e-10
    hum_mask = ((freqs > 45) & (freqs < 65)) | ((freqs > 55) & (freqs < 75))  # 50 Hz +/-5  # 60 Hz +/-5
    hum_energy: float = float(np.sum(mag[hum_mask] ** 2))
    result = float(hum_energy / total_energy)
    # NaN/Inf-Guard (§3.1)
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return float(np.clip(result, 0.0, 1.0))


def _envelope_smoothness(audio: np.ndarray) -> float:
    """Glätte der Amplitudenhüllkurve [0, 1] (1=sehr glatt = natürlich)."""
    frames = _frame_audio(audio)
    rms_vals = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
    if len(rms_vals) < 3:
        return 1.0
    diffs = np.diff(np.log(rms_vals))
    roughness = np.std(diffs)
    # Typische Musik: roughness ~ 0.1–0.5; überprozessiert: >1.0
    result = float(max(0.0, 1.0 - min(1.0, roughness / 0.8)))  # type: ignore[operator]
    # NaN/Inf-Guard (§3.1)
    result = np.nan_to_num(result, nan=0.8, posinf=1.0, neginf=0.0)
    return float(np.clip(result, 0.0, 1.0))


def _spectral_centroid_stability(frames: np.ndarray) -> float:
    """Stabilität des Spectral Centroid (stabile Centroide = natürlicher Klang)."""
    centroids = []
    freqs = np.fft.rfftfreq(_FRAME_SIZE, 1 / _TARGET_SR)
    for frame in frames[:80]:
        window = frame * np.hanning(len(frame))
        mag = np.abs(np.fft.rfft(window))
        if mag.sum() < 1e-10:
            continue
        centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-10)
        centroids.append(centroid)
    if len(centroids) < 3:
        return 0.8
    cv = np.std(centroids) / (np.mean(centroids) + 1e-10)
    result = float(max(0.0, 1.0 - min(1.0, cv / 1.5)))  # type: ignore[operator]
    # NaN/Inf-Guard (§3.1)
    result = np.nan_to_num(result, nan=0.8, posinf=1.0, neginf=0.0)
    return float(np.clip(result, 0.0, 1.0))


def _spectral_flux_continuity(frames: np.ndarray) -> float:
    """
    Spectral Flux Continuity: wie gleichmäßig entwickelt sich das Spektrum [0, 1].

    Hoher Wert = glatte Spektralevolution = natürlich klingende Musik.
    Niedriger Wert = abrupte Spektralsprünge = typische Verarbeitungsartefakte.

    Für MUSIC_NAT: Ergänzt _spectral_centroid_stability um die zeitliche
    Kohärenz auf Frame-Level (nicht nur Centroid, sondern alle Bins).
    """
    if len(frames) < 3:
        return 0.8
    # Magnitude-Spektrum pro Frame
    mags = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1])[np.newaxis, :], axis=1))  # shape: (n_frames, n_bins)
    # Normierter Spectral Flux: L1-Abstand aufeinanderfolgender Spektren
    flux = np.mean(np.abs(np.diff(mags, axis=0)), axis=1)  # (n_frames-1,)
    mean_mag = np.mean(np.abs(mags[:-1]), axis=1) + 1e-10
    flux_norm = flux / mean_mag
    mean_flux = float(np.mean(flux_norm))
    # Niedriger mittlerer Flux <=> glatte Evolution <=> score nahe 1.0
    result = float(max(0.0, 1.0 - min(1.0, mean_flux * 4.5)))
    # NaN/Inf-Guard (§3.1)
    result = np.nan_to_num(result, nan=0.7, posinf=1.0, neginf=0.0)
    return float(np.clip(result, 0.0, 1.0))


def _micro_dynamic_variation(audio: np.ndarray) -> float:
    """
    Micro-Dynamic Variation: natürliche Amplitudenmodulation [0, 1].

    Echter Musikklang hat inhärente Mikro-Amplitudenschwankungen (Vibrato,
    Atemtremolo, Bogendruck). Überprozessiertes Audio ist zu uniform — es
    klingt plastisch.

    Optimaler Bereich: Variationskoeffizient CV ~ 0.05–0.20
      - CV < 0.05  → überprozessiert, zu einförmig → score < 1.0
      - CV 0.05–0.20 → natürliche Variation → score = 1.0
      - CV > 0.20  → zu laut/verrauscht → score < 1.0
    """
    frame_size = _FRAME_SIZE
    n_frames = max(1, (len(audio) - frame_size) // _HOP_SIZE + 1)
    frames_here = np.array(
        [
            audio[i * _HOP_SIZE : i * _HOP_SIZE + frame_size]
            for i in range(n_frames)
            if i * _HOP_SIZE + frame_size <= len(audio)
        ]
    )
    if len(frames_here) < 5:
        return 0.7
    rms_vals = np.sqrt(np.mean(frames_here**2, axis=1)) + 1e-10
    # Lokaler CV in gleitenden Fenstern (10 Frames, 50 % Overlap)
    win = min(10, len(rms_vals))
    local_cvs = [
        float(np.std(rms_vals[i : i + win]) / (np.mean(rms_vals[i : i + win]) + 1e-10))
        for i in range(0, len(rms_vals) - win + 1, max(1, win // 2))
    ]
    if not local_cvs:
        return 0.7
    cv = float(np.mean(local_cvs))
    if 0.05 <= cv <= 0.20:
        return 1.0
    elif cv < 0.05:
        result = float(cv / 0.05)  # Zu uniform → linear 0→1
        result = np.nan_to_num(result, nan=0.7, posinf=1.0, neginf=0.0)
        return float(np.clip(result, 0.0, 1.0))
    else:
        result = float(max(0.0, 1.0 - (cv - 0.20) / 0.30))  # Zu variabel → 1→0
        # NaN/Inf-Guard (§3.1)
        result = np.nan_to_num(result, nan=0.7, posinf=1.0, neginf=0.0)
        return float(np.clip(result, 0.0, 1.0))


# ─── Score-Berechnung ────────────────────────────────────────────────────


@dataclass
class MusicMOS:
    """Music MOS Scores (1–5, analog DNSMOS P.835)."""

    MUSIC_SIG: float  # Signal Quality
    MUSIC_BAK: float  # Background Quality
    MUSIC_OVR: float  # Overall Quality
    MUSIC_NAT: float  # Naturalness

    def to_dict(self) -> dict[str, float]:
        return {
            "MUSIC_SIG": round(self.MUSIC_SIG, 3),
            "MUSIC_BAK": round(self.MUSIC_BAK, 3),
            "MUSIC_OVR": round(self.MUSIC_OVR, 3),
            "MUSIC_NAT": round(self.MUSIC_NAT, 3),
        }


def _scale_to_mos(raw: float, mos_min: float = 1.5, mos_max: float = 5.0) -> float:
    """Skaliert normierten Rohwert [0, 1] auf MOS-Skala [1–5]."""
    return float(np.clip(mos_min + raw * (mos_max - mos_min), 1.0, 5.0))


def score_music_mos(audio: np.ndarray, sample_rate: int) -> MusicMOS:
    """
    Berechnet Music-MOS (analog DNSMOS P.835) für Musikmaterial.

    Args:
        audio: numpy ndarray, mono oder stereo, beliebige Samplerate
        sample_rate: Samplerate des Eingabesignals

    Returns:
        MusicMOS dataclass mit MUSIC_SIG, MUSIC_BAK, MUSIC_OVR, MUSIC_NAT
    """
    # Plugin-Pfad (wenn ML-Scorer verfügbar)
    try:
        import importlib

        plugin = importlib.import_module("music_mos_plugin")
        if hasattr(plugin, "score"):
            return plugin.score(audio, sample_rate)  # type: ignore[no-any-return]
    except ImportError:
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # DSP-Pfad
    resampled = _resample_to_16k(audio, sample_rate)
    if np.sqrt(np.mean(resampled**2)) < _MIN_SIGNAL_RMS:
        return MusicMOS(1.0, 1.0, 1.0, 1.0)

    frames = _frame_audio(resampled)

    # MUSIC_SIG: Harmonizität × Envelope-Glätte × dynamische Breite
    harmonicity = _harmonicity(frames)
    dyn_range = min(1.0, max(0.0, (_broadband_dynamic_range(resampled) - 10) / 40))
    music_sig_raw = 0.5 * harmonicity + 0.3 * _envelope_smoothness(resampled) + 0.2 * dyn_range
    MUSIC_SIG = _scale_to_mos(music_sig_raw)

    # MUSIC_BAK: 1 - f(Rauschen, Klicks, Hum)
    noise_floor = _noise_floor_db(resampled)
    noise_norm = max(0.0, min(1.0, (noise_floor + 60) / 55))  # -60→0, -5→1
    click_bad = min(1.0, _click_density(resampled) * 200)
    hum_bad = min(1.0, _hum_energy_ratio(resampled) * 20)
    music_bak_raw = noise_norm * (1 - 0.5 * click_bad) * (1 - 0.5 * hum_bad)
    MUSIC_BAK = _scale_to_mos(music_bak_raw)

    # MUSIC_NAT: 5-Komponenten-Score für maximale Naturalness-Auflösung
    # Centroid-Stabilität + Envelope-Glätte + Harmonizität (klassisch)
    # + Spectral Flux Continuity (neu v10.0.0) — glatte zeitl. Evolution
    # + Micro-Dynamic Variation (neu v10.0.0) — natürl. Amplitudenmodulation
    music_nat_raw = (
        0.28 * _spectral_centroid_stability(frames)
        + 0.22 * _envelope_smoothness(resampled)
        + 0.22 * harmonicity
        + 0.18 * _spectral_flux_continuity(frames)
        + 0.10 * _micro_dynamic_variation(resampled)
    )
    MUSIC_NAT = _scale_to_mos(music_nat_raw)

    # MUSIC_OVR: Gewichtetes Mittel — v10.0.0: NAT-Gewicht 0.25→0.35,
    # SIG 0.40→0.33, BAK 0.35→0.32 (validiert gegen Referenz-Corpus)
    MUSIC_OVR = _scale_to_mos(0.33 * (MUSIC_SIG - 1) / 4 + 0.32 * (MUSIC_BAK - 1) / 4 + 0.35 * (MUSIC_NAT - 1) / 4)

    return MusicMOS(
        MUSIC_SIG=MUSIC_SIG,
        MUSIC_BAK=MUSIC_BAK,
        MUSIC_OVR=MUSIC_OVR,
        MUSIC_NAT=MUSIC_NAT,
    )


def _broadband_dynamic_range(audio: np.ndarray) -> float:
    """Dynamikumfang in dB (P95 - P5 der Energieverteilung)."""
    frames = _frame_audio(audio)
    rms_vals = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-10)
    if len(rms_vals) < 4:
        return 20.0
    result = float(np.percentile(rms_vals, 95) - np.percentile(rms_vals, 5))
    # NaN/Inf-Guard (§3.1)
    result = np.nan_to_num(result, nan=20.0, posinf=60.0, neginf=0.0)
    return float(np.clip(result, 0.0, 80.0))
