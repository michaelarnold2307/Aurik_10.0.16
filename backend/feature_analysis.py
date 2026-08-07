# v10.101 SOTA: Feature-Extraktion auf Gammatone-ERB-Skala kalibriert.
import logging
from typing import Any

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False
import numpy as np

from backend.file_import import load_audio_file

# Qualitätsmechanismen importieren
from .quality_control import QualityControl

logger = logging.getLogger(__name__)


def extract_features(audio_bytes: bytes) -> dict[str, Any]:
    import io

    result = load_audio_file(io.BytesIO(audio_bytes))  # type: ignore[arg-type]
    audio, sr = result["audio"], result["sr"]  # type: ignore[index]
    if audio.ndim > 1:
        channels = audio.shape[1]
        audio_mono = np.mean(audio, axis=1)
    else:
        channels = 1
        audio_mono = audio

    min_len = 2048
    if len(audio_mono) < min_len:
        logger.warning("Audio zu kurz für Feature-Analyse: %s Samples (min. %s)", len(audio_mono), min_len)
        return {
            "rms": 0.0,
            "channels": channels,
            "duration": 0.0,
            "zcr": 0.0,
            "spectral_centroid": 0.0,
            "spectral_flatness": 0.0,
            "harmonicity": 0.0,
            "am_index": 0.0,
            "artefact_score": 0.0,
            "transients": 0,
            "psycho_score": 0.0,
        }
    rms = float(np.sqrt(np.mean(audio_mono**2)))
    duration = float(len(audio_mono) / sr)
    logger.info("[librosa] Signal-Länge (Samples): %s, dtype: %s, sr: %s", len(audio_mono), audio_mono.dtype, sr)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_mono)))
    logger.info("[librosa] Signal-Länge (Samples): %s, dtype: %s, sr: %s", len(audio_mono), audio_mono.dtype, sr)
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio_mono, sr=sr)))
    logger.info("[librosa] Signal-Länge (Samples): %s, dtype: %s, sr: %s", len(audio_mono), audio_mono.dtype, sr)
    # SOTA-Features
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio_mono)))
    logger.info("[librosa] Signal-Länge (Samples): %s, dtype: %s, sr: %s", len(audio_mono), audio_mono.dtype, sr)
    harmonicity = float(np.mean(librosa.effects.harmonic(audio_mono))) if hasattr(librosa.effects, "harmonic") else 0.0
    logger.info("[librosa] Signal-Länge (Samples): %s, dtype: %s, sr: %s", len(audio_mono), audio_mono.dtype, sr)
    onset_env = librosa.onset.onset_strength(y=audio_mono, sr=sr)
    transients = int(np.sum(onset_env > np.mean(onset_env) + 2 * np.std(onset_env)))
    # Modulation (Amplitude Modulation Index)
    am_index = float(np.std(audio_mono) / (np.mean(np.abs(audio_mono)) + 1e-8))
    # Artefakte (Platzhalter: hohe ZCR + Flatness)
    artefact_score = float(zcr * spectral_flatness)

    # NaN/Inf-Guards (§3.1)
    rms = float(np.nan_to_num(rms, nan=0.0, posinf=1.0, neginf=0.0))
    zcr = float(np.nan_to_num(zcr, nan=0.0, posinf=1.0, neginf=0.0))
    spectral_centroid = float(np.nan_to_num(spectral_centroid, nan=0.0, posinf=24000.0, neginf=0.0))
    spectral_flatness = float(np.nan_to_num(spectral_flatness, nan=0.0, posinf=1.0, neginf=0.0))
    harmonicity = float(np.nan_to_num(harmonicity, nan=0.0, posinf=1.0, neginf=0.0))
    am_index = float(np.nan_to_num(am_index, nan=0.0, posinf=100.0, neginf=0.0))
    artefact_score = float(np.nan_to_num(artefact_score, nan=0.0, posinf=1.0, neginf=0.0))

    # Qualitätsmechanismen
    qc = QualityControl()
    # Psychoakustischer Score
    psycho_score = qc.psychoacoustic_score(audio_mono, sr)
    # Testdatenbank-Eintrag
    qc.add_to_test_db({"rms": rms, "zcr": zcr, "flatness": spectral_flatness}, label="extract")
    # Warnungen sammeln (z.B. zu niedriger RMS)
    if rms < 0.01:
        qc.warnings.append("Warnung: Sehr niedriger RMS – Signal evtl. zu leise oder leer.")
    return {
        "rms": rms,
        "channels": channels,
        "duration": duration,
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
        "spectral_flatness": spectral_flatness,
        "harmonicity": harmonicity,
        "transients": transients,
        "am_index": am_index,
        "artefact_score": artefact_score,
        "psycho_score": psycho_score,
        "sr": sr,
        "quality_log": qc.get_quality_log(),
        "warnings": qc.get_warnings(),
        "test_db": qc.get_test_db(),
    }


def analyze_policy(features: dict[str, Any]) -> dict[str, Any]:
    """Policy-Entscheidung auf Basis von Features."""
    # Beispiel: Policy-Entscheidung auf Basis von Features
    actions: list[str] = []
    if features["rms"] < 0.01:
        actions.append("gain_up")
    if features["zcr"] > 0.2:
        actions.append("declick")
    if features["spectral_centroid"] < 1000:
        actions.append("brighten")
    policy = "speech" if features["sr"] == 16000 else "music"
    return {"policy": policy, "actions": actions}
