import numpy as np
import scipy.signal as signal

logger = __import__("logging").getLogger(__name__)


def analyze_defects_features(audio, sr):
    defects = set()
    # Mono-Summe für Analyse
    audio_mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
    # Knacken/Crackle (Vinyl/Shellac): hohe Amplitudenänderungen, Impulsdetektion
    crackle_events: int = int(np.sum(np.abs(np.diff(audio_mono)) > 0.5))
    if crackle_events > sr * 0.1:
        defects.add("crackle")
    # Rauschen/Hiss: hoher Energieanteil im Hochfrequenzbereich
    # nperseg dynamisch setzen, damit f und Pxx immer gleiche Länge haben
    nperseg = min(256, len(audio_mono)) if len(audio_mono) >= 2 else 2
    f, Pxx = signal.welch(audio_mono, sr, nperseg=nperseg)
    hiss_mask = f > 8000
    hiss_energy = np.mean(Pxx[hiss_mask]) if np.any(hiss_mask) else 0
    if hiss_energy > 0.01:
        defects.add("hiss")
    # Kompressionsartefakte (MP3): Fluktuationen im Spektrum, hohe Zero Crossing Rate
    zcr = np.mean(np.abs(np.diff(np.sign(audio_mono))))
    if zcr > 0.15:
        defects.add("compression_artifacts")
    # Wow/Flutter (Tape): periodische Modulation im Bassbereich
    bass = signal.lfilter([1], [1, -0.99], audio_mono)
    wow_flutter = np.std(bass)
    if wow_flutter > 0.05:
        defects.add("wow_flutter")
    # Dropouts: längere Abschnitte mit sehr niedriger Energie
    window = int(sr * 0.05)
    dropout_windows: int = int(
        np.sum(
            [np.mean(np.abs(audio_mono[i : i + window])) < 0.005 for i in range(0, len(audio_mono) - window, window)]
        )
    )
    if dropout_windows > 2:
        defects.add("dropouts")
    # Clipping: Samples am Maximalwert
    if np.sum(np.abs(audio_mono) > 0.98) > sr * 0.01:
        defects.add("clipping")
    # Brummen/Hum: Energie bei 50/60 Hz
    hum_mask = (f > 49) & (f < 61)
    hum_energy = np.mean(Pxx[hum_mask]) if np.any(hum_mask) else 0
    if hum_energy > 0.01:
        defects.add("hum")
    return defects


def detect_media_chain_and_defects(file_path):
    from backend.file_import import load_audio_file

    result = load_audio_file(file_path, do_carrier_analysis=False)
    if result is None or result.get("error") or result["audio"] is None:
        logger.error("erkennen_media_chain_and_defects: Audio-Import fehlgeschlagen: %s", file_path)
        return [], set()
    audio, sr = result["audio"], result["sr"]
    defects = analyze_defects_features(audio, sr)
    chain = []
    # Mapping von Defekten auf Medienkette
    if "crackle" in defects:
        chain.append("vinyl")
    if "hiss" in defects or "wow_flutter" in defects:
        chain.append("tape")
    if "compression_artifacts" in defects:
        chain.append("mp3")
    if "hum" in defects:
        chain.append("shellac")
    if not chain:
        chain = ["unknown"]
    if not defects:
        defects = {"unknown"}
    return chain, list(defects)
