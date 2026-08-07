"""
§2.35c [RELEASE_MUST] Vocal Register Detector — Aurik 10.0.0

Erkennt das Vokalregister (Kopfstimme / Bruststimme / Fry / Flüstern) aus dem
Audiosignal via FCPE-F0 + spektraler Flachheit. Gibt einen register-adaptiven
energy_bias_db-Wert zurück, der von NR-Algorithmen (DeepFilterNet, OMLSA, SGMSE+)
verwendet wird.

Mapping (§2.35c Spec normativ):
    Kopfstimme (head voice):  energy_bias = -3 dB  (hohe Harmonik-Dichte → konservativ)
    Bruststimme (chest voice): energy_bias = -6 dB  (Default; mittlere Harmonik-Energie)
    Fry / Flüstern:            energy_bias = -9 dB  (niedrige Harmonik-Kohärenz → aggressiver)

Singleton-Pattern (thread-safe double-checked locking).
"""

from __future__ import annotations

import logging
import threading

import numpy as np

# pylint: disable=import-outside-toplevel

logger = logging.getLogger(__name__)

# Vokalregister-Schwellen für F0-basierte Klassifikation
# Kopfstimme: F0 > 300 Hz (Sopran/Tenor-Kopfregister); Fry: F0 < 80 Hz oder stark inharmonisch
_HEAD_VOICE_F0_HZ = 300.0
_FALSETTO_F0_HZ = 350.0  # Falsetto: F0 > 350 Hz + teilweise atemhaft (§0p)
_FRY_F0_HZ = 80.0
_FRY_FLATNESS_THRESHOLD = 0.60  # hohe spektrale Flachheit = wenig harmonische Struktur
_FALSETTO_FLATNESS_MIN = 0.30  # Falsetto: leicht atemhafte Textur, F1-Absenkung
_FALSETTO_FLATNESS_MAX = 0.72  # Grenze zu Flüstern
_WHISPER_FLATNESS_THRESHOLD = 0.75  # sehr flach = Flüstern

# Energy-Bias-Werte pro Register (dB, negativ = Harmonik-Schutz)
_ENERGY_BIAS_HEAD = -3.0
_ENERGY_BIAS_CHEST = -6.0
_ENERGY_BIAS_FRY_WHISPER = -9.0
_ENERGY_BIAS_FALSETTO = -4.5  # Falsetto: zwischen Kopf (-3) und Fry (-9)

# Kanonisches Register→Energy-Bias-Mapping — einzige Quelle der Wahrheit (§0j / §0p).
# Importierbar von allen DSP-/Plugin-Modulen; nie lokal duplizieren.
REGISTER_BIAS: dict[str, float] = {
    "chest": _ENERGY_BIAS_CHEST,  # −6.0 dB
    "head": _ENERGY_BIAS_HEAD,  # −3.0 dB
    "falsetto": _ENERGY_BIAS_FALSETTO,  # −4.5 dB
    "fry_whisper": _ENERGY_BIAS_FRY_WHISPER,  # −9.0 dB  (kombiniertes Detektor-Label)
    "passaggio": _ENERGY_BIAS_HEAD,  # −3.0 dB  (Mittelwert Brust/Kopf, §0p)
    "fry": -12.0,  # −12.0 dB (eigenständiges Fry-Label, §0p)
    "whisper": -15.0,  # −15.0 dB (maximaler Schutz, §0p)
    "unknown": _ENERGY_BIAS_CHEST,  # −6.0 dB  (Fallback)
}


def _spectral_flatness(mono: np.ndarray, sr: int) -> float:
    """Mittlere spektrale Flachheit (Wiener-Entropie, [0,1])."""
    try:
        from scipy.signal import welch as _welch

        nperseg = min(2048, len(mono))
        if nperseg < 64:
            return 0.5
        _, psd = _welch(mono.astype(np.float64), fs=sr, nperseg=nperseg)
        psd = np.maximum(psd, 1e-12)
        geo_mean = float(np.exp(np.mean(np.log(psd))))
        arith_mean = float(np.mean(psd))
        return float(np.clip(geo_mean / (arith_mean + 1e-12), 0.0, 1.0))
    except Exception as e:
        logger.warning("vocal_register_detector.py::_spectral_flatness Ersatzpfad: %s", e)
        return 0.5


def _estimate_f0_median(mono: np.ndarray, sr: int) -> float | None:
    """Schätzt medianen F0 via FCPE (Primär) → pYIN (Fallback) → None."""
    try:
        from plugins.fcpe_plugin import get_fcpe_plugin as _get_fcpe

        result = _get_fcpe().analyze(mono, sr)
        f0 = result.get("f0") if isinstance(result, dict) else None
        if f0 is not None and len(f0) > 0:
            voiced = np.asarray(f0)[np.asarray(f0) > 50.0]
            if len(voiced) >= 3:
                return float(np.median(voiced))
    except Exception as e:
        logger.warning("vocal_register_detector.py::_estimate_f0_median Ersatzpfad: %s", e)

    # pYIN-Fallback
    try:
        import librosa  # type: ignore[import]

        f0_pyin, voiced_flag, _ = librosa.pyin(
            mono.astype(np.float32),
            fmin=50.0,
            fmax=1000.0,
            sr=sr,
            frame_length=2048,
        )
        voiced_f0 = f0_pyin[voiced_flag & (f0_pyin > 0)]
        if len(voiced_f0) >= 3:
            return float(np.median(voiced_f0))
    except Exception as e:
        logger.warning("vocal_register_detector.py::_estimate_f0_median Ersatzpfad: %s", e)

    return None


def detect_vocal_register(
    audio: np.ndarray,
    sr: int,
    panns_singing: float = 0.0,
) -> tuple[str, float]:
    """
    Erkennt Vokalregister und gibt (register_label, energy_bias_db) zurück.

    Args:
        audio:         Mono oder Stereo, float32, SR=48000
        sr:            Abtastrate (48000 erwartet)
        panns_singing: PANNs-Gesangskonfidenz [0,1] — bei < 0.25 wird Fallback genutzt

    Returns:
        (register, energy_bias_db) — register ∈ {"head", "chest", "fry_whisper", "falsetto", "unknown"}
        energy_bias_db ∈ {-3.0, -4.5, -6.0, -9.0}
    """
    # Mono extrahieren
    mono: np.ndarray
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1 if audio.shape[1] <= 8 else 0).astype(np.float64)
    else:
        mono = audio.astype(np.float64)

    # Bei fehlendem Gesangs-Evidenz: Chest-Default (−6 dB)
    if panns_singing < 0.25:
        return "chest", _ENERGY_BIAS_CHEST

    # Spektrale Flachheit — erkennt Fry/Flüstern
    flatness = _spectral_flatness(mono, sr)
    if flatness >= _WHISPER_FLATNESS_THRESHOLD:
        logger.debug(
            "§2.35c VocalRegister: flatness=%.3f → Flüstern (energy_bias=%.1f dB)",
            flatness,
            _ENERGY_BIAS_FRY_WHISPER,
        )
        return "fry_whisper", _ENERGY_BIAS_FRY_WHISPER

    # F0-Schätzung für Head/Chest-Unterscheidung
    f0_med = _estimate_f0_median(mono[: min(len(mono), int(60 * sr))], sr)

    if f0_med is None:
        # F0 nicht schätzbar: Flachheit als Tiebreaker
        if flatness >= _FRY_FLATNESS_THRESHOLD:
            return "fry_whisper", _ENERGY_BIAS_FRY_WHISPER
        return "chest", _ENERGY_BIAS_CHEST

    if f0_med < _FRY_F0_HZ:
        logger.debug(
            "§2.35c VocalRegister: f0_median=%.1f Hz < 80 Hz → Fry (energy_bias=%.1f dB)",
            f0_med,
            _ENERGY_BIAS_FRY_WHISPER,
        )
        return "fry_whisper", _ENERGY_BIAS_FRY_WHISPER

    if f0_med >= _FALSETTO_F0_HZ and _FALSETTO_FLATNESS_MIN <= flatness <= _FALSETTO_FLATNESS_MAX:
        # Falsetto: hohe F0 + teilweise atemhaft (F1-Absenkung, §0p)
        logger.debug(
            "§2.35c VocalRegister: f0_median=%.1f Hz + flatness=%.3f → Falsett (energy_bias=%.1f dB)",
            f0_med,
            flatness,
            _ENERGY_BIAS_FALSETTO,
        )
        return "falsetto", _ENERGY_BIAS_FALSETTO

    if f0_med >= _HEAD_VOICE_F0_HZ:
        logger.debug(
            "§2.35c VocalRegister: f0_median=%.1f Hz ≥ 300 Hz → Kopfstimme (energy_bias=%.1f dB)",
            f0_med,
            _ENERGY_BIAS_HEAD,
        )
        return "head", _ENERGY_BIAS_HEAD

    logger.debug(
        "§2.35c VocalRegister: f0_median=%.1f Hz → Bruststimme (energy_bias=%.1f dB)",
        f0_med,
        _ENERGY_BIAS_CHEST,
    )
    return "chest", _ENERGY_BIAS_CHEST


# ---------------------------------------------------------------------------
# Thread-safe Singleton-Wrapper
# ---------------------------------------------------------------------------


class _VocalRegisterCache:
    """Leichtgewichtiger Cache: Ergebnis gilt für max. 120 s Audio (4 MB Mono)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[int, tuple[str, float]] = {}  # id(audio) → result

    def get_or_compute(self, audio: np.ndarray, sr: int, panns_singing: float) -> tuple[str, float]:
        """Gibt cached register detection for the audio object or compute it zurück."""
        key = id(audio)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            result = detect_vocal_register(audio, sr, panns_singing)
            self._cache[key] = result
            if len(self._cache) > 16:
                # LRU-Annäherung: ältestes Element entfernen
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            return result


_cache_instance: _VocalRegisterCache | None = None
_cache_lock = threading.Lock()


def get_vocal_register_cache() -> _VocalRegisterCache:
    """Singleton-Zugriff auf den Register-Cache."""
    global _cache_instance  # pylint: disable=global-statement
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = _VocalRegisterCache()
    return _cache_instance


# ---------------------------------------------------------------------------
# §Passaggio-Schutz — Registerübergangs-Glättung (v10.0.0)
# ---------------------------------------------------------------------------

# Fensterbreite (in Frames) für Übergangsglätte um erkannte Passaggio-Punkte.
# ±5 Frames bei 30 ms Hop = ±150 ms Glättungszone.
_PASSAGGIO_SMOOTH_HALF_WINDOW = 5


def detect_vocal_register_temporal(
    audio: np.ndarray,
    sr: int,
    panns_singing: float = 0.0,
    segment_ms: float = 300.0,
    hop_ms: float = 100.0,
) -> list[tuple[float, float, str, float]]:
    """
    Zeitaufgelöste Register-Erkennung mit Passaggio-Glättung.

    Gibt eine Liste von (start_s, end_s, register, energy_bias_db) zurück.
    An erkannten Registerübergängen (Passaggio) wird eine Glättungszone
    von ±_PASSAGGIO_SMOOTH_HALF_WINDOW Frames angelegt, in der der
    energy_bias linear zwischen den Registerwerten interpoliert wird.

    §Passaggio: Registersprünge (Brust→Kopf) erzeugen bei frameweise
    unabhängiger Klassifikation einen abrupten energy_bias-Sprung von
    -6 dB auf -3 dB. Das erzeugt einen messbaren Timbre-Knick an jeder
    Registergrenze. Diese Funktion glättet den Übergang.

    Args:
        audio:        Eingangssignal (mono/stereo, float32)
        sr:           Abtastrate (48000 Hz erwartet)
        panns_singing: PANNs-Gesangskonfidenz [0,1]
        segment_ms:   Analysefensterbreite in ms
        hop_ms:       Hop-Schrittweite in ms

    Returns:
        Liste von (start_s, end_s, register, energy_bias_db)
    """
    if panns_singing < 0.25:
        # Kein Gesang erkannt → uniform Chest-Default
        duration_s = len(audio) / sr if audio.ndim == 1 else audio.shape[-1] / sr
        return [(0.0, float(duration_s), "chest", _ENERGY_BIAS_CHEST)]

    mono: np.ndarray
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1 if audio.shape[1] <= 8 else 0).astype(np.float32)
    else:
        mono = audio.astype(np.float32)

    n = len(mono)
    seg_len = max(64, int(segment_ms / 1000.0 * sr))
    hop_len = max(32, int(hop_ms / 1000.0 * sr))

    # Per-Segment F0-Schätzung (FCPE) + Register-Klassifikation
    registers: list[str] = []
    biases: list[float] = []

    try:
        from plugins.fcpe_plugin import get_fcpe_plugin as _get_fcpe

        _fcpe_result = _get_fcpe().analyze(mono, sr)
        f0_full = (
            np.asarray(_fcpe_result.get("f0", []), dtype=np.float32) if isinstance(_fcpe_result, dict) else np.array([])
        )
    except Exception:
        f0_full = np.array([])

    for start in range(0, n - seg_len + 1, hop_len):
        seg = mono[start : start + seg_len]
        flatness = _spectral_flatness(seg, sr)

        # Whisper / Fry aus globaler F0 im Segment-Zeitbereich
        seg_start_s = start / sr
        seg_end_s = (start + seg_len) / sr

        if flatness >= _WHISPER_FLATNESS_THRESHOLD:
            registers.append("fry_whisper")
            biases.append(_ENERGY_BIAS_FRY_WHISPER)
            continue

        # F0 im Segment-Bereich aus vorberechneten FCPE-Werten
        if len(f0_full) > 0:
            f0_hop_s = len(mono) / sr / max(len(f0_full), 1)
            fi_start = int(seg_start_s / f0_hop_s)
            fi_end = int(seg_end_s / f0_hop_s)
            seg_f0 = f0_full[max(0, fi_start) : min(len(f0_full), fi_end + 1)]
            voiced_f0 = seg_f0[seg_f0 > 50.0]
        else:
            voiced_f0 = np.array([])

        if len(voiced_f0) < 2:
            # Fallback pYIN
            try:
                import librosa  # type: ignore[import]

                f0_p, vf, _ = librosa.pyin(seg, fmin=50.0, fmax=1000.0, sr=sr, frame_length=min(2048, seg_len))
                voiced_f0 = f0_p[vf & (f0_p > 0)] if len(f0_p) > 0 else np.array([])
            except Exception:
                voiced_f0 = np.array([])

        if len(voiced_f0) == 0:
            reg = "chest" if flatness < _FRY_FLATNESS_THRESHOLD else "fry_whisper"
            biases.append(_ENERGY_BIAS_CHEST if reg == "chest" else _ENERGY_BIAS_FRY_WHISPER)
            registers.append(reg)
            continue

        f0_med = float(np.median(voiced_f0))
        if f0_med < _FRY_F0_HZ:
            registers.append("fry_whisper")
            biases.append(_ENERGY_BIAS_FRY_WHISPER)
        elif f0_med >= _FALSETTO_F0_HZ and _FALSETTO_FLATNESS_MIN <= flatness <= _FALSETTO_FLATNESS_MAX:
            registers.append("falsetto")
            biases.append(_ENERGY_BIAS_FALSETTO)
        elif f0_med >= _HEAD_VOICE_F0_HZ:
            registers.append("head")
            biases.append(_ENERGY_BIAS_HEAD)
        else:
            registers.append("chest")
            biases.append(_ENERGY_BIAS_CHEST)

    if not registers:
        return [(0.0, float(n / sr), "chest", _ENERGY_BIAS_CHEST)]

    # Passaggio-Glättung: Linear interpolieren an erkannten Übergangspunkten
    biases_arr = np.array(biases, dtype=np.float64)
    n_frames = len(biases_arr)

    for i in range(1, n_frames):
        if registers[i] != registers[i - 1]:
            # Übergang erkannt (Passaggio) — Glättungszone ±W Frames
            w = _PASSAGGIO_SMOOTH_HALF_WINDOW
            i_start = max(0, i - w)
            i_end = min(n_frames, i + w + 1)
            zone_len = i_end - i_start
            bias_from = biases_arr[i_start]
            bias_to = biases_arr[i_end - 1] if i_end < n_frames else biases_arr[-1]
            # Linearer Übergang in der Glättungszone
            for k, idx in enumerate(range(i_start, i_end)):
                alpha = k / max(zone_len - 1, 1)
                biases_arr[idx] = bias_from * (1.0 - alpha) + bias_to * alpha

    # Segmente zusammenführen
    result: list[tuple[float, float, str, float]] = []
    for idx, (reg, bias) in enumerate(zip(registers, biases_arr.tolist())):
        start_s = float(idx * hop_len / sr)
        end_s = float(min((idx + 1) * hop_len + seg_len, n) / sr)
        result.append((start_s, end_s, reg, float(bias)))

    logger.debug(
        "§Passaggio erkennen_vocal_register_temporal: %d Segmente, %d Übergänge geglättet",
        len(result),
        sum(1 for i in range(1, len(registers)) if registers[i] != registers[i - 1]),
    )
    return result


# ---------------------------------------------------------------------------
# §Multi-Singer-Erkennung (v10.0.0)
# ---------------------------------------------------------------------------

# Schwellwert für simultane F0-Konturen (normierte ACF-Peak-Distanz)
_MULTI_F0_MIN_DISTANCE_HZ = 30.0  # Mindestabstand zweier Grundfrequenzen in Hz
_MULTI_F0_FRAME_VOTE_RATIO = 0.25  # 25% der Frames müssen Multi-F0 zeigen


def detect_multi_singer(
    audio: np.ndarray,
    sr: int,
    panns_singing: float = 0.0,
) -> bool:
    """
    Erkennt ob mehrere Sänger/innen gleichzeitig aktiv sind.

    Methode: Harmonic Sum Spectrum (HSS) zur Detektion simultaner F0-Konturen.
    Wenn ≥ 2 dominante F0-Hypothesen mit Mindestabstand gefunden werden in
    mindestens _MULTI_F0_FRAME_VOTE_RATIO der stimmhaften Frames, liegt
    ein Multi-Singer-Signal vor.

    Anwendung in UV3: Wenn True → metadata["multi_singer"] = True und
    singer_identity_cosine-Gate deaktiviert (Resemblyzer-Embedding ist für
    einzelne Identitätsprüfung konzipiert, nicht für Duett/Chor).

    §0c Universalität: Funktioniert für Duette, Terzette, Chöre.

    Args:
        audio:         Eingangssignal (mono/stereo)
        sr:            Abtastrate
        panns_singing: PANNs-Gesangskonfidenz [0,1]

    Returns:
        True wenn Multi-Singer-Konstellation erkannt.
    """
    # Kurzschluss: kein Gesang → kein Multi-Singer
    if panns_singing < 0.25:
        return False

    mono: np.ndarray
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1 if audio.shape[1] <= 8 else 0).astype(np.float64)
    else:
        mono = audio.astype(np.float64)

    if len(mono) < sr:
        return False

    try:
        # HSS (Harmonic Sum Spectrum) Frame-weise Multi-F0-Detektion
        n_fft = 4096  # Hohe Frequenzauflösung für F0-Trennung
        hop = 1024
        window = np.hanning(n_fft)
        n_freqs = n_fft // 2 + 1
        freqs = np.linspace(0, sr / 2.0, n_freqs)

        # F0-Bereich: 60–800 Hz (Singstimmen)
        f0_min_hz = 60.0
        f0_max_hz = 800.0

        multi_f0_frame_count = 0
        total_voiced_frames = 0

        for i in range(0, len(mono) - n_fft, hop):
            frame = mono[i : i + n_fft] * window
            rms = float(np.sqrt(np.mean(frame**2)))
            if rms < 5e-4:
                continue

            mag = np.abs(np.fft.rfft(frame))

            # HSS: Für jede Kandidat-F0, summiere Harmonik-Energie (1. bis 5. Oberton)
            f0_range_mask = (freqs >= f0_min_hz) & (freqs <= f0_max_hz)
            candidate_freqs = freqs[f0_range_mask]
            hss = np.zeros(len(candidate_freqs))

            for j, f0_cand in enumerate(candidate_freqs):
                for harmonic in range(1, 6):
                    hf = f0_cand * harmonic
                    if hf >= sr / 2.0:
                        break
                    hf_idx = int(round(hf / (sr / n_fft)))
                    hf_idx = min(hf_idx, n_freqs - 1)
                    # Interpolierte Magnitude (Hann-Fenster breitert Peaks)
                    hss[j] += mag[hf_idx] if hf_idx < n_freqs else 0.0

            if np.max(hss) < 1e-6:
                continue

            total_voiced_frames += 1

            # Suche nach dem dominantesten F0
            best_idx = int(np.argmax(hss))
            best_f0 = float(candidate_freqs[best_idx])

            # Unterdrücke ±_MULTI_F0_MIN_DISTANCE_HZ um besten Peak und suche zweiten
            suppress_mask = np.abs(candidate_freqs - best_f0) < _MULTI_F0_MIN_DISTANCE_HZ
            # Harmonische des ersten F0 ebenfalls unterdrücken
            for harm in [2.0, 3.0, 0.5]:
                harm_f0 = best_f0 * harm
                suppress_mask |= np.abs(candidate_freqs - harm_f0) < _MULTI_F0_MIN_DISTANCE_HZ * 0.5

            hss_residual = hss.copy()
            hss_residual[suppress_mask] = 0.0

            # §Gap9 v10.0.0: Zweiter Peak: muss signifikant sein (>55% des ersten).
            # Schwellwert-Anhebung von 0.40 → 0.55: Hintergrundchor typischerweise
            # 40–55% Energie des Solisten → löst kein Multi-Singer-Gate mehr aus.
            # Nur echte Duette / gleichberechtigte Chöre (> 55%) deaktivieren das
            # singer_identity_cosine-Gate (§0p VQI-Gate).
            second_peak = float(np.max(hss_residual))
            if second_peak > 0.55 * float(hss[best_idx]):
                multi_f0_frame_count += 1

        if total_voiced_frames < 5:
            return False

        multi_ratio = multi_f0_frame_count / max(total_voiced_frames, 1)
        is_multi = bool(multi_ratio >= _MULTI_F0_FRAME_VOTE_RATIO)

        logger.debug(
            "§Multi-Singer: Verhaeltnis=%.2f (%d/%d Frames) → %s",
            multi_ratio,
            multi_f0_frame_count,
            total_voiced_frames,
            "MULTI" if is_multi else "SOLO",
        )
        return is_multi

    except Exception as exc:
        logger.debug("erkennen_multi_singer (unkritisch): %s", exc)
        return False
