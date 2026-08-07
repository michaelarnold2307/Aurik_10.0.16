"""§MNF (V21) Export-Rauschboden-Guard.

Stellt nach NR-Phasen sicher, dass Rauschboden-Rekonstruktion nicht zu
unmusikalischer digitaler Stille oder zu erneutem analogem Trägerrauschen führt.
Analoge Tonträger werden im Export auf CD-ähnlichen Rauschboden gezielt; ihre
Hiss-/Oberflächenrausch-Textur darf nicht als Mindestboden zurückkehren.

Era-Textur-Erweiterung (v10.0.0): Wenn ``original_audio`` übergeben wird,
extrahiert der Guard das Spektral-Profil des Originals aus dessen Ruhezonen
und verwendet es als Vorlage für die Rausch-Injektion — statt synthetischem
Butterworth-Rauschen. Diese Textur-Rekonstruktion ist nur noch für explizite
Override-Böden aktiv; normale Analogträger bekommen keinen analogen Mindestboden.

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.noise_floor_guard import apply_noise_floor_minimum
    result.audio = apply_noise_floor_minimum(result.audio, sr, material,
                                             original_audio=pre_nr_audio)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Material-Rauschböden in dBFS.
# None = kein analoger Mindestboden; der Export darf auf CD-ähnlichen Boden fallen.
_MATERIAL_FLOORS_DBFS: dict[str, float | None] = {
    "shellac": None,
    "wax_cylinder": None,
    "lacquer_disc": None,
    "wire_recording": None,
    "reel_tape": None,
    "tape": None,
    "vinyl": None,
    "cassette": None,
    "minidisc": None,
    "cd_digital": None,
    "dat": None,
    "mp3_low": None,
    "mp3_high": None,
    "unknown": None,
}

# Low-Pass-Eckfrequenz pro Material (rauschfärbend)
_MATERIAL_NOISE_CUTOFF_HZ: dict[str, float] = {
    "shellac": 7000.0,
    "wax_cylinder": 5000.0,
    "lacquer_disc": 8000.0,
    "wire_recording": 6000.0,
    "reel_tape": 16000.0,
    "tape": 16000.0,
    "vinyl": 18000.0,
    "cassette": 14000.0,
}


def _shaped_noise(n_samples: int, sr: int, cutoff_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Einfaches pink-geformtes Rauschen (Butterworth LP) für Rauschboden-Injektion."""
    white = rng.standard_normal(n_samples).astype(np.float32)
    sos = butter(2, cutoff_hz / (sr / 2.0), btype="low", output="sos")
    return np.asarray(sosfiltfilt(sos, white), dtype=np.float32)  # type: ignore[no-any-return]


def _extract_noise_psd_profile(
    audio_1ch: np.ndarray,
    silence_threshold: float,
    frame_len: int,
    min_frames: int = 8,
) -> np.ndarray | None:
    """Extrahiert das mittlere PSD-Profil aus Ruhezonen des Original-Audios.

    Args:
        audio_1ch: 1-Kanal-Audio (float32).
        silence_threshold: Maximale RMS-Amplitude für einen "leisen" Frame.
        frame_len: Frame-Länge in Samples.
        min_frames: Mindestanzahl leiser Frames für verlässliches Profil.

    Returns:
        PSD-Profil als 1D-Array (Länge = frame_len // 2 + 1) oder None wenn
        zu wenige leise Frames gefunden wurden.
    """
    n = len(audio_1ch)
    n_frames = n // frame_len
    fft_len = frame_len
    psd_sum = np.zeros(fft_len // 2 + 1, dtype=np.float64)
    n_quiet = 0
    for i in range(n_frames):
        start = i * frame_len
        end = start + frame_len
        seg = audio_1ch[start:end]
        rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
        if rms < silence_threshold:
            spectrum = np.abs(np.fft.rfft(seg.astype(np.float64), n=fft_len)) ** 2
            psd_sum += spectrum
            n_quiet += 1
    if n_quiet < min_frames:
        return None
    psd_mean = psd_sum / n_quiet
    # Sicherheits-Floor: verhindert Null-Division bei sehr stillem Original
    psd_mean = np.maximum(psd_mean, 1e-20)
    return psd_mean.astype(np.float64)  # type: ignore[no-any-return]


def _generate_texture_noise(
    n_samples: int,
    psd_profile: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Erzeugt Rauschen mit dem spektralen Profil von ``psd_profile``.

    Args:
        n_samples: Gewünschte Sample-Anzahl.
        psd_profile: PSD-Profil (Output von :func:`_extract_noise_psd_profile`).
        rng: Zufallsgenerator.

    Returns:
        Rauschen mit originalgetreuer Spektralfarbe, normiert auf RMS ≈ 1.
    """
    fft_len = (len(psd_profile) - 1) * 2
    white = rng.standard_normal(n_samples).astype(np.float64)
    n_blocks = max(1, n_samples // fft_len)
    result = np.zeros(n_samples, dtype=np.float64)
    amplitude_profile = np.sqrt(psd_profile)
    amplitude_profile /= amplitude_profile.mean() + 1e-20  # Normierung
    for b in range(n_blocks):
        start = b * fft_len
        end = min(start + fft_len, n_samples)
        seg = white[start:end]
        if len(seg) < fft_len:
            seg = np.pad(seg, (0, fft_len - len(seg)))
        spectrum = np.fft.rfft(seg, n=fft_len)
        spectrum *= amplitude_profile
        shaped = np.fft.irfft(spectrum, n=fft_len).real
        result[start:end] += shaped[: end - start]
    # Normieren auf RMS ≈ 1
    rms = float(np.sqrt(np.mean(result**2) + 1e-12))
    result /= rms + 1e-12
    return result.astype(np.float32)  # type: ignore[no-any-return]


def apply_noise_floor_minimum(
    audio: np.ndarray,
    sr: int,
    material: str,
    floor_dbfs: float | None = None,
    *,
    frame_ms: float = 20.0,
    original_audio: np.ndarray | None = None,
) -> np.ndarray:
    """Hebt vollständig stille Frames nur bei explizitem Override auf einen Boden an.

    Era-Textur-Rekonstruktion (v10.0.0): Wenn ``original_audio`` übergeben wird,
    wird das Spektral-Profil des Trägers aus dessen Ruhezonen extrahiert und als
    Vorlage für das Ersatzrauschen verwendet. Standardmäßig ist dieser Pfad für
    analoge Tonträger deaktiviert, damit der Export CD-ähnlichen Rauschboden
    statt analogem Trägerrauschen behalten kann. Ohne ``original_audio`` wird
    synthetisches Butterworth-Rauschen injiziert, falls ein Override gesetzt ist.

    Args:
        audio: Audio nach NR-Phase. Shape [N] oder [2, N].
        sr: Sample-Rate (muss 48000 sein).
        material: Materialklasse (z.B. ``"vinyl"``).
        floor_dbfs: Optionaler Override für den Rauschboden in dBFS.
            Wenn None, wird der materialspezifische Wert aus ``_MATERIAL_FLOORS_DBFS`` verwendet.
            Analoge Standardmaterialien haben dort bewusst None.
        frame_ms: Frame-Länge in ms für Stille-Detektion.
        original_audio: Optionales Pre-NR-Audio (gleiche Shape wie ``audio``).
            Wenn übergeben, wird das Spektralprofil des Originals für era-
            konforme Rausch-Textur-Rekonstruktion verwendet.

    Returns:
        Audio mit angehobenem Rauschboden (Float32, geclippt auf [-1.0, 1.0]).
    """
    assert sr == 48000
    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        mat_key = str(material).lower().strip()

        # Floor bestimmen
        effective_floor: float | None = floor_dbfs
        if effective_floor is None:
            effective_floor = _MATERIAL_FLOORS_DBFS.get(mat_key)
        if effective_floor is None:
            return audio  # Kein analoger Boden nötig

        floor_linear = float(10.0 ** (effective_floor / 20.0))
        # Stille-Schwelle: 6 dB unter dem Mindestboden
        silence_threshold = floor_linear * 0.5

        frame_len = max(64, int(sr * frame_ms / 1000.0))
        cutoff = _MATERIAL_NOISE_CUTOFF_HZ.get(mat_key, 18000.0)

        rng = np.random.default_rng(seed=42)  # Deterministisch für Tests
        is_stereo = audio.ndim == 2

        # Era-Textur: PSD-Profile aus Original-Audio extrahieren (kanal-weise)
        _psd_profiles: list[np.ndarray | None] = [None, None]
        if original_audio is not None:
            try:
                orig_norm = np.nan_to_num(original_audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                # Profil für jeden Kanal separat (Stereo) oder einmalig (Mono)
                if is_stereo and orig_norm.ndim == 2:
                    for _ci in range(min(2, orig_norm.shape[0])):
                        _psd_profiles[_ci] = _extract_noise_psd_profile(
                            orig_norm[_ci], silence_threshold * 2.0, frame_len
                        )
                else:
                    _orig_1ch = orig_norm if orig_norm.ndim == 1 else orig_norm[0]
                    _psd_base = _extract_noise_psd_profile(_orig_1ch, silence_threshold * 2.0, frame_len)
                    _psd_profiles[0] = _psd_base
                    _psd_profiles[1] = _psd_base
            except Exception as _psd_exc:
                logger.debug("Era-Textur PSD-Extraktion (nicht blockierend): %s", _psd_exc)

        def _process_channel(ch: np.ndarray, ch_idx: int = 0) -> np.ndarray:
            ch_out = ch.copy()
            n = len(ch)
            n_frames = n // frame_len
            # Noise-Quelle: Original-Textur wenn verfügbar, sonst Butterworth-Synthetik
            psd = _psd_profiles[ch_idx]
            if psd is not None:
                noise = _generate_texture_noise(n, psd, rng)
                logger.debug("V21 Era-Textur-Rauschen (material=%s): originales PSD-Profil aktiv", mat_key)
            else:
                noise = _shaped_noise(n, sr, cutoff, rng)
            for i in range(n_frames):
                start = i * frame_len
                end = start + frame_len
                rms = float(np.sqrt(np.mean(ch[start:end] ** 2) + 1e-12))
                if rms < silence_threshold:
                    # Rauschen auf exaktem Bodenpegel skalieren
                    noise_seg = noise[start:end]
                    noise_rms = float(np.sqrt(np.mean(noise_seg**2) + 1e-12))
                    scale = floor_linear / (noise_rms + 1e-12)
                    ch_out[start:end] = noise_seg * scale
            return ch_out

        if is_stereo:
            ch0 = _process_channel(audio[0], ch_idx=0)
            ch1 = _process_channel(audio[1], ch_idx=1)
            result = np.stack([ch0, ch1], axis=0)
        else:
            result = _process_channel(audio, ch_idx=0)

        out: np.ndarray = np.clip(result, -1.0, 1.0).astype(np.float32)
        return out

    except Exception as exc:
        logger.debug("anwenden_noise_floor_minimum nicht blockierend: %s", exc)
        return audio
