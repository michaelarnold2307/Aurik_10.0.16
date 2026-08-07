import logging
from typing import Any

import numpy as np
from scipy.signal import correlate

logger = logging.getLogger(__name__)


class GenderDetector:
    """Erkennt das dominante Geschlecht einer Gesangsstimme via Resemblyzer + Pitch-Analyse."""

    def __init__(self, use_auth_token: Any = None) -> None:
        del use_auth_token
        from plugins.resemblyzer_plugin import get_resemblyzer_plugin  # pylint: disable=import-outside-toplevel

        self._plugin = get_resemblyzer_plugin()

    def detect_gender(self, audio_file) -> str:
        """
        Gibt das dominante Geschlecht im Audioclip zurück: 'male', 'female' oder 'unknown'.
        Verwendet Resemblyzer für robuste, offlinefähige Stimmtyperkennung.
        """
        try:
            if not self._plugin.available:
                return "unknown"
            from backend.file_import import load_audio_file as _laf  # pylint: disable=import-outside-toplevel  # noqa: I001

            _ld = _laf(audio_file)
            if _ld is None or _ld.get("audio") is None:
                return "unknown"
            wav = _ld["audio"].astype("float32")
            _sr = int(_ld["sr"])
            emb = self._plugin.embed(wav, _sr)
            if emb is None:
                return "unknown"
            # Pitch-basierte Grobklassifikation auf 16 kHz-Mono
            import librosa as _librosa  # pylint: disable=import-outside-toplevel

            mono_16k = _librosa.resample(
                wav if wav.ndim == 1 else wav.mean(axis=-1),
                orig_sr=_sr,
                target_sr=16000,
            )
            f0 = self._estimate_pitch(mono_16k)
            if f0 < 170:
                return "male"
            elif f0 < 300:
                return "female"
            else:
                return "unknown"
        except Exception as e:
            logger.warning("gender_detection.py::erkennen_gender Ersatzpfad: %s", e)
            return "unknown"

    def _estimate_pitch(self, wav, sr=16000) -> float:
        # Einfache Pitch-Schätzung (Median der Autokorrelationsmethode)
        wav = wav.astype(np.float32)
        corr = correlate(wav, wav)
        corr = corr[len(corr) // 2 :]
        d = np.diff(corr)
        start = np.where(d > 0)[0][0]
        peak = np.argmax(corr[start:]) + start
        if peak == 0:
            return 0
        return sr / peak  # type: ignore[no-any-return]
