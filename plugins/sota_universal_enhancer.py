import logging
from typing import Literal

import numpy as np

_log = logging.getLogger(__name__)

# §4.4: MP-SENet 2023 ersetzt FullSubNetPlusPlugin + DCCRNPlugin (beide entfernt, verboten)
try:
    from plugins.mp_senet_plugin import MpSenetPlugin
except ImportError as e:  # pragma: no cover
    MpSenetPlugin = None  # type: ignore[misc,assignment]
    _log.warning("MpSenetPlugin nicht verfügbar: %s", e)

try:
    from plugins.mdx23c_plugin import MDX23CPlugin
except ImportError as e:  # pragma: no cover
    MDX23CPlugin = None  # type: ignore[misc,assignment]
    _log.warning("MDX23CPlugin nicht verfügbar: %s", e)

try:
    from plugins.wpe_plugin import WpePlugin
except ImportError as e:  # pragma: no cover
    WpePlugin = None  # type: ignore[misc,assignment]
    _log.warning("WpePlugin nicht verfügbar: %s", e)


class SOTAUniversalEnhancer:
    """
    Wählt und nutzt automatisch das optimale SOTA-Modell für Sprache, Musik oder gemischte Inhalte.
    """

    def __init__(self, mode: Literal["auto", "speech", "music", "mix"] = "auto"):
        self.mode = mode
        # §4.4: MP-SENet 2023 übernimmt Sprach-/Fallback-Enhancement (ersetzt FullSubNet+/DCCRN)
        self.speech_model = MpSenetPlugin() if MpSenetPlugin else None  # type: ignore[truthy-function]
        self.music_model = MDX23CPlugin() if MDX23CPlugin else None  # type: ignore[truthy-function]
        self.mix_model = WpePlugin() if WpePlugin else None  # type: ignore[truthy-function]
        self.fallback_model = MpSenetPlugin() if MpSenetPlugin else None  # type: ignore[truthy-function]

    def detect_type(self, audio: np.ndarray, sr: int) -> str:
        # Platzhalter: In der Praxis sollte hier ein KI- oder Heuristik-basierter Musik/Sprache/Mix-Detektor stehen
        # Für Demo: Wenn Varianz > 1.5 -> Musik, sonst Sprache
        if np.var(audio) > 1.5:
            return "music"
        return "speech"

    def process(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        typ = self.detect_type(audio, sr) if self.mode == "auto" else self.mode
        if typ == "speech" and self.speech_model:
            # §4.4: MP-SENet 2023 — enhance(audio, sr) → MpSenetResult
            result = self.speech_model.enhance(audio, sr)
            enhanced = np.nan_to_num(
                np.asarray(result.audio if hasattr(result, "audio") else result, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            return np.clip(enhanced, -1.0, 1.0)  # type: ignore[no-any-return]
        elif typ == "music" and self.music_model:
            return self.music_model.process(audio, sr)
        elif typ == "mix" and self.mix_model:
            # WpePlugin: enhance(audio, sr) → ndarray
            result = self.mix_model.enhance(audio, sr)  # type: ignore[assignment]
            return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)  # type: ignore[call-overload, no-any-return]
        elif self.fallback_model:
            result = self.fallback_model.enhance(audio, sr)
            out = result.audio if hasattr(result, "audio") else result
            return np.clip(np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)  # type: ignore[no-any-return]
        else:
            raise RuntimeError("Kein passendes SOTA-Modell für diesen Signaltyp verfügbar.")
