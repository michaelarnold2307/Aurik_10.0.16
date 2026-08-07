"""
Kontextanalyse-Modul für AURIK: Erkennt automatisch Genre, Medium, Instrumentierung,
Sprachanteil und Produktionskontext aus Audiodaten.
SOTA-Architektur, modular und erweiterbar.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    def __init__(self, sr: int = 48000):
        self.sr = sr
        self._medium_detector = None  # lazy init

    def _get_medium_detector(self):
        """Lazy-Init des MediumDetectors (portiert aus backend.context_analysis)."""
        if self._medium_detector is None:
            try:
                from backend.core.forensics.medium_detector import MediumDetector  # type: ignore[import]

                self._medium_detector = MediumDetector()  # type: ignore[assignment]
            except ImportError:
                self._medium_detector = False  # type: ignore[assignment]  # dauerhaft deaktiviert
        return self._medium_detector if self._medium_detector is not False else None

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def analyze_with_medium(
        self,
        features: dict[str, Any] | None,
        user_profile: dict[str, Any] | None = None,
        reference_audio=None,
        detected_medium: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Erweiterte Analyse mit Medium-Detection-Integration
        (portiert aus backend.context_analysis, §6.7.1).

        Args:
            features: Feature-Dict aus vorangegangener Analyse oder None.
            user_profile: Optionales Nutzer-Profil-Dict.
            reference_audio: Tuple (audio_np, sr) oder None.
            detected_medium: Bereits erkanntes Medium-Result-Dict oder None.

        Returns:
            Erweiterter Kontext-Dict mit Medium-Detection-Feldern.
        """
        context: dict[str, Any] = {}

        # Medium-Detection
        if detected_medium is not None:
            context["detected_medium"] = detected_medium.get("type", "unknown")
            context["detected_medium_chain"] = detected_medium.get("chain", "unknown")
            context["medium_confidence"] = detected_medium.get("confidence", 0.0)
            context["medium_indicators"] = detected_medium.get("indicators", [])
            context["is_multi_generation"] = detected_medium.get("is_multi_generation", False)
            context["medium_full_result"] = detected_medium
        elif reference_audio is not None and isinstance(reference_audio, tuple) and len(reference_audio) == 2:
            detector = self._get_medium_detector()
            if detector is not None:
                audio_ref, sr_ref = reference_audio
                detected = detector.detect(audio_ref, sr_ref)
                context["detected_medium"] = detected.get("type", "unknown")
                context["detected_medium_chain"] = detected.get("chain", "unknown")
                context["medium_confidence"] = detected.get("confidence", 0.0)
                context["medium_indicators"] = detected.get("indicators", [])
                context["is_multi_generation"] = detected.get("is_multi_generation", False)
                context["medium_full_result"] = detected
            else:
                context["detected_medium"] = "unknown"
                context["detected_medium_chain"] = "unknown"
                context["medium_confidence"] = 0.0
        else:
            context["detected_medium"] = "unknown"
            context["detected_medium_chain"] = "unknown"
            context["medium_confidence"] = 0.0

        # Integration mit bestehender analyze() — Spektral-Features hinzufügen
        if reference_audio is not None and isinstance(reference_audio, tuple):
            audio_arr, _ = reference_audio
            spectral = self.analyze(np.asarray(audio_arr, dtype=np.float64))
            context.update(spectral)

        if features:
            context.setdefault("genre_hint", features.get("genre_hint", context.get("genre", "Unbekannt")))
            context["dynamic_level"] = "high" if features.get("rms", 0) > 0.2 else "low"
            context["transient_rich"] = features.get("transients", 0) > 10
            context["harmonicity"] = features.get("harmonicity", 0)
            context["sample_rate"] = features.get("sr")
            context["artefact_risk"] = features.get("artefact_score", 0) > 0.1
            context["channels"] = features.get("channels")

        if user_profile:
            context["user_level"] = user_profile.get("level", "default")
            context["user_goals"] = user_profile.get("goals", [])

        return context

    def analyze(self, audio: np.ndarray) -> dict[str, Any]:
        """
        Analysiert das Audiosignal und gibt einen Feature-Dict zurück.

        Extrahierte Features (scipy-only, kein ML):
          - duration_sec, rms, dynamic_range_db
          - zero_crossing_rate (normiert auf 1/s)
          - spectral_centroid_hz (gewichteter Mittelwert der Frequenzachse)
          - spectral_flatness  (Wiener-Entropie: geometr./arithm. Mittel der Power)
          - spectral_rolloff_hz (85%-Energie-Schwelle)
          - tempo_bpm          (Energie-Autokorrelation, grob)
          - is_speech, genre, instrumentation, production_context
        """
        audio = np.asarray(audio, dtype=np.float64)
        sr = self.sr
        features: dict[str, Any] = {}

        # --- Zeitbasis ---
        features["duration_sec"] = float(len(audio) / sr)

        # --- RMS & Dynamikbereich ---
        rms = float(np.sqrt(np.mean(audio**2)))
        features["rms"] = rms
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 1e-8
        features["dynamic_range_db"] = float(20.0 * np.log10(max(peak / max(rms, 1e-8), 1e-8)))

        # --- Zero-Crossing-Rate (Samples/Zeit, normiert auf 1/s) ---
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0 * sr)
        features["zero_crossing_rate"] = zcr

        # --- FFT-basierte Spektral-Features ---
        fft_len = min(len(audio), 8192)
        if fft_len > 0:
            freqs = np.fft.rfftfreq(fft_len, d=1.0 / sr)
            power = np.abs(np.fft.rfft(audio[:fft_len])) ** 2
            total_power = float(np.sum(power))

            # Spektrales Zentroid (Hz) – gewichteter Mittelwert
            centroid = float(np.sum(freqs * power) / total_power) if total_power > 1e-30 else 0.0
            features["spectral_centroid_hz"] = centroid

            # Spektrale Flachheit (Wiener-Entropie) – 0=Ton, 1=Rauschen
            log_mean = float(np.mean(np.log(power + 1e-30)))
            arith_mean = float(np.mean(power))
            flatness = float(np.exp(log_mean) / max(arith_mean, 1e-30))
            features["spectral_flatness"] = float(np.clip(flatness, 0.0, 1.0))

            # Rolloff: Frequenz bei 85% kumulativer Energie
            cumpower = np.cumsum(power)
            if total_power > 1e-30:
                rolloff_idx = np.searchsorted(cumpower, 0.85 * total_power)
                features["spectral_rolloff_hz"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])  # type: ignore[call-overload]
            else:
                features["spectral_rolloff_hz"] = 0.0
        else:
            centroid = 0.0
            features["spectral_centroid_hz"] = 0.0
            features["spectral_flatness"] = 0.5
            features["spectral_rolloff_hz"] = 0.0

        # --- Tempo-Schätzung via Energie-Autokorrelation ---
        features["tempo_bpm"] = self._estimate_tempo(audio, sr)

        # --- Sprach-Heuristik ---
        # Sprache: ZCR 50–300 Hz, Centroid 500–3000 Hz, geringe Dynamik
        is_speech = 50 < zcr < 400 and 500 < centroid < 3500 and features["dynamic_range_db"] < 30
        features["is_speech"] = bool(is_speech)

        # --- Genre-Klassifikation (regelbasiert) ---
        features["genre"] = self._classify_genre(
            centroid=centroid,
            bpm=features["tempo_bpm"],
            flatness=features["spectral_flatness"],
            zcr=zcr,
            rms=rms,
            rolloff=features["spectral_rolloff_hz"],
        )

        # --- Instrumentierung ---
        if centroid > 6000:
            features["instrumentation"] = "Electronic"
        elif centroid > 3000:
            features["instrumentation"] = "Mixed/Band"
        elif centroid > 1500:
            features["instrumentation"] = "Acoustic/Mixed"
        else:
            features["instrumentation"] = "Acoustic/Orchestral"

        # --- Produktionskontext ---
        if rms > 0.15:
            features["production_context"] = "Studio (mastered)"
        elif rms > 0.05:
            features["production_context"] = "Studio"
        else:
            features["production_context"] = "Live/Field"

        return features

    # ------------------------------------------------------------------
    # Interne Helfer
    # ------------------------------------------------------------------

    def _estimate_tempo(self, audio: np.ndarray, sr: int) -> float:
        """
        Grobe BPM-Schätzung via Onset-Energie-Autokorrelation.
        Zuverlässig im Bereich 60–200 BPM.
        """
        try:
            # RMS-Energie in 10-ms-Frames
            frame = max(1, int(sr * 0.01))
            n_frames = len(audio) // frame
            if n_frames < 20:
                return 0.0
            energy = np.array([np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2)) for i in range(n_frames)])
            # Onset-Stärke: positive Energie-Differenz
            onset = np.maximum(np.diff(energy), 0.0)
            if len(onset) < 20:
                return 0.0
            # Autokorrelation — FFT-based O(N log N)
            from backend.core.core_utils import fft_autocorr

            ac = fft_autocorr(onset)
            # BPM = 60 / (lag * frame_sec)
            frame_sec = frame / sr
            min_lag = int(60.0 / (200.0 * frame_sec))  # 200 BPM
            max_lag = int(60.0 / (60.0 * frame_sec))  # 60 BPM
            min_lag = max(min_lag, 1)
            max_lag = min(max_lag, len(ac) - 1)
            if min_lag >= max_lag:
                return 0.0
            best_lag = np.argmax(ac[min_lag:max_lag]) + min_lag
            bpm = 60.0 / (best_lag * frame_sec)
            return float(np.clip(bpm, 40.0, 250.0))
        except Exception as e:
            logger.warning("context_Analyse.py::_estimate_tempo Ersatzpfad: %s", e)
            return 0.0

    def _classify_genre(
        self,
        centroid: float,
        bpm: float,
        flatness: float,
        zcr: float,
        rms: float,
        rolloff: float,
    ) -> str:
        """Regelbasierte Genre-Klassifikation aus spektralen Features."""
        # Electronic: hoher Centroid, hohe Flatness, schnelles Tempo
        if centroid > 5000 and bpm > 110:
            return "Electronic/Dance"
        # Metal/Rock: hohe ZCR, hoher Centroid, lautes Signal
        if zcr > 300 and centroid > 3000 and rms > 0.1:
            return "Rock/Metal"
        # Hip-Hop/R&B: tiefer Centroid, gemäßigtes Tempo
        if centroid < 2000 and 70 < bpm < 110 and rms > 0.05:
            return "Hip-Hop/R&B"
        # Jazz: geringer Centroid, variabler Tempo
        if centroid < 3000 and bpm < 140 and flatness < 0.3 and rms < 0.15:
            return "Jazz/Blues"
        # Klassik: geringer Centroid, hohe Dynamik
        if centroid < 4000 and rms < 0.08 and flatness < 0.2:
            return "Classical/Orchestral"
        # Pop: mittlerer Centroid, mittleres Tempo
        if 2000 < centroid < 6000 and 90 < bpm < 140:
            return "Pop"
        # Fallback nach Centroid
        if centroid > 4000:
            return "Electronic/Pop"
        elif centroid > 2000:
            return "Rock/Indie"
        else:
            return "Classical/Jazz"
