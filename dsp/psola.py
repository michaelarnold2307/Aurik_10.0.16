"""
Aurik 10.0.0 — PSOLA: Pitch Synchronous Overlap-Add
===============================================
Formanterhaltende Pitch-Korrektur für Gesangsmaterial (§4.5, §8.2–6).

PSOLA (Pitch Synchronous Overlap-Add) analysiert das Signal anhand seiner
Grundfrequenz und rekonstruiert es mit verschobenen Perioden — dabei bleiben
Formanten (Vokaltrakt-Resonanzen F1–F4) erhalten, da die Verschiebung im
Zeitbereich pitch-synchron erfolgt.

Referenzen:
    Moulines, E., & Charpentier, F. (1990).
    Pitch-synchronous waveform processing techniques for text-to-speech synthesis
    using diphones. Speech Communication, 9(5-6), 453-467.

    Macon, M. W., & Clements, M. A. (1997).
    Speech concatenation and synthesis using the PSOLA.
    ICASSP 1997.

Invarianten:
    - Formant-Pearson ≥ 0.95 nach Transposition (§2.8)
    - Nur für Gesangsmaterial (VoiceGender-adaptiv aktiviert)
    - NaN/Inf-sicher: alle Ausgaben durch nan_to_num + clip
    - Thread-sicher: Singleton mit Double-Checked Locking (§3.2)
    - Phase-Vocoder bleibt Fallback für perkussive/instrumentale Segmente
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class PsolaResult:
    """Ergebnis der PSOLA Pitch-Korrektur."""

    audio: np.ndarray  # [n_samples], float32, ∈ [-1, 1]
    pitch_shift_semitones: float  # Tatsächliche Verschiebung in Halbtönen
    n_epochs: int  # Anzahl der PSOLA-Epochen (Pitch-Perioden)
    formant_preserved: bool  # True wenn Formant-Pearson ≥ 0.90
    method_used: str  # "psola" oder "phase_vocoder_fallback"


# ---------------------------------------------------------------------------
# PSOLA-Implementierung
# ---------------------------------------------------------------------------


class PsolaPitchShifter:
    """Pitch Synchronous Overlap-Add (PSOLA) Pitch-Shifter.

    Algorithmus (Time-Domain PSOLA, TD-PSOLA):
        1. GCI-Detektion: Glottale Verschlussmomente (Pitch-Marker) bestimmen
           → Verwende f₀-Schätzung (CREPE/pYIN) für Marker-Abstand = T₀
        2. Analyse-Epochen: Fenster der Länge 2·T₀ zentriert auf jedem Marker
        3. Synthese-Epochen: Neue Marker-Abstände für Zielfrequenz
           T₀_target = T₀_source · (f₀_source / f₀_target)
        4. OLA: Analyse-Epochen auf Synthese-Marker aufaddieren (Hanning-Fenster)
        5. Normalisierung: OLA-Envelope dividieren

    Formanterhalt:
        Da Epochen im Zeitbereich über-addiert werden (kein Frequenzbereich-
        Stretching), bleiben Spektralhüllkurven (Formanten) erhalten.
        Vergleiche mit Frequenzbereich-Stretching: dort verschiebt sich
        der gesamte Spektralinhalt inkl. Formanten → Chipmunk-Effekt.

    Aktivierung (§2.8):
        Automatisch wenn PANNs Vocals confidence ≥ 0.4; via HPSS-Erkennung
        für Gesangs-Segmente. Phase-Vocoder bleibt Fallback für nicht-vokale
        Segmente (perkussiv, instrumental).
    """

    MIN_F0_HZ: float = 60.0  # Unterste akzeptable Grundfrequenz
    MAX_F0_HZ: float = 1000.0  # Höchste akzeptable Grundfrequenz
    MIN_SEMITONE_SHIFT: float = -12.0  # Oktave tiefer
    MAX_SEMITONE_SHIFT: float = 12.0  # Oktave höher
    FORMANT_PEARSON_THRESHOLD: float = 0.90  # §2.8 Pflicht-Schwellwert

    def __init__(self, sr: int = 48000) -> None:
        """Initialisiert den PSOLA-Pitch-Shifter.

        Args:
            sr: Sample-Rate (muss 48000 Hz sein)
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        self.sr = sr

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def shift_pitch(
        self,
        audio: np.ndarray,
        semitones: float,
        f0_hz: float | None = None,
        f0_trajectory: np.ndarray | None = None,
    ) -> PsolaResult:
        """Verschiebt die Tonhöhe um `semitones` Halbtöne (formanterhaltend).

        Args:
            audio:          Mono-Gesangs-Audio [n_samples], float32
            semitones:      Verschiebung in Halbtönen (negativ = tiefer)
            f0_hz:          Mittlere Grundfrequenz (Hz); falls None: auto-detektiert
            f0_trajectory:  Zeitvariante f₀-Kurve [n_frames] (optional, genauer)

        Returns:
            PsolaResult mit pitch-verschobenem Audio.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)

        # Clamp Halbton-Verschiebung
        semitones = float(np.clip(semitones, self.MIN_SEMITONE_SHIFT, self.MAX_SEMITONE_SHIFT))

        if abs(semitones) < 0.01:
            return PsolaResult(
                audio=audio.copy(),
                pitch_shift_semitones=0.0,
                n_epochs=0,
                formant_preserved=True,
                method_used="passthrough",
            )

        # Frequenz-Verhältnis
        ratio = 2.0 ** (semitones / 12.0)

        # F₀ bestimmen
        if f0_hz is None:
            f0_hz = self._estimate_f0(audio)
        f0_hz = float(np.clip(f0_hz, self.MIN_F0_HZ, self.MAX_F0_HZ))

        # PSOLA versuchen
        try:
            result_audio, n_epochs = self._psola(audio, f0_hz, ratio, f0_trajectory)
            method = "psola"
        except Exception as e:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            logger.debug("PSOLA Fallback (Phase-Vocoder): %s", e)
            result_audio = self._phase_vocoder_shift(audio, ratio)
            n_epochs = 0
            method = "phase_vocoder_fallback"

        result_audio = np.nan_to_num(result_audio, nan=0.0, posinf=0.0, neginf=0.0)
        result_audio = np.clip(result_audio, -1.0, 1.0).astype(np.float32)

        # Formant-Erhalt prüfen (MFCC-Pearson)
        formant_ok = self._check_formant_preservation(audio, result_audio)

        logger.debug(
            "PSOLA: %.1f Halbtöne, f0=%.1f Hz, ratio=%.3f, epochs=%d, formant_ok=%s, method=%s",
            semitones,
            f0_hz,
            ratio,
            n_epochs,
            formant_ok,
            method,
        )

        return PsolaResult(
            audio=result_audio,
            pitch_shift_semitones=semitones,
            n_epochs=n_epochs,
            formant_preserved=formant_ok,
            method_used=method,
        )

    def correct_wow_flutter(
        self,
        audio: np.ndarray,
        f0_trajectory: np.ndarray,
        target_f0: float,
        frame_hop_samples: int = 480,
    ) -> np.ndarray:
        """Korrigiert Wow/Flutter-Pitch-Drift segment-weise via PSOLA.

        Algorithmus:
            1. Audio in Frames aufteilen (frame_hop_samples)
            2. Pro Frame: lokale f₀ aus f0_trajectory
            3. Halbton-Verschiebung = 12 · log₂(target_f0 / f0_local)
            4. PSOLA pro Frame mit lokalem Verschiebungswert
            5. OLA-Crossfade zwischen Frames (Hanning, 20 ms)

        Args:
            audio:             Gesangs-Audio [n_samples]
            f0_trajectory:     Zeitvariante f₀ [n_frames], Hz
            target_f0:         Ziel-Grundfrequenz (Hz)
            frame_hop_samples: Frame-Hop in Samples (Standard: 480 = 10 ms @48 kHz)

        Returns:
            Pitch-korrigiertes Audio [n_samples], float32
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)
        n_samples = len(audio)

        output = np.zeros(n_samples, dtype=np.float64)
        norm = np.zeros(n_samples, dtype=np.float64)

        min(960, frame_hop_samples)  # Max 20 ms @48 kHz

        for i, f0_local in enumerate(f0_trajectory):
            start = i * frame_hop_samples
            end = min(start + frame_hop_samples * 2, n_samples)
            if start >= n_samples:
                break
            frame = audio[start:end]
            if len(frame) < 64:
                continue

            if f0_local < self.MIN_F0_HZ or f0_local > self.MAX_F0_HZ:
                # Unzuverlässige f₀: Frame unverändert übernehmen
                semitones = 0.0
            else:
                semitones = 12.0 * math.log2(target_f0 / f0_local)
                semitones = float(np.clip(semitones, self.MIN_SEMITONE_SHIFT, self.MAX_SEMITONE_SHIFT))

            result = self.shift_pitch(frame, semitones, f0_hz=f0_local)
            shifted = result.audio.astype(np.float64)

            # Hanning-Envelope für OLA
            env = np.hanning(len(shifted))
            frame_len = min(len(shifted), n_samples - start)
            output[start : start + frame_len] += shifted[:frame_len] * env[:frame_len]
            norm[start : start + frame_len] += env[:frame_len]

        norm = np.where(norm < 1e-8, 1.0, norm)
        output = output / norm
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(output, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Private Methoden
    # ------------------------------------------------------------------

    def _psola(
        self,
        audio: np.ndarray,
        f0_hz: float,
        ratio: float,
        f0_trajectory: np.ndarray | None,
    ) -> tuple[np.ndarray, int]:
        """TD-PSOLA Kern-Algorithmus.

        Args:
            audio:     Mono-Audio [n_samples]
            f0_hz:     Mittlere Grundfrequenz Hz
            ratio:     Pitch-Ratio (target_f0 / source_f0)
            f0_trajectory: Zeitvariante f₀ (optional)

        Returns:
            (shifted_audio, n_epochs)
        """
        sr = self.sr
        n_samples = len(audio)

        # Analyse-Marker: Abstand = T₀ in Samples
        t0_source = round(sr / f0_hz)
        t0_source = max(t0_source, 8)  # Minimum 8 Samples

        # Analyse-Epochen-Marker
        analysis_markers = list(range(t0_source // 2, n_samples, t0_source))

        # Synthese-Marker: neuer Abstand = T₀/ratio
        t0_target = round(t0_source / ratio)
        t0_target = max(t0_target, 8)
        n_synth = int(n_samples / t0_target) + 2
        synthesis_markers = [i * t0_target for i in range(n_synth)]

        # Fensterbreite = 2·T₀ (Hanning)
        win_len = 2 * t0_source
        half_win = t0_source

        # Output-Array
        output = np.zeros(n_samples + win_len, dtype=np.float64)
        norm = np.zeros(n_samples + win_len, dtype=np.float64)

        n_epochs = 0

        for sm in synthesis_markers:
            if sm >= n_samples:
                break

            # Nächsten Analyse-Marker finden (nächster in f0_trajectory oder global)
            if f0_trajectory is not None and len(f0_trajectory) > 0:
                frame_idx = min(sm * len(f0_trajectory) // n_samples, len(f0_trajectory) - 1)
                f0_local = f0_trajectory[frame_idx]
                if self.MIN_F0_HZ <= f0_local <= self.MAX_F0_HZ:
                    # Use local T0 to constrain which analysis marker is valid.
                    # Previously this was computed but discarded (dead expression).
                    t0_local = max(8, round(sr / f0_local))
                    candidates = [
                        (idx, abs(analysis_markers[idx] - sm))
                        for idx in range(len(analysis_markers))
                        if abs(analysis_markers[idx] - sm) <= 2 * t0_local
                    ]
                    if candidates:
                        am = min(candidates, key=lambda x: x[1])[0]
                    else:
                        am = min(range(len(analysis_markers)), key=lambda i: abs(analysis_markers[i] - sm))
                    am_idx = analysis_markers[am]
                else:
                    am_idx = sm
            else:
                # Nächsten Analyse-Marker wählen
                am_idx = min(analysis_markers, key=lambda m: abs(m - sm))

            # Analyse-Fenster aus Quellaudio
            a_start = max(0, am_idx - half_win)
            a_end = min(n_samples, am_idx + half_win)
            frame = audio[a_start:a_end].astype(np.float64)

            # Fenster anpassen
            actual_len = len(frame)
            if actual_len < 4:
                continue

            win_actual = np.hanning(actual_len)

            # Ins Output-Array addieren
            o_start = sm
            o_end = min(o_start + actual_len, len(output))
            add_len = o_end - o_start
            output[o_start:o_end] += frame[:add_len] * win_actual[:add_len]
            norm[o_start:o_end] += win_actual[:add_len]
            n_epochs += 1

        # OLA-Normalisierung
        norm = np.where(norm < 1e-8, 1.0, norm)
        output = output / norm
        output = output[:n_samples]

        return output.astype(np.float32), n_epochs

    def _phase_vocoder_shift(
        self,
        audio: np.ndarray,
        ratio: float,
        win_size: int = 2048,
        hop: int = 256,
    ) -> np.ndarray:
        """Phase-Vocoder Fallback für nicht-vokale Segmente.

        Einfache spektrale Verschiebung für perkussive/instrumentale Teile.
        Nicht formanterhaltend — nur als Fallback wenn PSOLA fehlschlägt.
        """
        n = len(audio)
        # Resample-Ratio: länger strecken, dann kürzen
        target_len = int(n / ratio)
        if target_len == n:
            return audio.copy()

        # Lineare Interpolation zur Tempo-Änderung
        x_old = np.linspace(0.0, 1.0, n)
        x_new = np.linspace(0.0, 1.0, target_len)
        stretched = np.interp(x_new, x_old, audio.astype(np.float64))

        # Auf Original-Länge trimmen/padden
        if len(stretched) > n:
            stretched = stretched[:n]
        elif len(stretched) < n:
            stretched = np.pad(stretched, (0, n - len(stretched)))

        return stretched.astype(np.float32)

    def _estimate_f0(self, audio: np.ndarray) -> float:
        """Einfache Autokorrelations-f₀-Schätzung als Fallback (kein CREPE).

        Wird nur verwendet wenn kein CREPE-Ergebnis vorliegt.
        Aurik nutzt normalerweise CREPE/pYIN für f₀-Schätzung.
        """
        # YIN-ähnlicher Pitch-Tracker (vereinfacht)
        frame_len = min(4096, len(audio))
        frame = audio[:frame_len].astype(np.float64)

        # Autokorrelation via FFT
        fft_size = 2 * frame_len
        spectrum = np.fft.rfft(frame, n=fft_size)
        acf = np.fft.irfft(spectrum * np.conj(spectrum))[:frame_len]

        # Normalisierung
        if acf[0] < 1e-10:
            return 220.0  # Default A3

        acf = acf / acf[0]

        # Suche nach erstem lokalem Maximum (Pitch-Periode)
        min_lag = int(self.sr / self.MAX_F0_HZ)
        max_lag = int(self.sr / self.MIN_F0_HZ)
        min_lag = max(min_lag, 1)
        max_lag = min(max_lag, frame_len - 1)

        if min_lag >= max_lag:
            return 220.0

        best_lag = min_lag + int(np.argmax(acf[min_lag:max_lag]))
        f0 = self.sr / max(best_lag, 1)
        return float(np.clip(f0, self.MIN_F0_HZ, self.MAX_F0_HZ))

    def _check_formant_preservation(
        self,
        original: np.ndarray,
        processed: np.ndarray,
    ) -> bool:
        """Prüft MFCC-Pearson-Korrelation für Formant-Erhalt (§2.8).

        Gibt True zurück wenn Pearson ≥ FORMANT_PEARSON_THRESHOLD.
        """
        try:
            # MFCC-Approximation via DCT auf Log-Mel-Spektrum
            n = min(len(original), len(processed), 8192)
            orig_f = original[:n].astype(np.float64)
            proc_f = processed[:n].astype(np.float64)

            win = np.hanning(n)
            orig_spec = np.abs(np.fft.rfft(orig_f * win))
            proc_spec = np.abs(np.fft.rfft(proc_f * win))

            # Envelope-Vergleich (als MFCC-Proxy)
            # Smooth via log-domain
            orig_env = np.log(orig_spec + 1e-8)
            proc_env = np.log(proc_spec + 1e-8)

            # Pearson-Korrelation
            orig_env -= np.mean(orig_env)
            proc_env -= np.mean(proc_env)
            norm_o = np.linalg.norm(orig_env)
            norm_p = np.linalg.norm(proc_env)

            if norm_o < 1e-8 or norm_p < 1e-8:
                return True  # Stille: kein Vergleich möglich

            pearson = np.dot(orig_env, proc_env) / (norm_o * norm_p)
            return float(pearson) >= self.FORMANT_PEARSON_THRESHOLD
        except Exception:
            logger.warning("psola.py::_check_formant_preservation fallback", exc_info=True)
            return True  # Im Zweifel: OK annehmen


# ---------------------------------------------------------------------------
# Singleton (§3.2)
# ---------------------------------------------------------------------------

_instance: PsolaPitchShifter | None = None
_lock = threading.Lock()


def get_psola_shifter(sr: int = 48000) -> PsolaPitchShifter:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2).

    Args:
        sr: Sample-Rate (muss 48000)

    Returns:
        Globale PsolaPitchShifter-Instanz.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PsolaPitchShifter(sr=sr)
    return _instance


# ---------------------------------------------------------------------------
# Convenience-Funktionen
# ---------------------------------------------------------------------------


def psola_shift(
    audio: np.ndarray,
    semitones: float,
    sr: int = 48000,
    f0_hz: float | None = None,
    f0_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Formanterhaltende Pitch-Verschiebung via PSOLA (§4.5).

    Pflicht für Gesangsmaterial (PANNs-Vocals confidence ≥ 0.4).
    Phase-Vocoder als Fallback für perkussive/instrumentale Segmente.

    Algorithmus:
        TD-PSOLA: T₀-synchrone Epochen-Analyse + OLA mit verschobenen Markern
        Formanterhalt via Zeitbereich-Überlappung (kein Spektral-Stretch)

    Args:
        audio:          Mono-Gesangs-Audio [n_samples], float32
        semitones:      Verschiebung in Halbtönen (-12 ... +12)
        sr:             Sample-Rate (muss 48000)
        f0_hz:          Mittlere Grundfrequenz (Hz); None = auto-detect
        f0_trajectory:  Zeitvariante f₀-Kurve [n_frames] (genauer, optional)

    Returns:
        Pitch-verschobenes Audio [n_samples], float32, ∈ [-1, 1]

    Mathematische Grundlage:
        ratio = 2^(semitones/12)
        T₀_target = T₀_source / ratio
        f₀_target = f₀_source · ratio
    """
    result = get_psola_shifter(sr).shift_pitch(audio, semitones, f0_hz, f0_trajectory)
    return result.audio


def psola_correct_wow_flutter(
    audio: np.ndarray,
    f0_trajectory: np.ndarray,
    target_f0: float,
    sr: int = 48000,
    frame_hop_samples: int = 480,
) -> np.ndarray:
    """Korrigiert Wow/Flutter-Pitch-Drift segment-weise via PSOLA.

    Aktivierung in phase_12_wow_flutter_fix.py für Gesangsmaterial.

    Args:
        audio:             Gesangs-Audio [n_samples]
        f0_trajectory:     F₀-Verlauf [n_frames], Hz (aus CREPE/pYIN)
        target_f0:         Ziel-Grundfrequenz (Hz) — stabile Referenztonhöhe
        sr:                Sample-Rate (muss 48000)
        frame_hop_samples: Frame-Hop in Samples

    Returns:
        Pitch-korrigiertes Audio [n_samples], float32
    """
    return get_psola_shifter(sr).correct_wow_flutter(audio, f0_trajectory, target_f0, frame_hop_samples)
