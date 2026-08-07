"""
StructuralSilenceIsolationProtocol (SSIP) — §2.68 [RELEASE_MUST]
=================================================================

Architekturell isoliert strukturelle Stille-Zonen (Intro, Outro, Fade-In/-Out)
von generativen/Inpainting-Phasen. Verhindert die drei dokumentierten Failure-Modes
aus §2.68a:

  1. Gap-Detektor klassifiziert Stille-Musik-Grenze als Dropout (Failure Mode 1)
  2. Modell-Kontext-Kontamination über Stille-Grenzen (Failure Mode 2)
  3. Silence-Mask-Null-Propagation (Failure Mode 3)

Kanonische Lösung: Isolation VOR der Verarbeitung — kein Post-Processing.
Das Inpainting-Modell sieht ausschliesslich Audio-Segmente, niemals Stille.

Pflicht-Integration: phase_55 + phase_24 + jede zukünftige generative Phase.

Spec: 02_pipeline_architecture.md §2.68 (v10.0.0)
VERBOTEN-Regeln: V14, V15, V16, V17, V18
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: StructuralSilenceIsolator | None = None
_lock = threading.Lock()


def get_structural_silence_isolator() -> StructuralSilenceIsolator:
    """Singleton-Getter (thread-safe, double-checked locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StructuralSilenceIsolator()
    return _instance


# ---------------------------------------------------------------------------
# Helper: standalone _get_structural_silence_zones for phase integration
# ---------------------------------------------------------------------------


def _get_structural_silence_zones(
    kwargs: dict,
    audio_original: np.ndarray,
    sr: int,
    material_key: str = "unknown",
) -> list[tuple[int, int]]:
    """Liefert immer gültige Stille-Zonen — niemals None (§2.68d Null-Propagation-Guard).

    Suchreihenfolge:
    1. kwargs["structural_silence_zones"]
    2. kwargs["restoration_context"]["structural_silence_zones"]
    3. Eigenständige Berechnung aus audio_original (Fallback, non-blocking)

    Returns:
        Liste aus (start_sample, end_sample)-Tupeln — leere Liste wenn keine Zone.
    """
    # Versuch 1: direkt in kwargs
    zones = kwargs.get("structural_silence_zones")
    if zones is not None:
        return list(zones)

    # Versuch 2: aus restoration_context
    ctx = kwargs.get("restoration_context") or {}
    if isinstance(ctx, dict):
        zones = ctx.get("structural_silence_zones")
        if zones is not None:
            return list(zones)

    # Versuch 3: eigenständige Berechnung (Fallback)
    logger.debug(
        "SSIP: structural_silence_zones nicht in context — "
        "eigenständige Berechnung aus Originalsignal_audio (Ersatzpfad, nicht blockierend)"
    )
    isolator = get_structural_silence_isolator()
    return isolator.detect_structural_silence_zones(audio_original, sr, material_key)


def _run_inpainting_with_ssip(
    audio: np.ndarray,
    sr: int,
    silence_zones: list[tuple[int, int]],
    inpainting_fn: Callable,
    **kwargs,
) -> np.ndarray:
    """Wraps jede Inpainting-Funktion mit dem vollständigen SSIP (§2.68c).

    Layer 1: Isolation — Modell sieht keine Stille
    Layer 2: Reassembly — Stille-Zonen aus Original
    Layer 3: Post-Audit — letztes Sicherheitsnetz (Hard-Reset)

    Pflicht-Pattern für phase_55 und phase_24.
    """
    _isolator = get_structural_silence_isolator()

    # Layer 1: Splitting — Modell erhält keine Stille als Kontext
    segments = _isolator.split_at_silence_boundaries(audio, sr, silence_zones)

    processed_segments: list[dict] = []
    for seg in segments:
        if seg["type"] == "silence":
            # Stille: immer Original, niemals durch Inpainting-Modell verarbeiten
            processed_segments.append(seg)
        else:
            # Audio-Segment: Inpainting anwenden
            try:
                processed_data = inpainting_fn(seg["data"], sr, **kwargs)
                processed_segments.append({**seg, "data": processed_data})
            except Exception as _exc:
                logger.warning(
                    "SSIP: inpainting_fn fehlgeschlagen für Segment [%d:%d] — Passthrough: %s",
                    seg["start"],
                    seg["end"],
                    _exc,
                )
                processed_segments.append(seg)  # Passthrough (Original)

    # Layer 2: Reassembly — Stille-Segmente nutzen Original-Samples
    n_total = audio.shape[-1] if audio.ndim > 1 else len(audio)
    result = _isolator.reassemble_from_segments(
        processed_segments,
        original_silence_audio=audio,
        n_samples_total=n_total,
    )

    # Layer 3: Post-Audit — Hard-Reset als letztes Sicherheitsnetz
    result = _isolator.post_inpainting_silence_audit(
        audio_before_inpainting=audio,
        audio_after_inpainting=result,
        silence_zones=silence_zones,
        sr=sr,
    )

    return result


# ---------------------------------------------------------------------------
# StructuralSilenceIsolator
# ---------------------------------------------------------------------------


class StructuralSilenceIsolator:
    """
    Architekturell trennt Stille von Audio-Inhalt vor jeder generativen Verarbeitung.

    NICHT: Post-Processing-Maske (greift nicht bei Kontextkontamination).
    SONDERN: Vorverarbeitungs-Isolation — das Modell sieht nur Audio-Segmente,
             nie Stille, nie Stille-Musik-Grenzen.

    Spec: §2.68b (v10.0.0)
    """

    # Silence-Schwellen — material-adaptiv
    SILENCE_THRESHOLDS_DBFS: dict[str, float] = {
        "shellac": -22.0,
        "wax_cylinder": -22.0,
        "vinyl": -33.0,
        "lacquer_disc": -33.0,
        "cassette": -43.0,
        "tape": -43.0,
        "reel_tape": -50.0,
        "cd_digital": -57.0,
        "dat": -60.0,
        "flac": -60.0,
        "streaming": -57.0,
        "unknown": -45.0,
    }

    # Sicherheitspuffer: Gap darf nicht näher als CONTEXT_GUARD_MS an Stille sein
    CONTEXT_GUARD_MS: float = 1500.0  # 1.5 s — typisches AR-Kontextfenster

    # Mindest-Stille-Dauer für strukturelle Klassifikation
    _MIN_SILENCE_DURATION_MS: float = 300.0

    # Analyse-Fenster für RMS-Berechnung
    _ANALYSIS_WINDOW_MS: float = 200.0

    def _get_silence_threshold(self, material_key: str) -> float:
        """Gibt material-adaptiven Stille-Schwellwert in dBFS zurück."""
        key = str(material_key).lower().strip()
        # Exakter Match
        if key in self.SILENCE_THRESHOLDS_DBFS:
            return self.SILENCE_THRESHOLDS_DBFS[key]
        # Teilstring-Match
        for k, v in self.SILENCE_THRESHOLDS_DBFS.items():
            if k in key:
                return v
        return self.SILENCE_THRESHOLDS_DBFS["unknown"]

    def detect_structural_silence_zones(
        self,
        audio_original: np.ndarray,
        sr: int,
        material_key: str = "unknown",
    ) -> list[tuple[int, int]]:
        """Erkennt strukturelle Stille aus dem ORIGINAL-Audio (vor allen Phasen).

        Bedingungen für strukturelle Stille (UND-Verknüpfung):
        1. RMS < SILENCE_THRESHOLDS_DBFS[material_key] (200 ms-Fenster)
        2. Dauer >= 300 ms
        3. Zone liegt am ANFANG oder ENDE des Signals, ODER dauert > 1000 ms

        WICHTIG: Immer aus ORIGINAL-Audio berechnen — niemals aus verarbeitetem Audio.

        Args:
            audio_original: Original-Audio (vor Pipeline) als (N,) oder (2, N) oder (N, 2).
            sr:             Sample-Rate in Hz.
            material_key:   Material-Schlüssel für adaptive Schwelle.

        Returns:
            Liste aus (start_sample, end_sample)-Tupeln. Leere Liste wenn keine Zone.
        """
        try:
            audio_arr = np.asarray(audio_original, dtype=np.float32)
            # Mono-Mix für Analyse
            if audio_arr.ndim == 2:
                if audio_arr.shape[0] == 2 and audio_arr.shape[1] > 2:
                    # (channels, samples)
                    mono = np.mean(audio_arr, axis=0)
                else:
                    # (samples, channels)
                    mono = np.mean(audio_arr, axis=1)
            else:
                mono = audio_arr

            n_samples = len(mono)
            if n_samples == 0:
                return []

            threshold_dbfs = self._get_silence_threshold(material_key)
            threshold_linear = float(10.0 ** (threshold_dbfs / 20.0))

            window_samples = max(1, int(self._ANALYSIS_WINDOW_MS / 1000.0 * sr))
            min_silence_samples = max(1, int(self._MIN_SILENCE_DURATION_MS / 1000.0 * sr))
            long_silence_samples = max(1, int(1000.0 / 1000.0 * sr))  # 1000 ms für Mitte

            # RMS-Berechnung pro Fenster
            n_windows = max(1, (n_samples + window_samples - 1) // window_samples)
            is_silent = np.zeros(n_samples, dtype=bool)

            for i in range(n_windows):
                w_start = i * window_samples
                w_end = min(n_samples, w_start + window_samples)
                rms = float(np.sqrt(np.mean(mono[w_start:w_end] ** 2) + 1e-12))
                if rms < threshold_linear:
                    is_silent[w_start:w_end] = True

            # Kontinuierliche Stille-Regionen finden
            zones: list[tuple[int, int]] = []
            in_silence = False
            silence_start = 0

            for i in range(n_samples + 1):
                currently_silent = is_silent[i] if i < n_samples else False
                if currently_silent and not in_silence:
                    silence_start = i
                    in_silence = True
                elif not currently_silent and in_silence:
                    silence_end = i
                    duration = silence_end - silence_start
                    in_silence = False

                    if duration < min_silence_samples:
                        continue

                    # Bedingung 3: Anfang/Ende ODER > 1000 ms
                    is_at_start = silence_start == 0
                    is_at_end = silence_end == n_samples
                    is_long = duration >= long_silence_samples

                    if is_at_start or is_at_end or is_long:
                        zones.append((silence_start, silence_end))

            logger.debug(
                "SSIP.erkennen_structural_silence_zones: %d Zone(n) (material=%s, Schwelle=%.1f dBFS)",
                len(zones),
                material_key,
                threshold_dbfs,
            )
            return zones

        except Exception as exc:
            logger.warning("SSIP.erkennen_structural_silence_zones fehlgeschlagen (nicht blockierend): %s", exc)
            return []

    def split_at_silence_boundaries(
        self,
        audio: np.ndarray,
        sr: int,
        silence_zones: list[tuple[int, int]],
    ) -> list[dict]:
        """Splittet Audio in Segmente ohne Stille-Zonen.

        Stille-Zonen werden als separate "silence"-Segmente zurückgegeben.
        Jedes "audio"-Segment hat Context-Guard-Abstand zu Stille (CONTEXT_GUARD_MS).

        Args:
            audio:         Audio-Array (N,) oder (2, N) oder (N, 2).
            sr:            Sample-Rate.
            silence_zones: Liste aus (start, end) Stille-Zonen.

        Returns:
            Liste aus Dicts mit keys:
            - "type": "audio" oder "silence"
            - "start": int (Sample-Index im Original)
            - "end": int (Sample-Index im Original)
            - "data": np.ndarray (Segment-Daten)
        """
        try:
            audio_arr = np.asarray(audio, dtype=np.float32)
            if audio_arr.ndim == 2:
                if audio_arr.shape[0] == 2 and audio_arr.shape[1] > 2:
                    n_samples = audio_arr.shape[1]
                else:
                    n_samples = audio_arr.shape[0]
            else:
                n_samples = len(audio_arr)

            if not silence_zones or n_samples == 0:
                return [{"type": "audio", "start": 0, "end": n_samples, "data": audio_arr}]

            # Stille-Zonen sortieren
            sorted_zones = sorted(silence_zones, key=lambda z: z[0])

            guard_samples = max(0, int(self.CONTEXT_GUARD_MS / 1000.0 * sr))

            # Markiere Segmente: silence, context_guard (Teil von silence für Modell), audio
            # Erzeuge Maske: 0=audio, 1=silence_or_guard
            mask = np.zeros(n_samples, dtype=np.int8)
            for zs, ze in sorted_zones:
                zs = max(0, int(zs))
                ze = min(n_samples, int(ze))
                # Stille + Context-Guard vor und nach Stille
                guard_start = max(0, zs - guard_samples)
                guard_end = min(n_samples, ze + guard_samples)
                mask[guard_start:guard_end] = 1
                # Eigentliche Stille markieren (2 = echte Stille, für Typ-Unterscheidung)
                mask[zs:ze] = 2

            # Segmente aus Maske erzeugen
            segments: list[dict] = []
            i = 0
            while i < n_samples:
                seg_type_val = int(mask[i])
                j = i
                while j < n_samples and mask[j] == seg_type_val:
                    j += 1

                if audio_arr.ndim == 2:
                    if audio_arr.shape[0] == 2 and audio_arr.shape[1] > 2:
                        seg_data = audio_arr[:, i:j]
                    else:
                        seg_data = audio_arr[i:j, :]
                else:
                    seg_data = audio_arr[i:j]

                # seg_type_val: 0=audio, 1=guard (auch als silence behandeln), 2=silence
                if seg_type_val == 0:
                    seg_type = "audio"
                else:
                    seg_type = "silence"

                segments.append(
                    {
                        "type": seg_type,
                        "start": i,
                        "end": j,
                        "data": seg_data,
                    }
                )
                i = j

            return segments

        except Exception as exc:
            logger.warning("SSIP.split_at_silence_boundaries fehlgeschlagen — Passthrough: %s", exc)
            audio_arr = np.asarray(audio, dtype=np.float32)
            n_samples = audio_arr.shape[-1] if audio_arr.ndim > 1 else len(audio_arr)
            return [{"type": "audio", "start": 0, "end": n_samples, "data": audio_arr}]

    def reassemble_from_segments(
        self,
        segments: list[dict],
        original_silence_audio: np.ndarray,
        n_samples_total: int,
    ) -> np.ndarray:
        """Fügt verarbeitete Audio-Segmente und ORIGINAL-Stille-Daten zusammen.

        Stille-Segmente: ORIGINAL-Daten — niemals verarbeitet.
        Audio-Segmente: verarbeitete Daten aus dem Inpainting.

        HARD RULE: In reassembliertem Output an Stille-Zonen-Positionen sind
        AUSSCHLIESSLICH die Original-Samples zulässig.
        """
        try:
            orig = np.asarray(original_silence_audio, dtype=np.float32)
            is_stereo_channels_first = orig.ndim == 2 and orig.shape[0] == 2 and orig.shape[1] > 2
            is_stereo_samples_first = orig.ndim == 2 and not is_stereo_channels_first

            if is_stereo_channels_first:
                result = np.zeros((2, n_samples_total), dtype=np.float32)
            elif is_stereo_samples_first:
                n_ch = orig.shape[1] if orig.ndim == 2 else 1
                result = np.zeros((n_samples_total, n_ch), dtype=np.float32)
            else:
                result = np.zeros(n_samples_total, dtype=np.float32)

            for seg in segments:
                s = int(seg["start"])
                e = int(seg["end"])
                seg_len = e - s
                if seg_len <= 0:
                    continue

                if seg["type"] == "silence":
                    # HARD RULE: Original-Samples für Stille
                    if is_stereo_channels_first:
                        result[:, s:e] = orig[:, s : min(e, orig.shape[1])]
                    elif is_stereo_samples_first:
                        result[s:e, :] = orig[s : min(e, orig.shape[0]), :]
                    else:
                        result[s:e] = orig[s : min(e, len(orig))]
                else:
                    # Audio-Segment: verarbeitete Daten eintragen
                    data = np.asarray(seg.get("data", np.zeros(seg_len, dtype=np.float32)), dtype=np.float32)
                    if is_stereo_channels_first:
                        if data.ndim == 2 and data.shape[0] == 2:
                            actual_len = min(seg_len, data.shape[1], n_samples_total - s)
                            result[:, s : s + actual_len] = data[:, :actual_len]
                        elif data.ndim == 1:
                            actual_len = min(seg_len, len(data), n_samples_total - s)
                            result[0, s : s + actual_len] = data[:actual_len]
                            result[1, s : s + actual_len] = data[:actual_len]
                    elif is_stereo_samples_first:
                        if data.ndim == 2:
                            actual_len = min(seg_len, data.shape[0], n_samples_total - s)
                            result[s : s + actual_len, :] = data[:actual_len, :]
                        else:
                            actual_len = min(seg_len, len(data), n_samples_total - s)
                            n_ch = result.shape[1]
                            result[s : s + actual_len, :] = data[:actual_len, np.newaxis] * np.ones((1, n_ch))
                    else:
                        if data.ndim > 1:
                            data = data.ravel()
                        actual_len = min(seg_len, len(data), n_samples_total - s)
                        result[s : s + actual_len] = data[:actual_len]

            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]

        except Exception as exc:
            logger.warning("SSIP.reassemble_from_segments fehlgeschlagen — Passthrough: %s", exc)
            return np.asarray(original_silence_audio, dtype=np.float32)  # type: ignore[no-any-return]

    def post_inpainting_silence_audit(
        self,
        audio_before_inpainting: np.ndarray,
        audio_after_inpainting: np.ndarray,
        silence_zones: list[tuple[int, int]],
        sr: int,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """Post-Inpainting-Sicherheitsnetz (letzter Layer).

        Für jede Stille-Zone:
          1. Messe Energie in audio_after_inpainting
          2. Messe Energie in audio_before_inpainting
          3. Wenn after > before + 3 dB: HARD RESET — Original-Samples einsetzen
             (kein Clipping — Hard-Reset reproduziert das Original exakt)

        VERBOTEN: Clamp/Clip als Alternative zu Hard-Reset (§2.68d, V17).

        Args:
            audio_before_inpainting: Pre-Inpainting-Audio als Referenz.
            audio_after_inpainting:  Post-Inpainting-Audio zum Prüfen.
            silence_zones:           Stille-Zonen als (start, end)-Liste.
            sr:                      Sample-Rate (nicht direkt verwendet, für API-Konsistenz).

        Returns:
            Audio mit zurückgesetzten Stille-Zonen wenn nötig.
        """
        if not silence_zones:
            return np.asarray(audio_after_inpainting, dtype=np.float32)  # type: ignore[no-any-return]

        try:
            before = np.asarray(audio_before_inpainting, dtype=np.float32)
            after = np.asarray(audio_after_inpainting, dtype=np.float32)
            result = after.copy()

            is_stereo_channels_first = after.ndim == 2 and after.shape[0] == 2 and after.shape[1] > 2
            is_stereo_samples_first = after.ndim == 2 and not is_stereo_channels_first

            n_resets = 0
            for zs, ze in silence_zones:
                zs = max(0, int(zs))
                if is_stereo_channels_first:
                    ze = min(int(ze), after.shape[1])
                    after_zone = after[:, zs:ze]
                    before_zone = before[:, zs:ze] if before.ndim == 2 else before[np.newaxis, zs:ze]
                elif is_stereo_samples_first:
                    ze = min(int(ze), after.shape[0])
                    after_zone = after[zs:ze, :]
                    before_zone = before[zs:ze, :] if before.ndim == 2 else before[zs:ze, np.newaxis]
                else:
                    ze = min(int(ze), len(after))
                    after_zone = after[zs:ze]
                    before_zone = before[zs:ze] if len(before) > zs else np.zeros(ze - zs, dtype=np.float32)

                if ze <= zs:
                    continue

                energy_after = float(np.sqrt(np.mean(after_zone.astype(np.float64) ** 2) + 1e-12))
                energy_before = float(np.sqrt(np.mean(before_zone.astype(np.float64) ** 2) + 1e-12))

                # +3 dB = Faktor 1.413 in Amplitude
                if energy_after > energy_before * 1.413:
                    # HARD RESET — Original-Samples einsetzen (kein Clip!)
                    if is_stereo_channels_first:
                        result[:, zs:ze] = before_zone
                    elif is_stereo_samples_first:
                        result[zs:ze, :] = before_zone
                    else:
                        result[zs:ze] = before_zone
                    n_resets += 1

            if n_resets > 0:
                logger.info(
                    "SSIP.post_inpainting_silence_audit: %d Zone(n) auf Originalsignal-Samples zurückgesetzt",
                    n_resets,
                )

            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]

        except Exception as exc:
            logger.warning("SSIP.post_inpainting_silence_audit fehlgeschlagen — Passthrough: %s", exc)
            return np.asarray(audio_after_inpainting, dtype=np.float32)  # type: ignore[no-any-return]
