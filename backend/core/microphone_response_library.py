"""
MicrophoneResponseLibrary — §6.4a [RELEASE_MUST]
=================================================

Historische Mikrofon-EQ-Profile für era-adaptive Signalverarbeitung.
Liefert EQ-Kurven für Phase_38 / Phase_06 um die Recording-Chain
des Originals zu modellieren.

Spec: 05_material_system.md §6.4a (v10.0.0)
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MicrophoneProfile:
    """Historisches Mikrofon-Profil (§6.4a).

    Attrs:
        name:            Mikrofon-Name (z.B. "RCA 44-BX (1932)").
        years_active:    Aktivitätszeitraum als (start, end) Tupel.
        type:            Mikrofon-Typ: "ribbon", "condenser", "dynamic", "crystal".
        freq_response_db: {Hz: dB} Frequenzgang — Referenz 1 kHz = 0 dB.
        genres:          Typische Einsatzgebiete.
        notes:           Freitext-Anmerkungen.
        materials:       Kompatible Materialtypen (aus JSON).
        pattern:         Richtcharakteristik (aus JSON, z.B. "bidirectional").
        rolloff_hz:      -3 dB Rolloff-Frequenz in Hz.
        profile_id:      Eindeutige ID (aus JSON).
    """

    name: str
    years_active: tuple[int, int]
    type: str
    freq_response_db: dict[int, float]
    genres: list[str]
    notes: str = ""
    # Erweiterte Felder aus JSON (nicht im Spec-Kern, aber für Scoring benötigt)
    materials: list[str] = field(default_factory=list)
    pattern: str = ""
    rolloff_hz: float = 0.0
    profile_id: str = ""


def _dict_to_profile(d: dict) -> Optional["MicrophoneProfile"]:
    """Wandelt ein JSON-Profil-Dict in ein MicrophoneProfile-Objekt."""
    try:
        era = d.get("era_decade", [1900, 1900])
        years_active = (int(era[0]), int(era[1])) if len(era) >= 2 else (int(era[0]), int(era[0]))
        freq_response_db: dict[int, float] = {}
        for point in d.get("eq_curve", []):
            hz = int(round(float(point["hz"])))
            db = float(point["db"])
            freq_response_db[hz] = db
        return MicrophoneProfile(
            name=str(d.get("name", "")),
            years_active=years_active,
            type=str(d.get("type", "dynamic")),
            freq_response_db=freq_response_db,
            genres=[str(g) for g in d.get("genres", [])],
            notes=str(d.get("notes", "")),
            materials=[str(m) for m in d.get("materials", [])],
            pattern=str(d.get("pattern", "")),
            rolloff_hz=float(d.get("rolloff_hz", 0.0)),
            profile_id=str(d.get("id", "")),
        )
    except Exception as exc:
        logger.warning("MicrophoneProfile: Konvertierung fehlgeschlagen (%s): %s", d.get("id", "?"), exc)
        return None


_instance: "MicrophoneResponseLibrary | None" = None
_lock = threading.Lock()

_PROFILES_PATH = Path(__file__).parent.parent / "data" / "microphone_profiles.json"


def get_microphone_response_library() -> "MicrophoneResponseLibrary":
    """Singleton-Getter (thread-safe, double-checked locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MicrophoneResponseLibrary()
    return _instance


class MicrophoneResponseLibrary:
    """Lädt und liefert historische Mikrofon-EQ-Profile (§6.4a)."""

    def __init__(self) -> None:
        self._profiles: list[MicrophoneProfile] = []
        self._load_profiles()

    def _load_profiles(self) -> None:
        try:
            with open(_PROFILES_PATH, encoding="utf-8") as f:
                data = json.load(f)
            raw_profiles = data.get("profiles", [])
            converted = [_dict_to_profile(p) for p in raw_profiles]
            self._profiles = [p for p in converted if p is not None]
            logger.info("MicrophoneResponseLibrary geladen %d profiles", len(self._profiles))
        except Exception as exc:
            logger.warning("MicrophoneResponseLibrary: profiles not geladen (%s)", exc)
            self._profiles = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(
        self,
        era_decade: int,
        genre_label: str,
        material_type: str,
    ) -> "MicrophoneProfile | None":
        """Gibt das am besten passende Mikrofon-Profil zurück.

        Scoring:
          +3 wenn era_decade im Profil-Bereich
          +2 wenn genre_label in genres
          +1 wenn material_type in materials

        Args:
            era_decade:   Aufnahme-Jahrzehnt als int (z.B. 1950 für 1950er).
            genre_label:  Genre (z.B. "jazz", "schlager", "rock").
            material_type: Material (z.B. "shellac", "vinyl", "reel_tape").

        Returns:
            Bestes MicrophoneProfile oder None wenn keine Profile geladen.
        """
        if not self._profiles:
            return None

        best_profile: MicrophoneProfile | None = None
        best_score = -1

        for profile in self._profiles:
            score = 0

            era_min, era_max = profile.years_active
            if era_min <= era_decade <= era_max + 10:
                score += 3

            genres = [g.lower() for g in profile.genres]
            if genre_label.lower() in genres:
                score += 2

            materials = [m.lower() for m in profile.materials]
            if material_type.lower() in materials:
                score += 1

            if score > best_score:
                best_score = score
                best_profile = profile

        return best_profile

    def get_eq_curve(
        self,
        era_decade: int,
        genre_label: str,
        material_type: str,
        target_sr: int = 48000,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Gibt EQ-Kurve als (freqs_hz, gains_linear) Arrays zurück.

        Args:
            era_decade:   Aufnahme-Jahrzehnt.
            genre_label:  Genre.
            material_type: Material.
            target_sr:    Sample-Rate für Nyquist-Begrenzung.

        Returns:
            Tuple (freqs: np.ndarray, gains_linear: np.ndarray) oder None.
            Frequenzen sind aufsteigend, gains_linear >= 0.

        Notes:
            - max wet_mix = 0.35 (§6.4a Invariante — kein hartes EQ-Match)
            - Verwende np.interp für Interpolation auf Ziel-Frequenzachse
        """
        profile = self.get_profile(era_decade, genre_label, material_type)
        if profile is None:
            return None

        freq_response_db = profile.freq_response_db
        if not freq_response_db:
            return None

        nyq = target_sr / 2.0
        freqs = []
        db_values = []

        for hz, db in sorted(freq_response_db.items()):
            if float(hz) <= nyq:
                freqs.append(float(hz))
                db_values.append(float(db))

        if len(freqs) < 2:
            return None

        freqs_arr = np.array(freqs, dtype=np.float32)
        gains_db = np.array(db_values, dtype=np.float32)
        gains_linear = np.power(10.0, gains_db / 20.0).astype(np.float32)

        return freqs_arr, gains_linear

    def apply_eq_curve(
        self,
        audio: np.ndarray,
        sr: int,
        era_decade: int,
        genre_label: str,
        material_type: str,
        wet_mix: float = 0.20,
    ) -> np.ndarray:
        """Wendet die EQ-Kurve als frequency-domain Shaping an.

        Args:
            audio:        Float32 Audio.
            sr:           Sample-Rate.
            era_decade:   Aufnahme-Jahrzehnt.
            genre_label:  Genre.
            material_type: Material.
            wet_mix:      Blend-Faktor [0, 0.35] (§6.4a Hard-Cap = 0.35).

        Returns:
            Audio mit applizierter EQ-Charakteristik, selbe Form und Länge.
        """
        wet_mix = float(np.clip(wet_mix, 0.0, 0.35))  # §6.4a Hard-Cap

        eq_result = self.get_eq_curve(era_decade, genre_label, material_type, sr)
        if eq_result is None:
            return audio

        freqs_eq, gains_eq = eq_result

        try:
            original_shape = audio.shape
            mono = audio.mean(axis=0) if audio.ndim == 2 and audio.shape[0] == 2 else audio
            if mono.ndim == 2:
                mono = mono.mean(axis=1)

            n = len(mono)
            if n < 64:
                return audio

            # FFT-basiertes EQ via Interpolation auf FFT-Bins
            fft_freqs = np.fft.rfftfreq(n, d=1.0 / sr).astype(np.float32)
            gains_interp = np.interp(fft_freqs, freqs_eq, gains_eq, left=gains_eq[0], right=gains_eq[-1])

            def _apply_to_channel(ch: np.ndarray) -> np.ndarray:
                spectrum = np.fft.rfft(ch.astype(np.float64))
                spectrum_eq = spectrum * gains_interp.astype(np.float64)
                result = np.fft.irfft(spectrum_eq, n=n)
                return result.astype(np.float32)  # type: ignore[no-any-return]

            if audio.ndim == 1:
                eq_audio = _apply_to_channel(audio)
            elif audio.ndim == 2 and audio.shape[0] == 2:
                eq_audio = np.stack(
                    [
                        _apply_to_channel(audio[0]),
                        _apply_to_channel(audio[1]),
                    ]
                )
            elif audio.ndim == 2 and audio.shape[1] == 2:
                eq_audio = np.stack(
                    [
                        _apply_to_channel(audio[:, 0]),
                        _apply_to_channel(audio[:, 1]),
                    ],
                    axis=1,
                )
            else:
                eq_audio = _apply_to_channel(audio.flatten()).reshape(original_shape)

            # Wet/Dry-Mix
            blended = (1.0 - wet_mix) * audio + wet_mix * eq_audio
            blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)
            blended = np.clip(blended, -1.0, 1.0)
            return blended.astype(np.float32)  # type: ignore[no-any-return]

        except Exception as exc:
            logger.warning("MicrophoneResponseLibrary.anwenden_eq_curve fehlgeschlagen: %s", exc)
            return audio
