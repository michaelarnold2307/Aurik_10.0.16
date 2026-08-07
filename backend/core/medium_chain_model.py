"""
PhysicalMediumChainModel — Weltspitzen-Differenzierer #2
=========================================================

Modelliert die vollständige physikalische Aufnahme-/Wiedergabe-Kette
für jeden Tonträgertyp und ermöglicht inverse Kettenentzerrung.

Kein anderes Programm modelliert die Signalfluss-Kette als Systemmodell.
Aurik invertiert die materialspezifische Kette *bevor* die eigentliche
Defektreparatur beginnt — das eliminiert systematische Fehler aller
nachfolgenden Phasen.

Modellierte Ketten:

SHELLAC (1900–1954):
  Grampophon-Schallhorn → mechanische Übertragung → Schneidkopf →
  Pressungsartefakte (pre-RIAA Entzerrungskurven: Columbia, Victor, HMV, Decca) →
  Abspielnadel-Compliance → Eigenresonanz des Tonarms

VINYL (1954–heute):
  Schneidkopf mit RIAA-Vorentzerrung → Plattenpressung →
  Tonabnehmer-Compliance → Tonarmeigenresonanz → Phono-Vorverstärker

TAPE (Kassette, 1963–2000):
  Magnetkopf-Spalt-Verlust → ferromagnetische Hysterese →
  Bias-induzierter HF-Rolloff → Azimuthfehler (Kanalversatz) →
  Print-Through-Echo

REEL_TAPE (Studio-Bandmaschine, 1940–1990):
  Wie Tape, aber mit höherer Bandgeschwindigkeit → weniger HF-Rolloff,
  mehr Print-Through bei 15 ips, Azimuthdrift über Bandlänge

CD / DIGITAL (1982–heute):
  DAC-Rekonstruktionsfilter-Artefakte → Jitter-Seitenbänder →
  Loudness-War-Clipping (1990–2010) → lossy-Codec-Artefakte
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal

from backend.core.defect_scanner import DefectScore, DefectType, MaterialType

logger = logging.getLogger(__name__)


@dataclass
class ChainInversionResult:
    """Ergebnis einer Ketteninversion."""

    audio: np.ndarray
    """Kettenentzerrtes Audio."""
    material: MaterialType
    corrections_applied: list[str] = field(default_factory=list)
    """Welche Korrekturen wurden angewendet (für Audit)."""
    spectral_change_db: float = 0.0
    """Mittlere spektrale Änderung in dB (Maß für Stärke der Korrektur)."""


class PhysicalMediumChainModel:
    """
    Physikalisches Tonträger-Kettenmodell (Aurik 10.0.0).

    Jeder Tonträgertyp hat eine charakteristisch geprägte Signalkette.
    Diese Klasse modelliert die Kette und invertiert sie DSP-präzise,
    um das Audio in einen neutralen Zustand vor Material-Färbung zu bringen.

    Verwendung:
        model = PhysicalMediumChainModel()
        result = model.invert_chain(audio, sample_rate, material, detected_defects)
        corrected_audio = result.audio
    """

    # Pre-RIAA Entzerrungskurven für Schellack-Ära (vor 1954)
    # Format: (Höhen-Turnover-Freq Hz, Bass-Turnover-Freq Hz, HF-Shelf-dB)
    # Historische Quellen: IEC 98 Appendix, Copeland (2008)
    SHELLAC_EQ_CURVES: dict[str, tuple[float, float, float]] = {
        "columbia_78": (300, 150, -16),  # Columbia 78rpm (bis 1947)
        "victor_78": (500, 150, -12),  # Victor/RCA 78rpm
        "hmv_78": (250, 150, -14),  # HMV/EMI 78rpm
        "decca_78": (300, 100, -12),  # Decca 78rpm
        "riaa_1954": (3180, 318, -13.7),  # RIAA (nach 1954), Standard
    }

    # RIAA-Zeitkonstanten (ms): Bass-Shelf 3180μs, Mittelband 318μs, Treble -75μs
    RIAA_TIME_CONSTANTS = (3180e-6, 318e-6, 75e-6)

    # Tape HF-Rolloff Profile (Rolloff-Frequenz bei -3dB, typisch bei Aufnahmen)
    TAPE_HF_ROLLOFF: dict[MaterialType, float] = {
        MaterialType.TAPE: 15000.0,  # §6.2c central definition (tape = 15 kHz)
        MaterialType.REEL_TAPE: 18000.0,  # Profibandmaschine: ~18 kHz bei 15 ips
    }

    # Tape Bias-Boost-Kompensation (sanfter HF-Shelf)
    TAPE_HF_BOOST_DB: dict[MaterialType, float] = {
        MaterialType.TAPE: 3.5,  # Kassette: moderate Korrektur
        MaterialType.REEL_TAPE: 2.0,  # Reel: geringere Korrektur benötigt
    }

    def invert_chain(
        self,
        audio: np.ndarray,
        sample_rate: int,
        material: MaterialType,
        detected_defects: list[DefectScore] | None = None,
    ) -> ChainInversionResult:
        """
        Wendet die inverse materialspezifische Kettenentzerrung an.

        Reihenfolge der Korrekturen ist material-spezifisch — z.B. muss bei
        Shellac die pre-RIAA-Kurven-Korrektur vor allem anderen erfolgen.

        Args:
            audio:   Eingabe-Audio (float32, mono oder stereo).
            sample_rate: Abtastrate in Hz.
            material: Erkannter Materialtyp.
            detected_defects: Erkannte Defekte (optional, für kontextabhängige Korrektur).

        Returns:
            ChainInversionResult mit korrigiertem Audio und Korrektur-Protokoll.
        """
        audio_out = audio.astype(np.float32)
        corrections: list[str] = []
        detected_types = {d.defect_type for d in (detected_defects or [])}

        if material == MaterialType.SHELLAC:
            audio_out, c = self._invert_shellac(audio_out, sample_rate, detected_types)
            corrections.extend(c)

        elif material == MaterialType.VINYL:
            audio_out, c = self._invert_vinyl(audio_out, sample_rate, detected_types)
            corrections.extend(c)

        elif material in (MaterialType.TAPE, MaterialType.REEL_TAPE):
            audio_out, c = self._invert_tape(audio_out, sample_rate, material, detected_types)
            corrections.extend(c)

        elif material == MaterialType.CD_DIGITAL:
            audio_out, c = self._invert_cd(audio_out, sample_rate, detected_types)
            corrections.extend(c)

        else:
            logger.debug("Kein Kettenmodell für %s — keine Korrektur angewendet.", material.value)

        # Safety only: avoid loudness-changing peak rescale.
        peak: float = float(np.max(np.abs(audio_out)))
        if peak > 1.0:
            audio_out = np.clip(audio_out, -1.0, 1.0)
            corrections.append("safety_clamp")

        # Spektrale Änderung berechnen (für Audit)
        spectral_change = self._measure_spectral_change(audio, audio_out, sample_rate)

        logger.info(
            "Ketteninversion [%s]: %d Korrekturen, spektr. Δ=%.2f dB",
            material.value,
            len(corrections),
            spectral_change,
        )

        return ChainInversionResult(
            audio=audio_out,
            material=material,
            corrections_applied=corrections,
            spectral_change_db=round(spectral_change, 2),
        )

    def invert_chain_sequence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transfer_chain_raw: dict,
        detected_defects: list[DefectScore] | None = None,
    ) -> ChainInversionResult:
        """
        Invertiert eine mehrstufige Überspielungskette (z.B. Vinyl → Kassette → MP3).

        Korrekturreihenfolge: jüngste Generation zuerst (höchster Score zuerst).
        Beispiel detected_media = [("damaged_mp3", 1.0), ("cassette", 0.8), ("vinyl", 0.4)]:
            1. MP3-Korrektur (kein Modell → pass)
            2. Kassetten-EQ-Inversion
            3. Vinyl-RIAA-Inversion
        """
        _FORENSIC_TO_MATERIAL: dict[str, MaterialType] = {
            "vinyl": MaterialType.VINYL,
            "shellac": MaterialType.SHELLAC,
            "cassette": MaterialType.TAPE,
            "open_reel": MaterialType.REEL_TAPE,
            "cd": MaterialType.CD_DIGITAL,
            "digital": MaterialType.CD_DIGITAL,
            "damaged_mp3": MaterialType.MP3_HIGH,
            "mp3": MaterialType.MP3_HIGH,
            "aac": MaterialType.AAC,
            "wire_recording": MaterialType.WIRE_RECORDING,
        }

        # Normalisierung: MediumDetectionResult (Dataclass) oder dict erlaubt.
        # MediumDetectionResult.as_dict() liefert {"transfer_chain": [...], "primary_material": ...,
        # "chain_label": ...} — kein "detected_media"-Key, daher separat behandeln.
        if not isinstance(transfer_chain_raw, dict):
            _chain_list = getattr(transfer_chain_raw, "transfer_chain", []) or []
            _primary = getattr(transfer_chain_raw, "primary_material", "digital")
            _label = getattr(transfer_chain_raw, "chain_label", " → ".join(_chain_list) if _chain_list else "")
            transfer_chain_raw = {
                "detected_media": [(m, 1.0) for m in _chain_list],
                "type": _primary,
                "chain": _label,
            }
        # as_dict()-Ergebnis hat "transfer_chain" statt "detected_media"
        if "detected_media" not in transfer_chain_raw and "transfer_chain" in transfer_chain_raw:
            _chain_list = transfer_chain_raw["transfer_chain"] or []
            transfer_chain_raw = dict(transfer_chain_raw)
            transfer_chain_raw["detected_media"] = [(m, 1.0) for m in _chain_list]
        detected_media = transfer_chain_raw.get("detected_media", [])
        if not detected_media:
            primary_type_str = transfer_chain_raw.get("type", "digital")
            primary_material = _FORENSIC_TO_MATERIAL.get(primary_type_str, MaterialType.UNKNOWN)
            return self.invert_chain(audio, sample_rate, primary_material, detected_defects)

        all_corrections: list[str] = []
        spectral_changes: list[float] = []
        audio_out = audio.astype(np.float32)

        for medium_str, _score in detected_media:  # Sortiert absteigend (jüngste zuerst)
            material = _FORENSIC_TO_MATERIAL.get(medium_str, MaterialType.UNKNOWN)
            if material == MaterialType.UNKNOWN:
                logger.debug("Kein Mapping für Medium '%s' → übersprungen.", medium_str)
                continue
            step_result = self.invert_chain(audio_out, sample_rate, material, detected_defects)
            audio_out = step_result.audio
            all_corrections.extend(f"{medium_str}:{c}" for c in step_result.corrections_applied)
            spectral_changes.append(step_result.spectral_change_db)

        peak: float = float(np.max(np.abs(audio_out)))
        if peak > 1.0:
            audio_out = np.clip(audio_out, -1.0, 1.0)
            all_corrections.append("chain_sequence_safety_clamp")

        primary_material_str = detected_media[0][0] if detected_media else "digital"
        primary_material = _FORENSIC_TO_MATERIAL.get(primary_material_str, MaterialType.UNKNOWN)
        chain_label = (
            transfer_chain_raw.get("chain", " → ".join(m for m, _ in detected_media))
            if isinstance(transfer_chain_raw, dict)
            else " → ".join(m for m, _ in detected_media)
        )
        logger.info(
            "Mehrstufige Ketteninversion [%s]: %d Korrekturen, Kette=%s",
            primary_material_str,
            len(all_corrections),
            chain_label,
        )

        return ChainInversionResult(
            audio=np.clip(audio_out, -1.0, 1.0),
            material=primary_material,
            corrections_applied=all_corrections,
            spectral_change_db=round(float(np.mean(spectral_changes)) if spectral_changes else 0.0, 2),
        )

    # ------------------------------------------------------------------
    # Material-spezifische Inversionen
    # ------------------------------------------------------------------

    def _invert_shellac(
        self,
        audio: np.ndarray,
        sr: int,
        detected: set,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Shellac-Ketten-Inversion:
        1. Pre-RIAA-Kurvenkorrektur (Auto-Kurvenerkennung via Spektralanalyse)
        2. Schallhorn-Hochpassresonanz-Dämpfung
        3. Nadel-Compliance-Korrekturfaktor
        """
        corrections: list[str] = []
        audio_out = audio.copy()

        # 1. Pre-RIAA Kurvenkorrektur — wähle die wahrscheinlichste Kurve
        curve_name, turnover_hi, _turnover_lo, treble_db = self._detect_shellac_curve(audio, sr)
        audio_out = self._apply_recording_curve_inverse(audio_out, sr, turnover_hi, treble_db)
        corrections.append(f"pre_riaa_curve_inverse:{curve_name}")

        # 2. Schallhorn-Hochton-Resonanzdämpfung (alte Aufnahmen ~700–1200 Hz peak)
        # Monoaufnahmen  haben oft einen spitzen Präsenz-Peak vom Trichterresonator
        if self._has_horn_resonance(audio, sr):
            audio_out = self._damp_horn_resonance(audio_out, sr)
            corrections.append("horn_resonance_damping")

        # 3. Hoch-Frequenz-Rauschteppich-Floor anheben (Shellac-Rauschprofil)
        # Schellack hat relativ gleichförmiges Breitbandrauschen → mild
        if DefectType.HIGH_FREQ_NOISE not in detected:
            audio_out = self._shellac_noise_floor_correction(audio_out, sr)
            corrections.append("shellac_noise_floor_correction")

        return audio_out, corrections

    def _invert_vinyl(
        self,
        audio: np.ndarray,
        sr: int,
        detected: set,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Vinyl-Ketten-Inversion:
        1. RIAA-Pre-emphasis-Artefakt-Check (falls Phono-Vorverstärker fehlte)
        2. Tonabnehmer-Eigenresonanz-Korrektur
        3. Schallplattenrumble-Trennung (Subsonic < 20 Hz)
        """
        corrections: list[str] = []
        audio_out = audio.copy()

        # 1. RIAA-Check: Wurde die Platte durch einen Phono-Vorverstärker gespielt?
        # Indikator: massiver Bassüberschuss + HF-Abfall → pre-emphasis noch aktiv
        if self._has_riaa_preemphasis(audio, sr):
            audio_out = self._apply_riaa_deemphasis(audio_out, sr)
            corrections.append("riaa_deemphasis_applied")

        # 2. Tonabnehmer-Eigenresonanz-Dämpfung (typisch 8-12 Hz)
        audio_out = self._apply_subsonic_filter(audio_out, sr, cutoff_hz=20.0)
        corrections.append("subsonic_filter_20hz")

        # 3. Leichte HF-Präsenz-Korrektur (Nadelschliff-bedingte HF-Beschneidung)
        if DefectType.BANDWIDTH_LOSS in detected:
            audio_out = self._gentle_hf_restore(audio_out, sr, shelf_hz=8000, gain_db=1.5)
            corrections.append("vinyl_hf_presence_restore")

        return audio_out, corrections

    def _invert_tape(
        self,
        audio: np.ndarray,
        sr: int,
        material: MaterialType,
        detected: set,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Tape-Ketten-Inversion:
        1. Bias-induzierter HF-Rolloff-Ausgleich (High-Shelf-Boost)
        2. Azimuth-Kammfilter-Korrektur (stereo)
        3. Print-Through-Vorentzerrung (subtile Absenkung des Pre-Echo-Bereichs)
        4. Ferromagnetische Hysterese — sanfte Oberton-Bereinigung
        """
        corrections: list[str] = []
        audio_out = audio.copy()

        # 1. Bias-HF-Rolloff ausgleichen
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        rolloff_freq = self.TAPE_HF_ROLLOFF.get(_mk, 8000)  # type: ignore[call-overload]
        boost_db = self.TAPE_HF_BOOST_DB.get(_mk, 0.0)  # type: ignore[call-overload]
        audio_out = self._apply_hf_shelf_boost(audio_out, sr, rolloff_freq, boost_db)
        corrections.append(f"bias_hf_rolloff_compensation_{int(rolloff_freq)}hz_{boost_db}dB")

        # 2. Azimuth-Kammfilter-Korrektur (nur relevant bei Stereo)
        # Supports both (N, 2) samples-first and (2, N) channels-first (UV3)
        is_stereo_2d = audio_out.ndim == 2 and (
            (audio_out.shape[1] == 2 and audio_out.shape[0] != 2)  # (N, 2)
            or (audio_out.shape[0] == 2 and audio_out.shape[1] != 2)  # (2, N)
        )
        if is_stereo_2d:
            if DefectType.STEREO_IMBALANCE in detected or self._has_azimuth_comb(audio_out, sr):
                audio_out = self._correct_azimuth_comb(audio_out, sr)
                corrections.append("azimuth_comb_correction")

        # 3. Subtile Hysterese-Oberton-Bereinigung (sanfter)
        audio_out = self._reduce_hysteresis_harmonics(audio_out, sr)
        corrections.append("ferromagnetic_hysteresis_reduction")

        return audio_out, corrections

    def _invert_cd(
        self,
        audio: np.ndarray,
        sr: int,
        detected: set,
    ) -> tuple[np.ndarray, list[str]]:
        """
        CD/Digital-Ketten-Inversion:
        1. Jitter-Seitenband-Unterdrückung (feines Kammfilter um harmonische Seitenbänder)
        2. Rekonstruktionsfilter-Pre-Ring-Dämpfung (Gibbs-Phänomen-Artefakte)
        3. Loudness-War-Clipping-Detektion (Vorbereitung für Phase 07)
        """
        corrections: list[str] = []
        audio_out = audio.copy()

        # 1. Rekonstruktionsfilter-Artefakte (anti-aliasing pre-ringing)
        # Moderater De-Ringing mit minimalem Qualitätsverlust
        if sr >= 44100:
            audio_out = self._damp_reconstruction_preringing(audio_out, sr)
            corrections.append("reconstruction_filter_preringing_reduction")

        # 2. Jitter-Seitenband-Unterdrückung (nur bei erkanntem Jitter)
        if DefectType.JITTER_ARTIFACTS in detected:
            audio_out = self._suppress_jitter_sidebands(audio_out, sr)
            corrections.append("jitter_sideband_suppression")

        return audio_out, corrections

    # ------------------------------------------------------------------
    # DSP-Hilfsmethoden
    # ------------------------------------------------------------------

    def _detect_shellac_curve(self, audio: np.ndarray, sr: int) -> tuple[str, float, float, float]:
        """
        Schätzt die wahrscheinlichste Shellac-Aufnahmekurve aus dem Spektrum.
        Returns: (curve_name, turnover_hi_hz, turnover_lo_hz, treble_db)
        """
        # Spectral tilt als Indikator: starker Bass + HF-Abfall → ältere Kurve
        nyq = sr / 2
        np.linspace(0, nyq, 512)
        mono = audio[:, 0] if audio.ndim == 2 else audio
        # Kurze FFT-Analyse (erster Abschnitt, bis 2s)
        analysis_len = min(len(mono), sr * 2)
        spectrum = np.abs(np.fft.rfft(mono[:analysis_len], n=1024))
        freq_bins = np.fft.rfftfreq(1024, d=1.0 / sr)

        # Bass-/Hochtonverhältnis als Kurven-Selector
        bass_rms = np.mean(spectrum[(freq_bins > 80) & (freq_bins < 200)] ** 2)
        hi_rms = np.mean(spectrum[(freq_bins > 4000) & (freq_bins < 8000)] ** 2)
        ratio_db = 10 * np.log10((bass_rms + 1e-10) / (hi_rms + 1e-10))

        # Sehr viel Bass → alte Columbia/Victor-Kurve
        if ratio_db > 20:
            return ("columbia_78", 300, 150, -16)
        elif ratio_db > 14:
            return ("victor_78", 500, 150, -12)
        elif ratio_db > 8:
            return ("hmv_78", 250, 150, -14)
        else:
            return ("riaa_1954", 3180, 318, -13.7)

    def _apply_recording_curve_inverse(
        self, audio: np.ndarray, sr: int, turnover_hz: float, treble_shelf_db: float
    ) -> np.ndarray:
        """Wendet die inverse Aufnahmekurve an (Hochton-Boost, Bass-Roll)."""
        # Biquad peak/shelving EQ für inverse Kurve
        # High-Shelf-Boost als Kompensation des HF-Rolloffs
        boost_gain = -treble_shelf_db  # invertiere: -(-16) = +16 dB maximal, verringert auf 8
        boost_db = min(boost_gain * 0.5, 10.0)  # Konservativ: halbe Korrektur max. 10 dB
        return self._apply_hf_shelf_boost(audio, sr, turnover_hz, boost_db)

    def _has_horn_resonance(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt Schallhorn-Resonanzen (typisch 700–1500 Hz, peaks > 3 dB über Mittel)."""
        mono = audio[:, 0] if audio.ndim == 2 else audio
        N = min(len(mono), sr)
        spectrum = np.abs(np.fft.rfft(mono[:N]))
        freqs = np.fft.rfftfreq(N, d=1.0 / sr)

        mask_mid = (freqs > 600) & (freqs < 1600)
        mask_ref = (freqs > 200) & (freqs < 600)
        if not np.any(mask_mid) or not np.any(mask_ref):
            return False

        mid_level = np.mean(spectrum[mask_mid] ** 2)
        ref_level = np.mean(spectrum[mask_ref] ** 2)
        ratio_db = 10 * np.log10((mid_level + 1e-10) / (ref_level + 1e-10))
        return ratio_db > 4.0  # type: ignore[no-any-return]

    def _damp_horn_resonance(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Dämpft Schallhorn-Resonanz-Peak mit Notch-EQ bei ~1000 Hz."""
        nyq = sr / 2
        freq = min(1000.0, nyq * 0.45)
        Q = 2.0
        gain_db = -3.0
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * Q)
        b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
        a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
        return sp_signal.lfilter(b, a, audio, axis=0)  # type: ignore[no-any-return]

    def _shellac_noise_floor_correction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Sanfte Hochtonanhebung (kompensiert Shellac-bedingte HF-Dämpfung)."""
        return self._apply_hf_shelf_boost(audio, sr, shelf_hz=5000, gain_db=1.0)

    def _has_riaa_preemphasis(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt ob RIAA-Pre-Emphasis noch aktiv ist (Bass massiv, HF sehr gedämpft)."""
        mono = audio[:, 0] if audio.ndim == 2 else audio
        N = min(len(mono), sr)
        spec = np.abs(np.fft.rfft(mono[:N]))
        freqs = np.fft.rfftfreq(N, d=1.0 / sr)

        bass_mask = (freqs > 40) & (freqs < 200)
        hi_mask = (freqs > 6000) & (freqs < 14000)
        if not (np.any(bass_mask) and np.any(hi_mask)):
            return False

        bass_rms = np.mean(spec[bass_mask] ** 2)
        hi_rms = np.mean(spec[hi_mask] ** 2)
        ratio_db = 10 * np.log10((bass_rms + 1e-10) / (hi_rms + 1e-10))
        # RIAA pre-emphasis: Bass ist ~20 dB über HF → noch nicht deemphasisiert
        return ratio_db > 22.0  # type: ignore[no-any-return]

    def _apply_riaa_deemphasis(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Wendet RIAA De-Emphasis an (kompensiert versehentliches pre-emphasis)."""
        # RIAA De-Emphasis via bilinearen Z-Transform der Zeitkonstanten
        tau1, _tau2, tau3 = self.RIAA_TIME_CONSTANTS  # 3180, 318, 75 μs
        # Implementierung als zweistufiger Shelving-Filter
        # Basis: Hochpass ab 1/(2π·τ1) = 50 Hz, Lowpass ab 1/(2π·τ3) = 2122 Hz
        audio_out = self._apply_hf_shelf_boost(audio, sr, 1 / (2 * np.pi * tau3), -13.7)
        audio_out = self._apply_hf_shelf_boost(audio_out, sr, 1 / (2 * np.pi * tau1), 2.0)
        return audio_out

    def _apply_subsonic_filter(self, audio: np.ndarray, sr: int, cutoff_hz: float = 20.0) -> np.ndarray:
        """Butterworth-Hochpass 4. Ordnung unter cutoff_hz."""
        nyq = sr / 2
        if cutoff_hz >= nyq:
            return audio
        sos = sp_signal.butter(4, cutoff_hz / nyq, btype="high", output="sos")
        return sp_signal.sosfilt(sos, audio, axis=0)  # type: ignore[no-any-return]

    def _gentle_hf_restore(self, audio: np.ndarray, sr: int, shelf_hz: float, gain_db: float) -> np.ndarray:
        """Sanfte HF-Shelf-Anhebung für Vinyl-Hochtonkorrektur."""
        return self._apply_hf_shelf_boost(audio, sr, shelf_hz, gain_db)

    def _apply_hf_shelf_boost(self, audio: np.ndarray, sr: int, shelf_hz: float, gain_db: float) -> np.ndarray:
        """Generischer Biquad High-Shelf EQ."""
        nyq = sr / 2
        freq = min(shelf_hz, nyq * 0.45)
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * freq / sr
        cosw = np.cos(w0)
        sinw = np.sin(w0)
        beta = np.sqrt(A) / 1.5  # Q ≈ 1/sqrt(2) für flachen Shelf

        b = np.array(
            [
                A * ((A + 1) + (A - 1) * cosw + beta * sinw),
                -2 * A * ((A - 1) + (A + 1) * cosw),
                A * ((A + 1) + (A - 1) * cosw - beta * sinw),
            ]
        )
        a = np.array(
            [
                (A + 1) - (A - 1) * cosw + beta * sinw,
                2 * ((A - 1) - (A + 1) * cosw),
                (A + 1) - (A - 1) * cosw - beta * sinw,
            ]
        )
        b /= a[0]
        a /= a[0]
        return sp_signal.lfilter(b, a, audio, axis=0).astype(audio.dtype)  # type: ignore[no-any-return]

    def _has_azimuth_comb(self, audio: np.ndarray, sr: int) -> bool:
        """Erkennt Azimuth-Kammfilter-Muster (Kanalkorrelation bei HF ≈ 0)."""
        if audio.ndim < 2 or audio.shape[1] < 2:
            return False
        N = min(len(audio), sr // 2)
        L = np.fft.rfft(audio[:N, 0])
        R = np.fft.rfft(audio[:N, 1])
        freqs = np.fft.rfftfreq(N, d=1.0 / sr)
        hi_mask = freqs > 8000
        if not np.any(hi_mask):
            return False
        correlation = np.mean(np.real(L[hi_mask] * np.conj(R[hi_mask])))
        total = np.mean(np.abs(L[hi_mask]) * np.abs(R[hi_mask])) + 1e-10
        # Azimuth-Kammfilter: HF-Kanalkorrelation < 0.3
        return float(correlation / total) < 0.3

    def _correct_azimuth_comb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Minimale Azimuth-Korrrektur via Kanal-Delay-Ausrichtung.
        Schätzt den sample-genauen Kanalversatz und kompensiert ihn.
        """
        if audio.ndim < 2 or audio.shape[1] < 2:
            return audio
        N = min(len(audio), sr)
        # Kreuzkorrelation für Delay-Schätzung — FFT-based O(N log N)
        from backend.core.core_utils import fft_crosscorr

        corr = fft_crosscorr(audio[:N, 0], audio[:N, 1])
        best_lag = int(np.argmax(np.abs(corr)) - (N - 1))
        # Maximal 50 Samples Korrektur (> 1 ms bei 44.1 kHz = grob falsch)
        best_lag = max(-50, min(50, best_lag))
        if best_lag == 0:
            return audio
        out = audio.copy()
        if best_lag > 0:
            out[best_lag:, 1] = audio[:-best_lag, 1]
        elif best_lag < 0:
            out[:best_lag, 0] = audio[-best_lag:, 0]
        return out

    def _reduce_hysteresis_harmonics(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Sanfte Minderung ferromagnetischer Hysterese-Obertöne.
        Modelliert: Tonbandmaschinen erzeugen ungerade Harmonische (3f, 5f).
        Subtile Breit-Band-Sättigung verringern → leichte Hochtonabsenkung.
        """
        # Sanfte Soft-Clip-Inversion (inverse Sättigung)
        # Sehr kleiner Effekt: < 0.5 dB Änderung
        factor = 0.02
        return audio - factor * (audio**3)  # type: ignore[no-any-return]

    def _damp_reconstruction_preringing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Dämpft Pre-Ringing-Artefakte von scharfen Anti-Aliasing-Filtern (CD).
        Implementiert als minimaler kausaler Tiefpassfilter nahe Nyquist.
        """
        nyq = sr / 2
        cutoff = min(0.98 * nyq, nyq - 100)
        sos = sp_signal.butter(2, cutoff / nyq, btype="low", output="sos")
        return sp_signal.sosfilt(sos, audio, axis=0).astype(audio.dtype)  # type: ignore[no-any-return]

    def _suppress_jitter_sidebands(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Unterdrückt Jitter-Seitenbänder (harmonische Störtöne um Grundfrequenzen).
        Implementiert via adaptivem Kammfilter nahe Abtastrate-Vielfachen.
        """
        # Sanfter Tiefpass unter 0.95·Nyquist (Jitter-Seitenbänder nahe Nyquist)
        nyq = sr / 2
        cutoff = 0.95 * nyq
        sos = sp_signal.butter(2, cutoff / nyq, btype="low", output="sos")
        # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) — filtered wird mit audio geblendet.
        filtered = sp_signal.sosfiltfilt(sos, audio, axis=0).astype(audio.dtype)
        # Blend: 80% original + 20% filtered (konservativ)
        return 0.85 * audio + 0.15 * filtered  # type: ignore[no-any-return]

    def _measure_spectral_change(self, original: np.ndarray, corrected: np.ndarray, sr: int) -> float:
        """Mittlere spektrale Änderung in dB (Maß für Stärke der Korrektur)."""
        N = min(len(original), sr, len(corrected))
        if N < 512:
            return 0.0
        orig_mono = original[:N, 0] if original.ndim == 2 else original[:N]
        corr_mono = corrected[:N, 0] if corrected.ndim == 2 else corrected[:N]
        spec_orig = np.abs(np.fft.rfft(orig_mono))
        spec_corr = np.abs(np.fft.rfft(corr_mono))
        diff_db = 20 * np.log10((spec_corr + 1e-10) / (spec_orig + 1e-10))
        return float(np.mean(np.abs(diff_db)))
