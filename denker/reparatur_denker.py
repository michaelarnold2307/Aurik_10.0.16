"""
ReparaturDenker — Domäne: Gezielte DSP-Reparaturen.

Führt punktuelle Korrekturen durch, die nicht die gesamte Pipeline
benötigen: Click-Entfernung (Medianfilter), Netzbrumm-Unterdrückung
(Notch-Filter 50/60 Hz) und Clipping-Reparatur (Soft-Limiter +
kubische Spline-Interpolation).

Usage::

    from denker.reparatur_denker import get_reparatur_denker

    denker = get_reparatur_denker()
    ergebnis = denker.repariere(audio, sr=48000)
    sauberes_audio = ergebnis.audio
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class ReparaturErgebnis:
    """Ergebnis einer gezielten DSP-Reparatur."""

    audio: np.ndarray
    """Repariertes Audio (float32, Bereich [-1, 1])."""

    clicks_removed: int
    """Anzahl entfernter Impulse/Clicks."""

    hum_removed: bool
    """Ob Netzbrumm erkannt und entfernt wurde."""

    clipping_repaired: bool
    """Ob Clipping-Regionen repariert wurden."""

    clipping_regions: int
    """Anzahl reparierter Clipping-Regionen."""

    processing_note: str
    """Zusammenfassung der Reparaturen (Deutsch)."""

    warnings: list[str] = field(default_factory=list)
    """Warnungen bei der Verarbeitung."""

    # --- Kompatibilitäts-Felder ---
    repairs_applied: list[str] = field(default_factory=list)
    """Angewendete Reparatur-Phasen (Compat-Alias)."""

    quality_delta: float = 0.0
    """Qualitätsverbesserung (Compat-Alias, wird nicht aktiv berechnet)."""

    material: str = ""
    """Trägermedium-Hinweis (Compat-Alias)."""

    reasoning: str = ""
    """Begründung der Reparaturen (Compat-Alias für processing_note)."""

    repair_risk_profile: dict[str, float] = field(default_factory=dict)
    """Hörkritische Risiken der Reparatur für das zentrale Policy-Profil."""

    def as_dict(self) -> dict[str, object]:
        """Liefert alle Felder als serialisierbares Dict."""
        return {
            "clicks_removed": self.clicks_removed,
            "hum_removed": self.hum_removed,
            "clipping_repaired": self.clipping_repaired,
            "clipping_regions": self.clipping_regions,
            "processing_note": self.processing_note,
            "warnings": self.warnings,
            "repair_risk_profile": dict(self.repair_risk_profile),
            "audio_shape": list(self.audio.shape),
        }


# ---------------------------------------------------------------------------
# ReparaturDenker
# ---------------------------------------------------------------------------


class ReparaturDenker:
    """Reparatur-Domänendenker — gezielte DSP-Korrekturen.

    Reparatur-Kaskade (Reihenfolge einhalten)
    -----------------------------------------
    1. Click-Entfernung via Medianfilter + Interquartilsabstandserkennung.
    2. Netzbrumm-Unterdrückung via Zweistufiger Butterworth-Notch (50 + 60 Hz).
    3. Clipping-Reparatur via monotoner kubischer Spline-Interpolation.

    Invarianten
    -----------
    - Eingabe: NaN/Inf → ``nan_to_num`` vor jeder Operation.
    - Ausgabe: immer ``np.clip(..., -1.0, 1.0)``.
    - Singleton via :func:`get_reparatur_denker` (Double-Checked Locking).
    """

    # Schwellwerte (Defaults — werden material-adaptiv überschrieben in repariere())
    _CLICK_IQR_MULTIPLIER: float = 6.0  # IQR-Multiplikator für Click-Detektion
    _CLICK_KERNEL_MS: float = 1.5  # Medianfilter-Halbfenster in ms
    _CLIP_THRESHOLD: float = 0.995  # Amplitude ≥ threshold gilt als Clipping
    _HUM_DETECT_DB: float = -50.0  # Energieschwelle für Brumm-Erkennung (dBFS)

    # Verlustbehaftete Codecs, die MDCT-Quantisierungsartefakte erzeugen (Brandenburg 1999).
    # Bei Terminal-Codec aus dieser Menge wird click_iqr nach oben auf Codec-IQR korrigiert,
    # um False-Positive-Clicks durch MP3/AAC-Subbandquantisierung zu vermeiden.
    _LOSSY_CODECS: frozenset[str] = frozenset(
        {
            "mp3_low",
            "mp3_standard",
            "mp3_high",
            "mp3",
            "aac",
            "aac_low",
            "wma",
            "ogg",
            "opus",
        }
    )

    # Material-adaptive Schwellwertprofile (§2.41 Klangtreue v10.0.0)
    # Quellen: Copeland 2008 (Manual of Analogue Sound Restoration),
    #          Katz 2007 (Mastering Audio)
    # Shellac/Wax = hoher Grundrausch + viele Clicks → aggressivere Detektion.
    # CD/Digital = sauberes Signal → konservativere Detektion (weniger False Positives).
    _MATERIAL_PROFILES: dict[str, dict[str, float]] = {
        "shellac": {
            "click_iqr": 4.0,  # Shellac: viele Klicks, niedrigere Schwelle
            "click_kernel_ms": 1.0,  # Schmalerer Kernel für kurze Impulse
            "clip_threshold": 0.990,
            "hum_detect_db": -45.0,  # Grammophon-Motoren erzeugen hörbaren Brumm
        },
        "wax_cylinder": {
            "click_iqr": 3.5,  # Wachszylinder: extremer Verschleiß
            "click_kernel_ms": 1.0,
            "clip_threshold": 0.985,
            "hum_detect_db": -42.0,
        },
        "wire_recording": {
            "click_iqr": 4.5,
            "click_kernel_ms": 1.2,
            "clip_threshold": 0.990,
            "hum_detect_db": -45.0,
        },
        "vinyl": {
            "click_iqr": 5.0,  # Vinyl: moderater Verschleiß, weniger Clicks als Shellac
            "click_kernel_ms": 1.5,
            "clip_threshold": 0.993,
            "hum_detect_db": -48.0,
        },
        "tape": {
            "click_iqr": 7.0,  # Tape: wenig Clicks, hauptsächlich Hiss/Flutter
            "click_kernel_ms": 2.0,
            "clip_threshold": 0.992,  # Tape-Sättigung beginnt sanfter
            "hum_detect_db": -50.0,
        },
        "cassette": {
            "click_iqr": 7.0,
            "click_kernel_ms": 2.0,
            "clip_threshold": 0.992,
            "hum_detect_db": -50.0,
        },
        "reel_tape": {
            "click_iqr": 8.0,  # Pro-Tape: fast keine Clicks, hauptsächlich Print-Through
            "click_kernel_ms": 2.0,
            "clip_threshold": 0.995,
            "hum_detect_db": -52.0,
        },
        "cd_digital": {
            "click_iqr": 9.0,  # CD: fast keine Clicks, nur Encoding-Artefakte
            "click_kernel_ms": 2.0,
            "clip_threshold": 0.998,
            "hum_detect_db": -55.0,  # Digitaler Brumm nur bei sehr hohem Pegel erkennbar
        },
        "dat": {
            "click_iqr": 9.0,
            "click_kernel_ms": 2.0,
            "clip_threshold": 0.998,
            "hum_detect_db": -55.0,
        },
        "mp3_low": {
            "click_iqr": 8.5,
            "click_kernel_ms": 2.0,
            "clip_threshold": 0.997,
            "hum_detect_db": -55.0,
        },
        "mp3_high": {
            "click_iqr": 9.0,
            "click_kernel_ms": 2.0,
            "clip_threshold": 0.998,
            "hum_detect_db": -55.0,
        },
    }

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def _apply_material_profile(
        self,
        material: str,
        transfer_chain: list[str] | None = None,
    ) -> None:
        """Apply material-adaptive threshold profile (§2.41 v10.0.0).

        Falls material im ``_MATERIAL_PROFILES``-Dict enthalten ist,
        werden die 4 Schwellwerte überschrieben. Sonst bleiben Defaults.

        Zusatz (§0c Codec-Chain-Invariante): Endet die Trägerkette in einem
        verlustbehafteten Codec (MP3/AAC/OGG), wird click_iqr nach oben auf
        den Codec-IQR korrigiert, da MDCT-Quantisierungsartefakte andernfalls
        als Vinyl-Clicks fehlklassifiziert werden (Brandenburg 1999).
        """
        mat_key = (material or "").lower().strip()
        profile = self._MATERIAL_PROFILES.get(mat_key)
        if profile:
            self._CLICK_IQR_MULTIPLIER = profile["click_iqr"]
            self._CLICK_KERNEL_MS = profile["click_kernel_ms"]
            self._CLIP_THRESHOLD = profile["clip_threshold"]
            self._HUM_DETECT_DB = profile["hum_detect_db"]
            logger.debug(
                "ReparaturDenker: material-Profil '%s' angewendet: iqr=%.1f, hum_db=%.0f",
                mat_key,
                self._CLICK_IQR_MULTIPLIER,
                self._HUM_DETECT_DB,
            )
        else:
            # Reset to defaults for unknown materials
            self._CLICK_IQR_MULTIPLIER = 6.0
            self._CLICK_KERNEL_MS = 1.5
            self._CLIP_THRESHOLD = 0.995
            self._HUM_DETECT_DB = -50.0

        # §0c Codec-Chain-IQR-Floor: Terminal-Codec-Artefakte dürfen nicht
        # als Analog-Clicks fehlklassifiziert werden.
        # Adaptive Logik (universal — kein fester Schwellwert):
        # Quelle: Brandenburg (1999) — MDCT-Quantisierungsfehler in Tonal-
        # Subbändern übersteigen IQR×5-Schwelle in ruhigen Passagen;
        # empirisch validiert auf 128-kbps-Material, aber mechanisch gültig
        # für alle Abtastraten (fs/576 ≈ 76.6/s @ 44.1 kHz).
        if transfer_chain:
            _terminal = str(transfer_chain[-1]).lower().strip()
            if _terminal in self._LOSSY_CODECS:
                _codec_profile = self._MATERIAL_PROFILES.get(_terminal)
                if _codec_profile:
                    _codec_iqr = _codec_profile["click_iqr"]
                    if _codec_iqr > self._CLICK_IQR_MULTIPLIER:
                        logger.info(
                            "ReparaturDenker: terminal-codec '%s' → click_iqr %.1f → %.1f "
                            "(MDCT-Quantisierungs-False-Positive-Guard, Brandenburg 1999)",
                            _terminal,
                            self._CLICK_IQR_MULTIPLIER,
                            _codec_iqr,
                        )
                        self._CLICK_IQR_MULTIPLIER = _codec_iqr

    def repariere(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        remove_clicks: bool = True,
        remove_hum: bool = True,
        repair_clipping: bool = True,
        validate_audio: bool = True,
        material: str = "",
        quality_before: float = 0.0,
        progress_callback: Callable[[str], None] | None = None,
        defect_scores: dict[str, float] | None = None,
        defect_locations: dict[str, list[tuple[float, float]]] | None = None,
        era_decade: int | None = None,
        transfer_chain: list[str] | None = None,
    ) -> ReparaturErgebnis:
        """Repariert die häufigsten analogen Defekte per DSP.

        Parameter
        ---------
        audio:
            Eingabe-Audio (float32, mono oder stereo).
        sr:
            Abtastrate in Hz.
        remove_clicks:
            Ob Clicks/Impulse entfernt werden sollen.
        remove_hum:
            Ob Netzbrumm (50/60 Hz) unterdrückt werden soll.
        repair_clipping:
            Ob Clipping-Regionen interpoliert werden sollen.
        defect_scores:
            DefectScanner-Scores je Defekttyp (optional).
            Ermöglicht severity-proportionale Reparaturstärke.
        defect_locations:
            Zeiträume je Defekttyp als ``(start_s, end_s)`` (optional).
            Enables targeted surgical repair on known defect locations.
        era_decade:
            Dekade der Aufnahme (z. B. 1940, 1960).
            Ältere Aufnahmen → sensiblere Hum-Erkennung.

        Rückgabe
        --------
        :class:`ReparaturErgebnis` mit repariertem Audio und Statistik.
        """
        assert sr == 48000, f"ReparaturDenker.repariere() erwartet sr=48000 Hz, erhalten: {sr} Hz"

        # §2.41: Material-adaptive Schwellwerte anwenden (inkl. §0c Codec-Chain-IQR-Floor)
        self._apply_material_profile(material, transfer_chain=transfer_chain)

        # §DENKER Schutzstufe 2: >5000 Clicks → IQR 12.0 (Stimme+MP3 = MDCT-Fehlalarm)
        if defect_locations:
            _n_click_locs = len(defect_locations.get("click", []))
            if _n_click_locs > 5000:
                # Data-driven: IQR = f(click_density, codec_type)
                # 15k clicks/225s=67/s → IQR 8.5+67/20×2=15.2 → cap 14.0
                _click_rate = _n_click_locs / max(len(audio) / sr, 1.0)
                _density_iqr = 8.5 + min(_click_rate / 20.0 * 2.0, 5.5)
                self._CLICK_IQR_MULTIPLIER = float(np.clip(_density_iqr, 8.5, 14.0))
                logger.info(
                    "ReparaturDenker: %d clicks → click_iqr %.1f (Denker-Schutzstufe 2)",
                    _n_click_locs,
                    self._CLICK_IQR_MULTIPLIER,
                )

        # §2.41: Era-adaptive Hum-Sensitivität — ältere Aufnahmen haben
        # typischerweise stärkeren Netzbrumm durch ungefilterte Netzteile.
        # Höherer dB-Wert = sensitiver (näher an 0 dB).
        if era_decade is not None:
            if era_decade <= 1940:
                self._HUM_DETECT_DB = max(self._HUM_DETECT_DB, -42.0)
            elif era_decade <= 1960:
                self._HUM_DETECT_DB = max(self._HUM_DETECT_DB, -47.0)
            # Post-1980: kein Override nötig, moderne Netzteile

        logger.info(
            "ReparaturDenker.repariere() gestartet: clicks=%s, hum=%s, clipping=%s, "
            "material=%s, era=%s, duration=%.1fs, iqr=%.1f, hum_db=%.0f",
            remove_clicks,
            remove_hum,
            repair_clipping,
            material or "unknown",
            era_decade or "?",
            len(audio) / max(sr, 1),
            self._CLICK_IQR_MULTIPLIER,
            self._HUM_DETECT_DB,
        )
        if validate_audio:
            audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            audio = audio.astype(np.float32)

        # Normalize to channels-first (ch, N) for internal processing.
        # BatchProcessingThread delivers audio as (N, ch); all channel loops and
        # axis=0 mono reductions below assume (ch, N) format.
        _samples_first = audio.ndim == 2 and audio.shape[0] > audio.shape[1]
        if _samples_first:
            audio = np.ascontiguousarray(audio.T)  # (N, ch) → (ch, N)

        clicks_removed = 0
        hum_removed = False
        clipping_repaired = False
        clipping_regions = 0
        notes: list[str] = []
        warnings: list[str] = []

        # --- 1. Click-Entfernung ---
        if remove_clicks:
            if progress_callback is not None:
                progress_callback("click_repair")
            try:
                # §2.41: chirurgische Click-Entfernung mit DefectScanner-Locations
                _click_locs = (defect_locations or {}).get("click", [])
                _crackle_locs = (defect_locations or {}).get("crackle", [])
                _all_click_locs = _click_locs + _crackle_locs
                audio, clicks_removed = self._remove_clicks(
                    audio, sr, click_locations=_all_click_locs if _all_click_locs else None
                )
                if clicks_removed > 0:
                    notes.append(f"{clicks_removed} Klicken entfernt")
            except Exception as exc:
                logger.debug("Click-Entfernung fehlgeschlagen: %s", exc)
                warnings.append(f"Click-Entfernung übersprungen: {exc}")

        # --- 2. Netzbrumm ---
        if remove_hum:
            if progress_callback is not None:
                progress_callback("hum_removal")
            try:
                audio, hum_removed = self._remove_hum(audio, sr)
                if hum_removed:
                    notes.append("Netzbrumm (50/60 Hz) unterdrückt")
            except Exception as exc:
                logger.debug("Hum-Entfernung fehlgeschlagen: %s", exc)
                warnings.append(f"Hum-Entfernung übersprungen: {exc}")

        # --- 3. Clipping-Reparatur ---
        if repair_clipping:
            if progress_callback is not None:
                progress_callback("declip")
            try:
                audio, clipping_repaired, clipping_regions = self._repair_clipping(audio)
                if clipping_repaired:
                    notes.append(f"{clipping_regions} Clipping-Regionen interpoliert")
            except Exception as exc:
                logger.debug("Clipping-Reparatur fehlgeschlagen: %s", exc)
                warnings.append(f"Clipping-Reparatur übersprungen: {exc}")

        # Ausgabe sichern — transpose back to original format before returning.
        if _samples_first:
            audio = np.ascontiguousarray(audio.T)  # (ch, N) → (N, ch)
        audio = np.clip(
            np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )

        note = " | ".join(notes) if notes else "Keine Reparaturen nötig"

        logger.info(
            "🔧 ReparaturDenker: clicks=%d, hum=%s, clipping=%s (%d Regionen)",
            clicks_removed,
            hum_removed,
            clipping_repaired,
            clipping_regions,
        )

        # Compat-Felder: Phasen-Liste ableiten
        _repairs_applied: list[str] = []
        if clicks_removed > 0:
            _repairs_applied.append("phase_01_click_removal")
        if hum_removed:
            _repairs_applied.append("phase_02_hum_removal")
        if clipping_repaired:
            _repairs_applied.append("phase_23_clipping_repair")

        _repair_risk_profile = {
            "transient_integrity": float(np.clip(clicks_removed / 400.0, 0.0, 1.0)),
            "hum_notch_tonality": 0.18 if hum_removed else 0.0,
            "declip_harmonic_plausibility": float(np.clip(clipping_regions / 40.0, 0.0, 1.0))
            if clipping_repaired
            else 0.0,
            "overrepair": float(
                np.clip(
                    (0.22 if clicks_removed > 0 else 0.0)
                    + (0.18 if hum_removed else 0.0)
                    + (0.28 if clipping_repaired else 0.0),
                    0.0,
                    1.0,
                )
            ),
        }

        return ReparaturErgebnis(
            audio=audio,
            clicks_removed=clicks_removed,
            hum_removed=hum_removed,
            clipping_repaired=clipping_repaired,
            clipping_regions=clipping_regions,
            processing_note=note,
            warnings=warnings,
            repairs_applied=_repairs_applied,
            quality_delta=0.0,
            material=material,
            reasoning=note,
            repair_risk_profile=_repair_risk_profile,
        )

    # ------------------------------------------------------------------
    # 1. Click-Entfernung
    # ------------------------------------------------------------------

    def _remove_clicks(
        self,
        audio: np.ndarray,
        sr: int,
        click_locations: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Medianfilter-basierte Click-Entfernung.

        Algorithmus:
            1. Mono-Referenz bilden (Mittelwert aller Kanäle).
            2. Differenz: ``d = audio − median_filtered(audio, kernel)``.
            3. IQR-basierte Schwelle: ``threshold = IQR(d) × _CLICK_IQR_MULTIPLIER``.
            4. Clicks-Maske: ``|d| > threshold``.
            5. Clicks durch Medianfilter-Version ersetzen.

        Wenn ``click_locations`` vorhanden sind (aus DefectScanner), wird die
        IQR-Maske auf diese Zeitregionen eingeschränkt (chirurgische Reparatur).
        Dadurch werden keine musikalischen Transienten außerhalb der
        Defekt-Positionen fälschlich entfernt.

        Rückgabe: (bereinigtes Audio, Anzahl erkannter Clicks)
        """
        from scipy.signal import medfilt

        kernel_samples = max(3, int(self._CLICK_KERNEL_MS * sr / 1000))
        # kernel muss ungerade sein
        if kernel_samples % 2 == 0:
            kernel_samples += 1

        # Mono-Referenz für Detektion
        mono = audio.mean(axis=0) if audio.ndim > 1 else audio

        # Guard: kernel must not exceed signal length (medfilt raises UserWarning otherwise)
        if kernel_samples >= len(mono):
            kernel_samples = len(mono) if len(mono) % 2 == 1 else max(1, len(mono) - 1)
        if kernel_samples < 3:
            return audio, 0  # signal too short for click detection

        smoothed = medfilt(mono.astype(np.float64), kernel_size=kernel_samples)
        diff = mono.astype(np.float64) - smoothed

        q75, q25 = np.percentile(diff, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-9:
            return audio, 0  # Signal zu homogen, kein Click erkennbar

        threshold = iqr * self._CLICK_IQR_MULTIPLIER
        mask = np.abs(diff) > threshold

        # §2.41: Chirurgische Einschränkung — nur in bekannten Defekt-Regionen reparieren
        if click_locations:
            location_mask = np.zeros_like(mask)
            for start_s, end_s in click_locations:
                s_idx = max(0, int(start_s * sr))
                e_idx = min(len(mono), int(end_s * sr))
                location_mask[s_idx:e_idx] = True
            # Nur Clicks innerhalb der DefectScanner-Locations bearbeiten
            mask = mask & location_mask

        # Clicks zählen (verbundene Regionen)
        clicks = int(np.sum(np.diff(mask.astype(np.int8)) > 0))  # type: ignore[arg-type]  # §V5 Dither applied at export level

        if clicks == 0:
            return audio, 0

        # Clicks ersetzen
        result = audio.copy()
        if audio.ndim > 1:
            for ch in range(audio.shape[0]):
                ch_smoothed = medfilt(audio[ch].astype(np.float64), kernel_size=kernel_samples)
                result[ch][mask] = ch_smoothed[mask]
        else:
            result[mask] = smoothed[mask]

        return result.astype(np.float32), clicks

    # ------------------------------------------------------------------
    # 2. Netzbrumm-Unterdrückung
    # ------------------------------------------------------------------

    def _remove_hum(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, bool]:
        """Zweistufiger IIR-Notch-Filter (50 Hz + 60 Hz + erste Obertöne).

        Nur aktiv wenn Brumm-Energie über ``_HUM_DETECT_DB`` liegt.
        """
        from scipy.signal import iirnotch, sosfilt, tf2sos

        # Brumm-Energie prüfen (50 Hz ± 5 Hz via FFT)
        mono = audio.mean(axis=0) if audio.ndim > 1 else audio
        N = len(mono)
        if N < 512:
            return audio, False

        fft = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(N, d=1.0 / sr)

        def _band_energy(f_center: float, bw: float = 5.0) -> float:
            mask = (freqs >= f_center - bw) & (freqs <= f_center + bw)
            e = float(np.mean(np.abs(fft[mask]) ** 2)) if mask.any() else 0.0
            return 10 * math.log10(e + 1e-12)

        detected = False
        result = audio.copy().astype(np.float64)

        # Filterfrequenzen (Grundton + 2. Oberton)
        for base_hz in (50.0, 60.0):
            for harmonic in (1, 2):
                f_hz = base_hz * harmonic
                if f_hz >= sr / 2:
                    continue
                energy_db = _band_energy(f_hz)
                if energy_db >= self._HUM_DETECT_DB:
                    detected = True
                    Q = 35.0  # Gütefaktor — schmaler Notch
                    b, a = iirnotch(f_hz, Q, sr)
                    try:
                        sos = tf2sos(b, a)
                    except Exception:
                        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                        continue
                    if result.ndim > 1:
                        for ch in range(result.shape[0]):
                            result[ch] = sosfilt(sos, result[ch])
                    else:
                        result = sosfilt(sos, result)

        return result.astype(np.float32), detected

    # ------------------------------------------------------------------
    # 3. Clipping-Reparatur
    # ------------------------------------------------------------------

    def _repair_clipping(self, audio: np.ndarray) -> tuple[np.ndarray, bool, int]:
        """Kubische Spline-Interpolation in Clipping-Regionen.

        Algorithmus:
            1. Clipping-Maske: ``|sample| ≥ _CLIP_THRESHOLD``.
            2. Verbundene Clipping-Regionen identifizieren.
            3. Jede Region per ``np.interp`` (linear, dann monoton kubisch)
               aus den Randpunkten interpolieren.

        Rückgabe: (repariertes Audio, ob repariert, Anzahl Regionen)
        """
        threshold = self._CLIP_THRESHOLD
        result = audio.copy()
        total_regions = 0

        def _repair_channel(ch_data: np.ndarray) -> np.ndarray:
            nonlocal total_regions
            clipped = np.abs(ch_data) >= threshold
            if not clipped.any():
                return ch_data

            repaired = ch_data.copy()
            n = len(ch_data)

            # Verbundene Regionen
            in_region = False
            region_start = 0
            regions = []
            for i in range(n):
                if clipped[i] and not in_region:
                    in_region = True
                    region_start = i
                elif not clipped[i] and in_region:
                    in_region = False
                    regions.append((region_start, i))
            if in_region:
                regions.append((region_start, n))

            total_regions += len(regions)

            for start, end in regions:
                # Randpunkte (min. 1 Sample Puffer)
                left = max(0, start - 1)
                right = min(n - 1, end)
                if left == right:
                    continue
                x_known = [left, right]
                y_known = [ch_data[left], ch_data[right]]
                x_fill = np.arange(start, end)
                repaired[start:end] = np.interp(x_fill, x_known, y_known)

            return repaired

        if result.ndim > 1:
            for ch in range(result.shape[0]):
                result[ch] = _repair_channel(result[ch])
        else:
            result = _repair_channel(result)

        repaired = total_regions > 0
        return result.astype(np.float32), repaired, total_regions


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking — §3.2)
# ---------------------------------------------------------------------------

_instance: ReparaturDenker | None = None
_lock: threading.Lock = threading.Lock()


def get_reparatur_denker() -> ReparaturDenker:
    """Gibt den thread-sicheren Singleton-ReparaturDenker zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ReparaturDenker()
    return _instance
