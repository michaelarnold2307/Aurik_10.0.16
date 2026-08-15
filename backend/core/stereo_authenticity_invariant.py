"""
StereoAuthenticitiyInvariant (§2.18 Spec)
=========================================

Prüft und erzwingt historisch korrekte Stereofeld-Eigenschaften basierend auf
dem EraClassifier-Ergebnis. Verhindert unhistorische Stereofeld-Manipulationen
bei Mono-Aufnahmen und frühen Stereo-Formaten.

Regeln:
    - Mono (decade ≤ 1950 oder M/S-Korrelation im Original ≥ 0.97):
      M/S-Korrelation nach Restaurierung ≥ 0.97
    - Decca-Wide-Stereo (1952–1960, LR ∈ [0.25, 0.65]):
      LR-Kreuzkorrelation nach Restaurierung ± 0.05
    - Abbey-Road-4-Kanal-Stereo (post-1967):
      Phantom-Center-Stabilität ≤ ±3° Azimutabweichung

Referenz: §2.18 Aurik-9-Spec (v10.0.0)
Autor: Aurik Development Team
Datum: 20. Februar 2026
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class StereoAuthResult:
    """Ergebnis der Stereofeld-Authentizitätsprüfung.

    Attributes:
        passed:           True wenn alle Regeln eingehalten werden.
        rule_triggered:   Name der verletzten Regel (oder "" wenn passed=True).
        ms_correlation:   Tatsächliche M/S-Korrelation ∈ [-1, 1].
        lr_cross_corr:    Tatsächliche LR-Kreuzkorrelation ∈ [0, 1].
        phantom_center_deg:  Phantom-Center-Azimut-Abweichung in Grad.
        original_type:    Erkannter Stereo-Typ ("mono", "decca_wide", "abbey_road", "modern").
        has_enforcement:  True wenn ein Korrektur-Eingriff stattfand.
        message:          Menschenlesbare Meldung auf Deutsch.
    """

    passed: bool = True
    rule_triggered: str = ""
    ms_correlation: float = 0.0
    lr_cross_corr: float = 0.0
    phantom_center_deg: float = 0.0
    original_type: str = "unknown"
    has_enforcement: bool = False
    message: str = ""


# ---------------------------------------------------------------------------
# Hilfsberechnungen
# ---------------------------------------------------------------------------


def _compute_ms_correlation(audio: np.ndarray) -> float:
    """Berechnet M/S-Korrelation (Mid-Side-Korrelation) eines Stereo-Signals.

    Args:
        audio: Stereo-Array shape (n_samples, 2) oder (2, n_samples).

    Returns:
        Korrelationskoeffizient ∈ [-1, 1]. Mono-Signal → 1.0.
    """
    if audio.ndim == 1:
        return 1.0

    if audio.shape[0] == 2 and audio.shape[1] != 2:
        left, right = audio[0].astype(np.float64), audio[1].astype(np.float64)
    elif audio.shape[-1] == 2:
        left, right = audio[:, 0].astype(np.float64), audio[:, 1].astype(np.float64)
    else:
        return 1.0

    mid = left + right
    side = left - right
    std_mid = np.std(mid)
    std_side = np.std(side)
    if std_mid < 1e-10 or std_side < 1e-10:
        return 1.0

    # Einfache Korrelation Mid und Original-L
    std_l = np.std(left)
    std_r = np.std(right)
    if std_l < 1e-10 or std_r < 1e-10:
        return 1.0
    _la = left - left.mean()
    _ra = right - right.mean()
    corr = float(np.dot(_la, _ra) / (float(np.linalg.norm(_la)) * float(np.linalg.norm(_ra)) + 1e-10))
    return float(np.clip(corr, -1.0, 1.0))


def _compute_lr_cross_correlation(audio: np.ndarray) -> float:
    """Berechnet LR-Kreuzkorrelation bei Lag 0.

    Args:
        audio: Stereo-Array.

    Returns:
        Normalisierte Korrleation ≥ 0.
    """
    if audio.ndim == 1:
        return 1.0

    if audio.shape[0] == 2 and audio.shape[1] != 2:
        left, right = audio[0].astype(np.float64), audio[1].astype(np.float64)
    elif audio.shape[-1] == 2:
        left, right = audio[:, 0].astype(np.float64), audio[:, 1].astype(np.float64)
    else:
        return 1.0

    std_l = np.std(left)
    std_r = np.std(right)
    if std_l < 1e-10 or std_r < 1e-10:
        return 1.0

    cross = float(np.mean(left * right)) / (std_l * std_r)
    return float(np.clip(abs(cross), 0.0, 1.0))


def _compute_phantom_center_azimuth_deg(audio: np.ndarray) -> float:
    """Schätzt Phantom-Center-Azimut-Abweichung in Grad.

    Perfekte Mitte → 0°.  Vollständig links → -90°.

    Args:
        audio: Stereo-Array.

    Returns:
        Azimut-Abweichung in Grad ∈ [-90, 90].
    """
    if audio.ndim == 1:
        return 0.0

    if audio.shape[0] == 2 and audio.shape[1] != 2:
        left, right = audio[0].astype(np.float64), audio[1].astype(np.float64)
    elif audio.shape[-1] == 2:
        left, right = audio[:, 0].astype(np.float64), audio[:, 1].astype(np.float64)
    else:
        return 0.0

    rms_l = float(np.sqrt(np.mean(left**2)) + 1e-10)
    rms_r = float(np.sqrt(np.mean(right**2)) + 1e-10)

    # Panning: 0 = Mitte, +1 = rechts, -1 = links
    panning = (rms_r - rms_l) / (rms_l + rms_r)
    azimuth_deg = np.degrees(np.arcsin(np.clip(panning, -1.0, 1.0)))
    return float(azimuth_deg)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class StereoAuthenticitiyInvariant:
    """Prüft und erzwingt historisch korrekte Stereofeld-Eigenschaften.

    Regeln nach EraClassifier-Ergebnis:
        Mono (decade ≤ 1950 oder Original M/S-Korrelation ≥ 0.97):
            → M/S-Korrelation nach Restaurierung ≥ 0.97
        Decca-Wide-Stereo (1952–1960, LR ∈ [0.25, 0.65]):
            → LR-Kreuzkorrelation bleibt in [0.25, 0.65] ± 0.05
        Abbey-Road-4-Kanal-Stereo (post-1967):
            → Phantom-Center-Stabilität ≤ ±3° Azimutabweichung

    Aktivierung:
        Automatisch wenn EraClassifier.confidence ≥ 0.40.
    """

    MONO_ERA_CORRELATION_THRESHOLD: float = 0.97
    DECCA_CORRELATION_RANGE: tuple = (0.25, 0.65)
    PHANTOM_CENTER_MAX_DEG: float = 3.0
    # §v10.14: Material-adaptive Toleranz — degradierte Medien haben inhärent
    # instabileres Stereofeld (Kassetten-Dropouts, Azimut-Fehler).
    PHANTOM_CENTER_MAX_DEG_RELAXED: float = 5.0
    PHANTOM_CENTER_RELAXED_MATERIALS: frozenset[str] = frozenset(
        {
            "cassette",
            "mp3_low",
            "mp3_mid",
            "aac",
            "streaming",
        }
    )
    DECCA_DECADE_START: int = 1952
    DECCA_DECADE_END: int = 1965
    ABBEY_ROAD_DECADE_START: int = 1967

    def check(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        era_result,  # EraResult from era_classifier
        sr: int,
    ) -> StereoAuthResult:
        """Prüft Stereofeld-Authentizität von Original und Restaurierung.

        Args:
            original:   Original-Audio (unrestauriert).
            restored:   Restauriertes Audio.
            era_result: EraResult-Objekt (decade, confidence).
            sr:         Sample-Rate in Hz.

        Returns:
            StereoAuthResult mit Prüfergebnis und Metriken.
        """
        original = np.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)
        restored = np.nan_to_num(restored, nan=0.0, posinf=0.0, neginf=0.0)

        result = StereoAuthResult()

        # EraResult-Felder sicher abrufen
        decade = getattr(era_result, "decade", 1970)
        confidence = getattr(era_result, "confidence", 0.0)

        if confidence < 0.40:
            result.passed = True
            result.message = "Ära-Konfidenz zu niedrig — Stereo-Prüfung übersprungen."
            return result

        # Metriken berechnen
        orig_ms_corr = _compute_ms_correlation(original)
        rest_ms_corr = _compute_ms_correlation(restored)
        orig_lr_cross = _compute_lr_cross_correlation(original)
        rest_lr_cross = _compute_lr_cross_correlation(restored)
        orig_azimuth = _compute_phantom_center_azimuth_deg(original)
        rest_azimuth = _compute_phantom_center_azimuth_deg(restored)

        result.ms_correlation = rest_ms_corr
        result.lr_cross_corr = rest_lr_cross
        result.phantom_center_deg = abs(rest_azimuth - orig_azimuth)

        # Ära-Typ bestimmen
        original_type = self._classify_stereo_type(original, decade, orig_ms_corr, orig_lr_cross)
        result.original_type = original_type

        logger.debug(
            "StereoAuth: Ära=%d Typ=%s orig_ms_corr=%.3f rest_ms_corr=%.3f LR=%.3f Azimut=%.1f°",
            decade,
            original_type,
            orig_ms_corr,
            rest_ms_corr,
            rest_lr_cross,
            result.phantom_center_deg,
        )

        # Regel 1: Mono-Ära (decade ≤ 1950 oder Original ≥ 0.97 M/S-Korrelation)
        if original_type == "mono":
            if rest_ms_corr < self.MONO_ERA_CORRELATION_THRESHOLD:
                # §v10.14: Nur kollabieren wenn das Stereofeld WIRKLICH pseudo ist.
                # rest_ms_corr < 0.97 aber > 0.80 = schmales Echtes Stereo → erhalten.
                # rest_ms_corr < 0.80 = künstlich verbreitert → kollabieren.
                if rest_ms_corr < 0.80:
                    result.passed = False
                    result.rule_triggered = "mono_era_pseudo_stereo"
                    result.message = (
                        f"Mono-Aufnahme (Ära ~{decade}) wurde unzulässig in Pseudo-Stereo "
                        f"konvertiert (M/S-Korrelation {rest_ms_corr:.3f} < 0.80)."
                    )
                    logger.warning("⚠️ StereoAuth: %s", result.message)
                    return result
                # Schmales echtes Stereo → durchlassen, nur loggen
                logger.info(
                    "StereoAuth: Schmales Stereo erkannt (ms_corr=%.3f, Ära=%d) — "
                    "kein Pseudo-Stereo, Stereofeld bleibt erhalten",
                    rest_ms_corr,
                    decade,
                )

        # Regel 2: Decca-Wide-Stereo (1952–1965, LR ∈ [0.25, 0.65])
        elif original_type == "decca_wide":
            lo, hi = self.DECCA_CORRELATION_RANGE
            tolerance = 0.05
            if not (lo - tolerance <= rest_lr_cross <= hi + tolerance):
                result.passed = False
                result.rule_triggered = "decca_wide_stereo_deviation"
                result.message = (
                    "Decca-Wide-Stereo-Fingerabdruck verändert "
                    f"(LR-Kreuzkorrelation {rest_lr_cross:.3f} außerhalb [{lo - tolerance:.2f}, {hi + tolerance:.2f}])."
                )
                logger.warning("⚠️ StereoAuth: %s", result.message)
                return result

        # Regel 3: Abbey-Road-Stereo (post-1967), Phantom-Center ≤ ±3°
        elif original_type == "abbey_road":
            azimuth_dev = abs(rest_azimuth - orig_azimuth)
            result.phantom_center_deg = azimuth_dev
            _max_deg = self.PHANTOM_CENTER_MAX_DEG
            # §v10.14: Material-adaptive Toleranz — degradierte Medien ≥5°
            _material_key = str(getattr(era_result, "material_key", "") or "").lower()
            if _material_key in self.PHANTOM_CENTER_RELAXED_MATERIALS:
                _max_deg = self.PHANTOM_CENTER_MAX_DEG_RELAXED
            if azimuth_dev > _max_deg:
                result.passed = False
                result.rule_triggered = "abbey_road_phantom_center"
                result.message = (
                    f"Abbey-Road-Stereo: Phantom-Center um {azimuth_dev:.1f}° verschoben (max. {_max_deg:.1f}°)."
                )
                logger.warning("⚠️ StereoAuth: %s", result.message)
                return result

        result.passed = True
        result.message = f"Stereofeld-Authentizität bestätigt (Typ: {original_type})."
        logger.debug("✅ StereoAuth: %s", result.message)
        return result

    def enforce(
        self,
        audio: np.ndarray,
        sr: int,
        original: np.ndarray,
        era_result,
    ) -> np.ndarray:
        """Erzwingt Stereofeld-Authentizität durch Korrektur (Letzmaßnahme).

        Bei Mono-Phase wird das Stereo-Signal auf Mono kollabiert und auf
        beiden Kanälen identisch ausgegeben.

        Args:
            audio:      Zu korrigierendes Audio.
            sr:         Sample-Rate.
            original:   Original (Referenz für Stereo-Typ).
            era_result: EraResult.

        Returns:
            Korrigiertes Audio (immer NaN/Inf-frei, geclippt auf ±1).
        """
        result = self.check(original, audio, era_result, sr)
        if result.passed:
            return np.clip(audio, -1.0, 1.0)  # type: ignore[no-any-return]

        decade = getattr(era_result, "decade", 1970)
        _compute_ms_correlation(original)

        if result.original_type == "mono":
            # Mono: Beide Kanäle identisch (Mid-Signal).
            # Guard: Wenn L+R ≈ 0 (anti-korrelierte Kanäle / phasengekehrtes Stereo),
            # würde mid = 0.5*(L+R) ≈ 0 → Stille erzeugen.
            # In diesem Fall den energiereicheren Kanal für beide Kanäle verwenden.
            if audio.ndim == 1:
                out = np.clip(audio, -1.0, 1.0)
            elif audio.shape[-1] == 2:
                # time-major (N, 2)
                ch_l, ch_r = audio[:, 0], audio[:, 1]
                mid = 0.5 * (ch_l + ch_r)
                mid_rms = float(np.sqrt(np.mean(mid**2) + 1e-12))
                orig_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
                if mid_rms < orig_rms * 0.10:
                    # Anti-Korrelation: energiereicheren Kanal als Mono verwenden
                    chosen = ch_l if np.mean(ch_l**2) >= np.mean(ch_r**2) else ch_r
                    mid = chosen
                    logger.warning(
                        "🔧 StereoAuth enforce: Anti-korrelierte Kanäle (mid_rms/orig=%.3f) — "
                        "Kanal mit höherer Energie als Mono-Basis verwendet (Ära %d)",
                        mid_rms / orig_rms,
                        decade,
                    )
                out = np.stack([mid, mid], axis=-1)
                out = np.clip(out, -1.0, 1.0)
            elif audio.shape[0] == 2:
                # channel-major (2, N)
                ch_l, ch_r = audio[0], audio[1]
                mid = 0.5 * (ch_l + ch_r)
                mid_rms = float(np.sqrt(np.mean(mid**2) + 1e-12))
                orig_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
                if mid_rms < orig_rms * 0.10:
                    # Anti-Korrelation: energiereicheren Kanal als Mono verwenden
                    chosen = ch_l if np.mean(ch_l**2) >= np.mean(ch_r**2) else ch_r
                    mid = chosen
                    logger.warning(
                        "🔧 StereoAuth enforce: Anti-korrelierte Kanäle (mid_rms/orig=%.3f) — "
                        "Kanal mit höherer Energie als Mono-Basis verwendet (Ära %d)",
                        mid_rms / orig_rms,
                        decade,
                    )
                out = np.stack([mid, mid], axis=0)
                out = np.clip(out, -1.0, 1.0)
            else:
                out = np.clip(audio, -1.0, 1.0)

            logger.info("🔧 StereoAuth enforce: Mono-Kollaps angewendet (Ära %d)", decade)
            return out  # type: ignore[no-any-return]

        # Für Decca-Wide + Abbey-Road: unverändert zurückgeben (kein Eingriff implementiert)
        logger.debug("StereoAuth enforce: Kein Eingriff für Typ '%s'", result.original_type)
        return np.clip(audio, -1.0, 1.0)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _classify_stereo_type(
        self,
        audio: np.ndarray,
        decade: int,
        ms_corr: float,
        lr_cross: float,
    ) -> str:
        """Klassifiziert den Stereo-Typ des Original-Signals."""
        # Mono erkennen
        if ms_corr >= self.MONO_ERA_CORRELATION_THRESHOLD:
            return "mono"
        # §v10.14: Nur bei tatsächlicher Mono-Quelle (ms_corr ≥ 0.95) Mono erzwingen.
        # Wenn das Original bereits klare Stereomerkmale hat (ms_corr < 0.95), ist es
        # kein Pseudo-Stereo — auch nicht bei Ära ≤ 1950 oder falscher CLAP-Ära.
        # Der ms_corr-Schwellwert ist der physikalische Fakt; der Ära-Check nur Backstop.
        if decade <= 1950 and ms_corr >= 0.95:
            return "mono"
        # Decca-Wide-Bereich
        lo, hi = self.DECCA_CORRELATION_RANGE
        if self.DECCA_DECADE_START <= decade <= self.DECCA_DECADE_END and lo <= lr_cross <= hi:
            return "decca_wide"
        # Abbey-Road (post-1967)
        if decade >= self.ABBEY_ROAD_DECADE_START:
            return "abbey_road"
        return "modern"


# ---------------------------------------------------------------------------
# Singleton (Thread-sicher, Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: StereoAuthenticitiyInvariant | None = None
_lock = threading.Lock()


def get_stereo_authenticity_invariant() -> StereoAuthenticitiyInvariant:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StereoAuthenticitiyInvariant()
    return _instance


def check_stereo_authenticity(
    original: np.ndarray,
    restored: np.ndarray,
    era_result,
    sr: int,
) -> StereoAuthResult:
    """Convenience-Funktion: Prüft Stereofeld-Authentizität.

    Args:
        original:   Original-Audio.
        restored:   Restauriertes Audio.
        era_result: EraResult-Objekt.
        sr:         Sample-Rate in Hz.

    Returns:
        StereoAuthResult mit Prüfergebnis.
    """
    return get_stereo_authenticity_invariant().check(original, restored, era_result, sr)


def _is_mono_source(audio: np.ndarray) -> bool:
    """Gibt True zurück, wenn das Audio als Mono-Quelle eingestuft wird.

    Convenience-Funktion: prüft, ob die M/S-Korrelation den Mono-Schwellwert
    (0.97) erreicht oder überschreitet — unabhängig von Ära oder Pegelkontext.

    Migriert aus ``core.stereo_authenticity`` (veraltet, wird als Shim
    weitergeführt).

    Args:
        audio: np.ndarray — (N, 2) oder (2, N) float32/64, Mono ebenfalls
               erlaubt (wird dann immer als Mono eingestuft).

    Returns:
        True  wenn M/S-Korrelation ≥ MONO_ERA_CORRELATION_THRESHOLD (0.97)
        False sonst
    """
    corr = _compute_ms_correlation(audio)
    return corr >= StereoAuthenticitiyInvariant.MONO_ERA_CORRELATION_THRESHOLD
