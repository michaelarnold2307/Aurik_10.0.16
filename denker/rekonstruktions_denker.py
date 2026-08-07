"""
RekonstruktionsDenker — Domäne: Lücken-Rekonstruktion (Dropouts).

Kapselt core.gap_reconstructor.GapReconstructor für das kontextuelle
Interpolieren von Stille-Lücken, Dropouts und kurzen Tape-Ausfällen.

Usage::

    from denker.rekonstruktions_denker import get_rekonstruktions_denker

    denker = get_rekonstruktions_denker()
    ergebnis = denker.rekonstruiere(audio, sr=48000, material_hint="tape")
    repariertes_audio = ergebnis.audio
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class RekonstruktionsErgebnis:
    """Ergebnis einer Lücken-Rekonstruktion."""

    audio: np.ndarray
    """Rekonstruiertes Audio (float32, Bereich [-1, 1])."""

    gaps_found: int
    """Anzahl erkannter Lücken/Dropouts."""

    gaps_repaired: int
    """Anzahl erfolgreich reparierter Lücken."""

    gaps_skipped: int
    """Anzahl übersprungener Lücken (zu groß / zu kurz)."""

    total_repaired_ms: float
    """Gesamte reparierte Zeitdauer in Millisekunden."""

    processing_time_ms: float
    """Verarbeitungszeit in Millisekunden."""

    detail_note: str
    """Zusammenfassung der Rekonstruktion (Deutsch)."""

    gap_details: list[Any] = field(default_factory=list)
    """Details jeder Lücke (GapInfo-Objekte)."""

    warnings: list[str] = field(default_factory=list)
    """Warnungen während der Verarbeitung."""

    # Kompatibilitäts-Alias
    gaps_filled: int = 0
    """Alias für gaps_repaired — API-Kompatibilität."""

    reconstruction_quality: float = 0.0
    """Qualität der Rekonstruktion ∈ [0, 1] (repaired / found)."""

    phases_applied: list[str] = field(default_factory=list)
    """Liste der angewandten Phasen-IDs."""

    # §11.7a ReconstructionContext fields (v10.0.0)
    bandwidth_limited: bool = False
    """True if BANDWIDTH_LOSS was detected in defect analysis."""

    estimated_original_bandwidth_hz: float = 0.0
    """Estimated original bandwidth before degradation (Hz)."""

    reconstruction_risk_profile: dict[str, float] = field(default_factory=dict)
    """Plausibilitäts-/Halluzinationsrisiko der Rekonstruktion für die zentrale Policy."""

    def as_dict(self) -> dict[str, object]:
        """Liefert alle Felder als serialisierbares Dict."""
        return {
            "gaps_found": self.gaps_found,
            "gaps_repaired": self.gaps_repaired,
            "gaps_skipped": self.gaps_skipped,
            "total_repaired_ms": self.total_repaired_ms,
            "processing_time_ms": self.processing_time_ms,
            "detail_note": self.detail_note,
            "warnings": self.warnings,
            "audio_shape": list(self.audio.shape),
        }


# ---------------------------------------------------------------------------
# RekonstruktionsDenker
# ---------------------------------------------------------------------------


class RekonstruktionsDenker:
    """Rekonstruktions-Domänendenker — orchestriert GapReconstructor.

    Invarianten
    -----------
    - Eingabe-Audio: NaN/Inf → ``nan_to_num`` vor Verarbeitung.
    - Ausgabe-Audio: immer ``np.clip(..., -1.0, 1.0)``.
    - Singleton via :func:`get_rekonstruktions_denker` (Double-Checked Locking).
    - Graceful Degradation: Bei Import-Fehler → DSP-Fallback (lineare
      Interpolation) statt Absturz.
    """

    def __init__(self) -> None:
        self._reconstructor: Any | None = None
        self._init_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def rekonstruiere(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        material: str | None = None,
        material_hint: str | None = None,
        defect_result: Any | None = None,
        repair_context: Any | None = None,
        validate_audio: bool = True,
        defect_locations: dict[str, list[tuple[float, float]]] | None = None,
        era_decade: int | None = None,
    ) -> RekonstruktionsErgebnis:
        """Erkennt und repariert Dropout-Lücken im Audio.

        Parameter
        ---------
        audio:
            Eingabe-Audio (float32, mono oder stereo).
        sr:
            Abtastrate in Hz.
        material_hint:
            Optionaler Träger-Hint (z. B. ``"tape"``, ``"vinyl"``).
        defect_result:
            Optional DefectAnalysisResult for context-aware reconstruction.
        repair_context:
            Optional ReparaturErgebnis from ReparaturDenker — provides info
            about which defects were already removed (clicks, hum, clipping)
            so reconstruction can skip redundant detection and focus on gaps.
        validate_audio:
            Ob Eingabe auf NaN/Inf geprüft werden soll.

        Rückgabe
        --------
        :class:`RekonstruktionsErgebnis` mit repariertem Audio und Statistik.
        """
        assert sr == 48000, f"RekonstruktionsDenker.rekonstruiere() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        # §11.7a: Log repair context for better reconstruction decisions
        if repair_context is not None:
            _rep_ops = getattr(repair_context, "operations_applied", [])
            logger.info(
                "RekonstruktionsDenker: repair_context vorhanden — %d Vorverarbeitungs-Operationen: %s",
                len(_rep_ops) if _rep_ops else 0,
                ", ".join(str(o) for o in _rep_ops[:5]) if _rep_ops else "keine",
            )
        logger.info(
            "RekonstruktionsDenker.rekonstruiere() gestartet: duration=%.1fs, material_hint=%s",
            len(audio) / max(sr, 1),
            material_hint or material,
        )
        if validate_audio:
            audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            audio = audio.astype(np.float32)

        # §6.4b Digital-Material Guard: digital formats have no physical transport
        # mechanism → no real dropout gaps. Any silence is intentional musical
        # content (rests, fade-outs). Codec dropouts are handled by phase_24
        # (AudioSR). Running GapReconstructor on digital material at −70 dB /
        # 0.5 ms yields thousands of false-positive "gaps" (e.g. 479 for a 3:45
        # MP3) and corrupt natural musical dynamics.
        _DIGITAL_MATERIAL_TYPES: frozenset[str] = frozenset(
            {
                "mp3_low",
                "mp3_high",
                "mp3",
                "aac",
                "cd_digital",
                "streaming",
                "wax_cylinder",  # wax_cylinder = misclassified digital
            }
        )
        _hint = (material_hint or material or "").lower()
        if _hint in _DIGITAL_MATERIAL_TYPES:
            logger.info(
                "RekonstruktionsDenker: GapReconstructor übersprungen für digitales Material '%s' "
                "— Verarbeitungsschritt_24 übernimmt Codec-Dropout-Reparatur.",
                _hint,
            )
            _no_op = RekonstruktionsErgebnis(
                audio=np.clip(audio, -1.0, 1.0),
                gaps_found=0,
                gaps_repaired=0,
                gaps_skipped=0,
                total_repaired_ms=0.0,
                processing_time_ms=0.0,
                detail_note="Kein Gap-Scan für digitales Material (§6.4b).",
                reconstruction_risk_profile={"hallucination": 0.0, "boundary_artifact": 0.0, "plausibility": 1.0},
            )
            return _no_op

        # §11.7a: Extract bandwidth context from defect analysis
        _bw_limited = False
        _est_bw_hz = 0.0
        if defect_result is not None:
            try:
                _bw_score = getattr(defect_result, "scores", {})
                # Check for BANDWIDTH_LOSS defect
                for _dt, _ds in _bw_score.items():
                    _dt_val = getattr(_dt, "value", str(_dt))
                    if _dt_val == "bandwidth_loss" and getattr(_ds, "severity", 0.0) > 0.05:
                        _bw_limited = True
                        # Extract estimated bandwidth from spectral fingerprint
                        _sf = getattr(defect_result, "spectral_fingerprint", {})
                        _est_bw_hz = float(_sf.get("effective_bandwidth_hz", 0.0))
                        logger.info(
                            "RekonstruktionsDenker: BANDWIDTH_LOSS erkannt "
                            "(severity=%.3f, est_bw=%.0f Hz) — hint passed to UV3",
                            float(getattr(_ds, "severity", 0.0)),
                            _est_bw_hz,
                        )
                        break
            except Exception as _bw_exc:
                logger.debug("Bandwidth extraction fehlgeschlagen: %s", _bw_exc)

        # §2.41: Material-adaptive GapReconstructor-Instanz
        reconstructor = self._get_reconstructor(material=material_hint or material)

        if reconstructor is None:
            result = self._dsp_fallback(audio, sr)
        else:
            try:
                raw = reconstructor.reconstruct(audio, sr, material_hint=material_hint or material)
                result = self._konvertiere(raw)
            except Exception as exc:
                logger.warning("GapReconstructor.reconstruct() fehlgeschlagen: %s — DSP-Ersatzpfad", exc)
                result = self._dsp_fallback(audio, sr, reason=str(exc))

        # §11.7a: Attach ReconstructionContext fields
        result.bandwidth_limited = _bw_limited
        result.estimated_original_bandwidth_hz = _est_bw_hz
        return result

    def erkenne_luecken(self, audio: np.ndarray, sr: int) -> list[Any]:
        """Erkennt Lücken ohne Reparatur (detect_only).

        Rückgabe
        --------
        Liste von ``GapInfo``-Objekten (leer wenn kein GapReconstructor verfügbar).
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        reconstructor = self._get_reconstructor()
        if reconstructor is None:
            return []
        try:
            detected_raw: object = reconstructor.detect_only(audio, sr)
            if detected_raw is None:
                return []
            if isinstance(detected_raw, list):
                return detected_raw
            if isinstance(detected_raw, Iterable):
                return list(detected_raw)
            logger.debug("erkennen_only lieferte unerwarteten Typ: %s", type(detected_raw).__name__)
            return []
        except Exception as exc:
            logger.debug("erkennen_only fehlgeschlagen: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    # Material-adaptive GapReconstructor-Konfigurationen (§2.41 v10.0.0)
    # Copeland 2008: Shellac-Dropouts kürzer (0.3 ms) und lauter als Tape-Dropouts.
    # Tape-Dropouts sind typischerweise länger (bis 2 s) und gradueller.
    _MATERIAL_GAP_CONFIGS: dict[str, dict[str, float]] = {
        "shellac": {
            "silence_threshold_db": -55.0,  # Shellac: höherer Grundrausch → höhere Schwelle
            "min_gap_duration_ms": 0.3,  # Sehr kurze Nadelsprünge
            "max_gap_duration_ms": 200.0,  # Physische Nadelsprünge selten > 200 ms
            "blend_ms": 1.0,  # Kurzes Blending, da scharfe Kanten
        },
        "wax_cylinder": {
            "silence_threshold_db": -50.0,
            "min_gap_duration_ms": 0.3,
            "max_gap_duration_ms": 300.0,
            "blend_ms": 1.0,
        },
        "vinyl": {
            "silence_threshold_db": -65.0,  # Vinyl: mittlerer Grundrausch
            "min_gap_duration_ms": 0.5,
            "max_gap_duration_ms": 400.0,
            "blend_ms": 1.5,
        },
        "tape": {
            "silence_threshold_db": -70.0,  # Tape: niedrigerer Grundrausch
            "min_gap_duration_ms": 1.0,  # Tape-Dropouts sind gradueller
            "max_gap_duration_ms": 2000.0,  # Tape-Dropouts können bis 2 s dauern
            "blend_ms": 2.5,  # Längeres Blending für natürlicheren Übergang
        },
        "reel_tape": {
            "silence_threshold_db": -72.0,  # Pro-Tape: sehr niedrig
            "min_gap_duration_ms": 1.0,
            "max_gap_duration_ms": 2000.0,
            "blend_ms": 2.5,
        },
        "cassette": {
            "silence_threshold_db": -65.0,
            "min_gap_duration_ms": 0.5,
            "max_gap_duration_ms": 1000.0,  # Kassette: Band-Dropouts bis 1 s
            "blend_ms": 2.0,
        },
    }

    def _get_reconstructor(self, material: str | None = None) -> Any | None:
        """Lädt GapReconstructor lazy mit material-adaptiver Config."""
        # Material-adaptive Config: immer frische Instanz mit passenden Schwellwerten
        if material and material.lower() in self._MATERIAL_GAP_CONFIGS:
            return self._build_reconstructor(material=material)
        if self._reconstructor is None:
            with self._init_lock:
                if self._reconstructor is None:
                    self._reconstructor = self._build_reconstructor()
        return self._reconstructor

    def _build_reconstructor(self, *, material: str | None = None) -> Any | None:
        """Instantiiert GapReconstructor mit material-adaptiver Config (§2.41)."""
        try:
            from backend.core.gap_reconstructor import GapReconstructor, GapReconstructorConfig

            # Material-adaptive Schwellwerte
            mat_key = (material or "").lower().strip()
            mat_cfg = self._MATERIAL_GAP_CONFIGS.get(mat_key)
            if mat_cfg:
                cfg = GapReconstructorConfig(
                    silence_threshold_db=mat_cfg["silence_threshold_db"],
                    min_gap_duration_ms=mat_cfg["min_gap_duration_ms"],
                    max_gap_duration_ms=mat_cfg["max_gap_duration_ms"],
                    ar_stabilize=True,
                    blend_ms=mat_cfg["blend_ms"],
                )
                logger.info(
                    "RekonstruktionsDenker: material-adaptive GapConfig '%s' — "
                    "silence=%.0f dB, min_gap=%.1f ms, max_gap=%.0f ms, blend=%.1f ms",
                    mat_key,
                    cfg.silence_threshold_db,
                    cfg.min_gap_duration_ms,
                    cfg.max_gap_duration_ms,
                    cfg.blend_ms,
                )
            else:
                # Default-Config (bisheriges Verhalten)
                cfg = GapReconstructorConfig(
                    silence_threshold_db=-70.0,
                    min_gap_duration_ms=0.5,
                    max_gap_duration_ms=500.0,
                    ar_stabilize=True,
                    blend_ms=1.5,
                )
            return GapReconstructor(config=cfg)
        except Exception as exc:
            logger.warning("GapReconstructor konnte nicht geladen werden: %s — DSP-Ersatzpfad aktiv", exc)
            return None

    def _konvertiere(self, raw: Any) -> RekonstruktionsErgebnis:
        """Wandelt GapReconstructionResult in RekonstruktionsErgebnis um."""
        out_audio = np.array(raw.audio, dtype=np.float32)
        out_audio = np.nan_to_num(out_audio, nan=0.0, posinf=0.0, neginf=0.0)
        out_audio = np.clip(out_audio, -1.0, 1.0)

        repaired = int(raw.gaps_repaired)
        found = int(raw.gaps_found)
        skipped = int(raw.gaps_skipped)
        total_ms = float(raw.total_repaired_ms) if math.isfinite(float(raw.total_repaired_ms)) else 0.0
        proc_ms = float(raw.processing_time_ms) if math.isfinite(float(raw.processing_time_ms)) else 0.0

        if repaired == 0 and found == 0:
            note = "Keine Lücken gefunden — kein Eingriff nötig"
        elif repaired == found:
            note = f"{repaired} Lücken vollständig rekonstruiert ({total_ms:.1f} ms gesamt)"
        else:
            note = f"{repaired} von {found} Lücken rekonstruiert, {skipped} übersprungen ({total_ms:.1f} ms gesamt)"

        logger.info(
            "🧩 RekonstruktionsDenker: %d/%d Lücken repariert (%.1f ms)",
            repaired,
            found,
            total_ms,
        )

        quality = float(repaired) / max(float(found), 1.0)
        quality = max(0.0, min(1.0, quality))

        phases: list[str] = []
        try:
            phases = list(getattr(raw, "phases_applied", []) or [])
        except Exception as phase_exc:
            logger.debug(
                "RekonstruktionsDenker: phases_angewendet nicht lesbar (Ursache: %s). "
                "Lösung: GapReconstructor-Rückgabeformat prüfen.",
                phase_exc,
            )
        if not phases and repaired > 0:
            phases = ["phase_24_dropout_repair", "phase_55_diffusion_inpainting"]

        return RekonstruktionsErgebnis(
            audio=out_audio,
            gaps_found=found,
            gaps_repaired=repaired,
            gaps_skipped=skipped,
            total_repaired_ms=total_ms,
            processing_time_ms=proc_ms,
            detail_note=note,
            gap_details=list(raw.gap_details or []),
            gaps_filled=repaired,
            reconstruction_quality=quality,
            phases_applied=phases,
            reconstruction_risk_profile={
                "hallucination": float(max(0.0, 1.0 - quality)),
                "boundary_artifact": float(min(1.0, skipped / max(found, 1))),
                "plausibility": float(quality),
                "bandwidth_uncertainty": 0.25 if total_ms > 0.0 else 0.0,
            },
        )

    def _dsp_fallback(
        self,
        audio: np.ndarray,
        sr: int,
        reason: str = "GapReconstructor nicht verfügbar",
    ) -> RekonstruktionsErgebnis:
        """Minimaler DSP-Fallback: lineare Interpolation von Stille-Lücken.

        Erkennt Stille-Segmente (< -70 dBFS, ≥ 50 Samples) und füllt
        sie mit linearer Interpolation aus den Randpunkten.
        """
        SILENCE_THRESHOLD = 10 ** (-70.0 / 20)  # -70 dBFS
        MIN_GAP_SAMPLES = 50

        audio_out = audio.copy()
        gap_count = 0

        def _fill_channel(ch_data: np.ndarray) -> np.ndarray:
            nonlocal gap_count
            n = len(ch_data)
            result = ch_data.copy()
            silent = np.abs(ch_data) < SILENCE_THRESHOLD
            in_gap = False
            gap_start = 0
            for i in range(n):
                if silent[i] and not in_gap:
                    in_gap = True
                    gap_start = i
                elif not silent[i] and in_gap:
                    in_gap = False
                    gap_len = i - gap_start
                    if gap_len >= MIN_GAP_SAMPLES:
                        left = max(0, gap_start - 1)
                        right = min(n - 1, i)
                        x_fill = np.arange(gap_start, i)
                        y = np.interp(x_fill, [left, right], [ch_data[left], ch_data[right]])
                        result[gap_start:i] = y
                        gap_count += 1
            return result

        if audio_out.ndim > 1:
            for ch in range(audio_out.shape[0]):
                audio_out[ch] = _fill_channel(audio_out[ch])
        else:
            audio_out = _fill_channel(audio_out)

        audio_out = np.clip(audio_out, -1.0, 1.0).astype(np.float32)

        quality = 1.0 if gap_count == 0 else 0.65

        return RekonstruktionsErgebnis(
            audio=audio_out,
            gaps_found=gap_count,
            gaps_repaired=gap_count,
            gaps_skipped=0,
            total_repaired_ms=0.0,
            processing_time_ms=0.0,
            detail_note=f"DSP-Fallback (lineare Interpolation): {gap_count} Lücken",
            warnings=[f"GapReconstructor nicht verfügbar — DSP-Fallback: {reason}"],
            gaps_filled=gap_count,
            reconstruction_quality=quality,
            phases_applied=["dsp_linear_interpolation"],
            reconstruction_risk_profile={
                "hallucination": 0.0,
                "boundary_artifact": 0.35 if gap_count else 0.0,
                "plausibility": float(quality),
            },
        )


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking — §3.2)
# ---------------------------------------------------------------------------

_instance: RekonstruktionsDenker | None = None
_lock: threading.Lock = threading.Lock()


def get_rekonstruktions_denker() -> RekonstruktionsDenker:
    """Gibt den thread-sicheren Singleton-RekonstruktionsDenker zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RekonstruktionsDenker()
    return _instance
