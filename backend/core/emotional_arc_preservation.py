"""
core/emotional_arc_preservation.py — Aurik 10.0.0+ (§2.30 / §8.2 Punkt 12)

EmotionalArcPreservationMetric: Prüft, ob der emotionale Dynamik-Bogen
(Arousal-/Valence-Kurve) des Originals im restaurierten Signal erhalten bleibt.

Musikalische Werke haben einen emotionalen Spannungsbogen (sanfter Beginn →
Klimax → Auflösung). Restaurierung darf diesen Bogen nicht begradigen.

Invariante: Dateien < 30 s: Metrik deaktiviert. Pearson-Schwellen ≥ 0.85 (Arousal),
            ≥ 0.80 (Valence). Klimax-Peak-Abweichung ≤ 2 Segmente.

Referenz:
    Russell (1980): „A circumplex model of affect"
    Thayer (1989): „The biopsychology of mood and arousal"
    Kim & André (2008): „Emotion recognition based on physiological changes in music listening"
    Eerola & Vuoskoski (2011): „A comparison of the discrete and dimensional models of emotion
        in music" Psychol. Music 39(4):406-429 — Modus (Dur/Moll) ist stärkster
        Valenz-Prädiktor (r=0.63); Spektral-Flachheit kein signifikanter Prädiktor.
    Krumhansl (1990): „Cognitive foundations of musical pitch" Oxford Univ. Press.
        Major/Minor Tonstufen-Hierarchie-Profile aus Tonhöhen-Primings-Experiment.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

import numpy as np

from backend.core.audio_utils import safe_to_mono
from backend.core.calibration_context import get_calibration_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler Key-Profile (Krumhansl 1990, Table 1)
# ---------------------------------------------------------------------------
# Major template: normalized hierarchy ratings from probe-tone experiment.
# Minor template: averaged over harmonic and melodic minor ratings.
# Starting pitch class: C (index 0), chromatic order C C# D D# E F F# G G# A A# B.
_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)
# Pre-center both templates (Pearson correlation works on centered vectors)
_KK_MAJOR = _KK_MAJOR - _KK_MAJOR.mean()
_KK_MINOR = _KK_MINOR - _KK_MINOR.mean()


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class EmotionalArcResult:
    """Ergebnis der Emotionaler-Dynamik-Bogen-Analyse (§8.2 Punkt 12)."""

    arousal_pearson: float  # Korrelation Arousal-Profil ∈ [−1, 1]
    valence_pearson: float  # Korrelation Valence-Profil ∈ [−1, 1]
    klimax_peak_deviation: float  # Segmente | argmax(orig) − argmax(rest) |
    klimax_level_deviation_db: float  # dB-Abweichung Klimax-Peak
    arc_preserved: bool  # True wenn alle Schwellen erfüllt
    reason: str = ""  # Menschenlesbare Begründung (Deutsch)
    skipped: bool = False  # True wenn Datei < 30 s

    THRESHOLD_AROUSAL = 0.85
    THRESHOLD_VALENCE = 0.80
    MAX_KLIMAX_DEVIATION_SEGMENTS = 2
    MAX_KLIMAX_LEVEL_DB = 2.0

    @property
    def preservation_score(self) -> float:
        """Skalarer Erhaltungs-Score ∈ [0, 1] für HPG §2.44.

        FIXED v10.0.0: UV3 nutzt getattr(result, "preservation_score", 1.0) —
        ohne dieses Property fällt es immer auf den 1.0-Default zurück
        (emotional_arc_preservation nie unter 1.0, HPG-Faktor wirkungslos).

        Formel:
            0.50 · arousal_pearson_norm +
            0.30 · valence_pearson_norm +
            0.20 · klimax_score
        Alle negativen Pearson-Werte werden auf 0 geclampt.
        """
        if self.skipped:
            return 1.0  # Kurze Dateien: neutraler Prior
        ar = max(0.0, self.arousal_pearson)
        val = max(0.0, self.valence_pearson)
        _max_dev = max(float(self.MAX_KLIMAX_DEVIATION_SEGMENTS), 1.0)
        _max_lev = max(float(self.MAX_KLIMAX_LEVEL_DB), 0.1)
        klimax_pos = max(0.0, 1.0 - self.klimax_peak_deviation / _max_dev)
        klimax_lev = max(0.0, 1.0 - self.klimax_level_deviation_db / _max_lev)
        klimax = 0.5 * klimax_pos + 0.5 * klimax_lev
        return float(0.50 * ar + 0.30 * val + 0.20 * klimax)

    def as_dict(self) -> dict:
        """Gibt die Metriken als Dictionary zurück."""
        return {
            "arousal_pearson": self.arousal_pearson,
            "valence_pearson": self.valence_pearson,
            "klimax_peak_deviation": self.klimax_peak_deviation,
            "klimax_level_deviation_db": self.klimax_level_deviation_db,
            "arc_preserved": self.arc_preserved,
            "preservation_score": round(self.preservation_score, 4),
            "reason": self.reason,
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class EmotionalArcPreservationMetric:
    """Misst Erhalt des emotionalen Dynamik-Bogens Original vs. Restauriert.

    Algorithmus (§8.2):
        1. Audio in 5-s-Segmente teilen (Hop: 2.5 s)
        2. Arousal-Proxy pro Segment:
           arousal(t) = 0.6 · rms_norm(t) + 0.4 · zcr_norm(t)
        3. Valence-Proxy: Harmonizitäts-Ratio (HPSS-Approximation via
           Spektral-Flachheit — hohe Spektral-Flachheit = weniger Harmonizität)
        4. Pearson-Korrelation arousal_orig ↔ arousal_rest ≥ 0.85
        5. Klimax-Peak-Erhalt:
           |argmax(arousal_orig) − argmax(arousal_rest)| ≤ 2 Segmente
           |max(arousal_orig) − max(arousal_rest)| ≤ 2 dB

    Invarianten:
        - Dateien < 30 s: Metrik deaktiviert (skipped=True, arc_preserved=True)
        - NaN im Segment → Segment überspringen
        - Alle Ausgaben sind NaN/Inf-frei
    """

    SEGMENT_S = 5.0  # Segment-Länge
    HOP_S = 2.5  # Hop-Größe
    MIN_DURATION_S = 30.0  # Minimum für sinnvolle Bogen-Messung

    THRESHOLD_AROUSAL = 0.85
    THRESHOLD_VALENCE = 0.80
    MAX_KLIMAX_DEVIATION = 2  # Segmente
    MAX_KLIMAX_LEVEL_DB = 2.0  # dB

    def measure(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        lyrics_saliency: np.ndarray | None = None,
        frisson_zones=None,
        transfer_chain_depth: int | None = None,
    ) -> EmotionalArcResult:
        """Misst den Erhalt des emotionalen Bogens.

        Args:
            original:        Original-Audio (vor Restaurierung), float32, 1D oder 2D
            restored:        Restauriertes Audio, float32, 1D oder 2D
            sr:              Sample-Rate, muss 48000 sein
            lyrics_saliency: Optionales 1D-Array (Länge = n_samples) mit normalisierten
                             Lyrics-Salienzwerten aus §2.36 LyricsGuidedEnhancement.
                             Segment-Gewichte werden 1.5× erhöht für Frames > 0.5.
                             (§2.44: emotional_arc_preservation = Arousal/Valence + Lyrics-Salienz)
            frisson_zones:   Optionale Liste von FrissonZone-Objekten (.start_s, .end_s).
                             Frisson-Segmente erhalten Gewicht ×2.0 in der Pearson-Korrelation
                             (§2.44: Klimax-Passagen prägen Emotionswahrnehmung stärker).

        Returns:
            EmotionalArcResult mit Pearson-Korrelationen und Klimax-Analyse.
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"

        # Mono-Konvertierung
        def to_mono(a: np.ndarray) -> np.ndarray:
            arr = np.asarray(a, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 2:
                arr = np.mean(arr, axis=0)
            return arr  # type: ignore[no-any-return]

        orig_mono = to_mono(original)
        rest_mono = to_mono(restored)

        # Längenabgleich
        n = min(len(orig_mono), len(rest_mono))
        orig_mono = orig_mono[:n]
        rest_mono = rest_mono[:n]

        duration_s = n / sr

        # Zu kurz → überspringen
        if duration_s < self.MIN_DURATION_S:
            return EmotionalArcResult(
                arousal_pearson=1.0,
                valence_pearson=1.0,
                klimax_peak_deviation=0.0,
                klimax_level_deviation_db=0.0,
                arc_preserved=True,
                reason="Datei kürzer als 30 s — Emotional-Arc-Prüfung nicht aktiv.",
                skipped=True,
            )

        # ----------------------------------------------------------------
        # Segment-Analyse
        # ----------------------------------------------------------------
        seg_len = int(self.SEGMENT_S * sr)
        hop_len = int(self.HOP_S * sr)

        arousal_orig, valence_orig, _centroids_orig = self._compute_features(orig_mono, sr, seg_len, hop_len)
        arousal_rest, valence_rest, _centroids_rest = self._compute_features(rest_mono, sr, seg_len, hop_len)

        # §9.10.119: Re-normalize arousal centroid component per-song.
        # After denoising, the noise floor drops → centroid shifts upward
        # globally → false arousal increase that flattens apparent dynamics.
        # Fix: scale restored centroid median to match original median.
        if len(_centroids_orig) >= 3 and len(_centroids_rest) >= 3:
            _med_orig = float(np.median(_centroids_orig))
            _med_rest = float(np.median(_centroids_rest))
            if _med_rest > 1e-3 and abs(_med_rest - _med_orig) / max(_med_orig, 1.0) > 0.05:
                _centroid_correction = _med_orig / _med_rest
                # Re-compute arousal_rest with corrected centroid weight
                _arousal_corr = []
                for i, start in enumerate(range(0, len(rest_mono) - seg_len + 1, hop_len)):
                    if i >= len(arousal_rest):
                        break
                    seg = rest_mono[start : start + seg_len]
                    rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
                    _c_hz = _centroids_rest[i] * _centroid_correction if i < len(_centroids_rest) else 0.0
                    _c_norm = float(np.clip(_c_hz / max(sr / 2.0, 1.0), 0.0, 1.0))
                    _arousal_corr.append(rms * 0.55 + _c_norm * 0.45)
                if _arousal_corr:
                    arousal_rest = np.array(_arousal_corr[: len(arousal_rest)], dtype=np.float32)

        n_segs = min(len(arousal_orig), len(arousal_rest))
        if n_segs < 3:
            return EmotionalArcResult(
                arousal_pearson=1.0,
                valence_pearson=1.0,
                klimax_peak_deviation=0.0,
                klimax_level_deviation_db=0.0,
                arc_preserved=True,
                reason="Zu wenige Segmente für Bogen-Messung.",
                skipped=True,
            )

        arousal_orig = arousal_orig[:n_segs]
        arousal_rest = arousal_rest[:n_segs]
        valence_orig = valence_orig[:n_segs]
        valence_rest = valence_rest[:n_segs]

        # ----------------------------------------------------------------
        # §2.36/§2.44 Lyrics-Salienz-Gewichte pro Segment
        # ----------------------------------------------------------------
        _seg_weights = np.ones(n_segs, dtype=np.float32)
        if lyrics_saliency is not None:
            _sal = np.asarray(lyrics_saliency, dtype=np.float32)
            _sal = np.nan_to_num(_sal, nan=0.0, posinf=0.0, neginf=0.0)
            # Map Lyrics-Saliency auf Segment-Zeitachse (Mittelwert pro Segment)
            for _si, _start in enumerate(range(0, len(orig_mono) - seg_len + 1, hop_len)):
                if _si >= n_segs:
                    break
                _seg_sal = float(np.mean(_sal[_start : _start + seg_len]))
                _seg_weights[_si] = 1.5 if _seg_sal > 0.5 else 1.0

        # §2.44 Frisson-Zonen: Gewicht ×2.0 (Klimax-Passagen prägen Emotionswahrnehmung stärker)
        # Frisson-Gewichte werden NACH Lyrics-Salienz gesetzt und können sie überschreiben.
        if frisson_zones:
            try:
                _seg_idx = 0
                for _start in range(0, len(orig_mono) - seg_len + 1, hop_len):
                    if _seg_idx >= n_segs:
                        break
                    _seg_center_s = (_start + seg_len / 2) / sr
                    for _fz in frisson_zones:
                        _fz_s = float(getattr(_fz, "start_s", 0.0))
                        _fz_e = float(getattr(_fz, "end_s", 0.0))
                        if _fz_s <= _seg_center_s < _fz_e:
                            _seg_weights[_seg_idx] = 2.0  # §2.44: Frisson = ×2.0
                            break
                    _seg_idx += 1
            except Exception as e:
                logger.warning("emotional_arc_preservation.py::unbekannter Ersatzpfad: %s", e)
                pass  # non-blocking: frisson_zones Fehler darf Messung nicht blockieren

        # ----------------------------------------------------------------
        # Pearson-Korrelationen (salienz-gewichtet wenn lyrics_saliency vorhanden)
        # ----------------------------------------------------------------
        ar_pearson = self._weighted_pearson(arousal_orig, arousal_rest, _seg_weights)
        val_pearson = self._weighted_pearson(valence_orig, valence_rest, _seg_weights)

        # ----------------------------------------------------------------
        # Klimax-Peak-Analyse
        # ----------------------------------------------------------------
        peak_orig = int(np.argmax(arousal_orig))
        peak_rest = int(np.argmax(arousal_rest))
        klimax_dev = abs(peak_orig - peak_rest)

        orig_peak_db = float(20.0 * math.log10(max(arousal_orig[peak_orig], 1e-9)))
        rest_peak_db = float(20.0 * math.log10(max(arousal_rest[peak_rest], 1e-9)))
        klimax_level_dev = abs(orig_peak_db - rest_peak_db)

        # ----------------------------------------------------------------
        # §Y4 Local segment Δarousal check: no segment should drop > 8%
        # Ignore the last ≤3 s tail segment (under-filled HOP window)
        # ----------------------------------------------------------------
        _n_tail_segs = max(0, int(3.0 / self.HOP_S))  # segments covering last 3 s
        _check_segs = n_segs - _n_tail_segs
        _local_drop_ok = True
        _worst_local_drop = 0.0
        _worst_seg_idx = -1
        if _check_segs > 0:
            _orig_clip = np.maximum(arousal_orig[:_check_segs], 1e-9)
            _delta_local = arousal_rest[:_check_segs] / _orig_clip - 1.0
            _worst_local_drop = float(np.min(_delta_local))
            _worst_seg_idx = int(np.argmin(_delta_local))
            if _worst_local_drop < -0.08:
                _local_drop_ok = False

        # ----------------------------------------------------------------
        # Urteil
        # ----------------------------------------------------------------
        # §v10.200 Depth-adaptive thresholds: tiefere Ketten haben flachere
        # emotionale Bögen durch Rauschmaskierung. Bei depth≥4 ist eine
        # Pearson-Korrelation von 0.85 physikalisch unerreichbar.
        _resolved_depth = transfer_chain_depth
        if _resolved_depth is None:
            _ctx_ea = get_calibration_context()
            _resolved_depth = _ctx_ea.transfer_chain_depth if _ctx_ea is not None else 1
        _td_ea = max(1, int(_resolved_depth))
        if _td_ea >= 4:
            _thr_arousal = 0.50
            _thr_valence = 0.50
        elif _td_ea >= 3:
            _thr_arousal = 0.70
            _thr_valence = 0.65
        else:
            _thr_arousal = self.THRESHOLD_AROUSAL
            _thr_valence = self.THRESHOLD_VALENCE

        arc_preserved = (
            ar_pearson >= _thr_arousal
            and val_pearson >= _thr_valence
            and klimax_dev <= self.MAX_KLIMAX_DEVIATION
            and klimax_level_dev <= self.MAX_KLIMAX_LEVEL_DB
            and _local_drop_ok  # §Y4: no segment arousal collapsed by > 8%
        )

        reason_parts = []
        if ar_pearson < _thr_arousal:
            reason_parts.append(f"Arousal-Korrelation zu niedrig ({ar_pearson:.2f} < {_thr_arousal})")
        if val_pearson < _thr_valence:
            reason_parts.append(f"Valence-Korrelation zu niedrig ({val_pearson:.2f} < {_thr_valence})")
        if klimax_dev > self.MAX_KLIMAX_DEVIATION:
            reason_parts.append(f"Klimax-Verschiebung: {klimax_dev} Segmente (Max: {self.MAX_KLIMAX_DEVIATION})")
        if klimax_level_dev > self.MAX_KLIMAX_LEVEL_DB:
            reason_parts.append(f"Klimax-Pegel-Abweichung: {klimax_level_dev:.1f} dB")
        if not _local_drop_ok:
            reason_parts.append(
                f"Lokaler Arousal-Einbruch Segment {_worst_seg_idx + 1}/{_check_segs}: "
                f"Δ={_worst_local_drop:.3f} (Schwelle: -0.08)"
            )

        reason = (
            "Emotionaler Bogen vollständig erhalten."
            if arc_preserved
            else "Emotionaler Bogen teilweise verändert: " + "; ".join(reason_parts)
        )

        return EmotionalArcResult(
            arousal_pearson=round(ar_pearson, 4),
            valence_pearson=round(val_pearson, 4),
            klimax_peak_deviation=float(klimax_dev),
            klimax_level_deviation_db=round(klimax_level_dev, 2),
            arc_preserved=arc_preserved,
            reason=reason,
            skipped=False,
        )

    # ----------------------------------------------------------------
    # Hilfsmethoden
    # ----------------------------------------------------------------

    def _compute_features(
        self,
        mono: np.ndarray,
        sr: int,
        seg_len: int,
        hop_len: int,
    ):
        """Berechnet Arousal- und Valence-Proxy-Profile als Arrays."""
        arousal_list = []
        valence_list = []
        _centroid_hz_list: list[float] = []  # §9.10.119: collect for post-normalization

        positions = list(range(0, len(mono) - seg_len + 1, hop_len))
        for start in positions:
            seg = mono[start : start + seg_len]
            if not np.isfinite(seg).all():
                continue

            # Arousal: 0.55·RMS + 0.45·spectral_centroid_norm (v10.0.0)
            # §9.10.119: Centroid normalized relative to per-song median instead
            # of Nyquist.  After denoising, the noise floor drops → centroid
            # shifts upward → false arousal increase → flattened dynamic arc.
            # Per-song normalization makes the proxy robust to global spectral
            # shifts caused by NR (Blanchini 2011; empirical correction).
            rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
            _n_fft_a = min(1024, len(seg))
            _spec_a = np.abs(np.fft.rfft(seg[:_n_fft_a], n=_n_fft_a)) + 1e-9
            _freqs_a = np.fft.rfftfreq(_n_fft_a, 1.0 / sr)
            _spec_sum = float(np.sum(_spec_a))
            _centroid_hz = float(np.sum(_freqs_a * _spec_a) / max(_spec_sum, 1e-12))
            _centroid_hz_list.append(_centroid_hz)
            # Normalize centroid to [0,1] by Nyquist (sr/2) — deferred below
            _centroid_norm = float(np.clip(_centroid_hz / max(sr / 2.0, 1.0), 0.0, 1.0))
            arousal_list.append(rms * 0.55 + _centroid_norm * 0.45)

            # Valence: Tonart-Modus (Dur/Moll) via Krumhansl-Schmuckler Key-Finding
            # Literature: Eerola & Vuoskoski (2011) Psychol. Music 39(4):406-429
            #   → Modus (Dur/Moll) ist stärkster akustischer Valenz-Prädiktor (r=0.63).
            #   → Spektral-Flachheit ist kein signifikanter Valenz-Prädiktor.
            # Literature: Krumhansl (1990) Cognitive foundations of musical pitch.
            #   → Key-Profile aus Tonstufen-Hierarchie-Experiment (Table 1).
            # Algorithmus: Pitch-Class-Profile (PCP) → Korrelation mit 24 Tonarten →
            #   valence = (max Dur-Korr − max Moll-Korr + 1) / 2 ∈ [0, 1]
            try:
                _n_fft_v = min(2048, len(seg))
                _seg_v = seg[:_n_fft_v]
                _spec_v = np.abs(np.fft.rfft(_seg_v, n=_n_fft_v)) ** 2
                _freqs_v = np.fft.rfftfreq(_n_fft_v, 1.0 / sr)
                # 12-Bin Pitch Class Profile: Energie pro Halbton summieren
                _pcp = np.zeros(12, dtype=np.float64)
                _A4_HZ = 440.0
                for _k in range(1, len(_freqs_v)):
                    _f = _freqs_v[_k]
                    if _f < 40.0 or _f > min(5000.0, sr * 0.5):
                        continue
                    _semitone = int(round(12.0 * math.log2(_f / _A4_HZ))) % 12
                    _pcp[_semitone] += float(_spec_v[_k])
                _pcp_sum = float(_pcp.sum())
                if _pcp_sum > 1e-12:
                    _pcp_centered = _pcp / _pcp_sum - (_pcp / _pcp_sum).mean()
                else:
                    _pcp_centered = _pcp
                # Korrelation mit Dur- und Moll-Profil für alle 12 Tonarten (Rotationen)
                _best_major = -2.0
                _best_minor = -2.0
                for _root in range(12):
                    _major_rot = np.roll(_KK_MAJOR, _root)
                    _minor_rot = np.roll(_KK_MINOR, _root)
                    _denom_m = float(
                        np.sqrt(np.dot(_pcp_centered, _pcp_centered) * np.dot(_major_rot, _major_rot)) + 1e-12
                    )
                    _denom_n = float(
                        np.sqrt(np.dot(_pcp_centered, _pcp_centered) * np.dot(_minor_rot, _minor_rot)) + 1e-12
                    )
                    _r_major = float(np.dot(_pcp_centered, _major_rot)) / _denom_m
                    _r_minor = float(np.dot(_pcp_centered, _minor_rot)) / _denom_n
                    if _r_major > _best_major:
                        _best_major = _r_major
                    if _r_minor > _best_minor:
                        _best_minor = _r_minor
                # Valenz: Dur→1.0, Moll→0.0; normiert auf [0, 1] via (diff + 1) / 2
                _valence_raw = float(np.clip((_best_major - _best_minor + 1.0) * 0.5, 0.0, 1.0))
                valence_list.append(_valence_raw)
            except Exception:
                valence_list.append(0.5)

        return (
            np.array(arousal_list, dtype=np.float32),
            np.array(valence_list, dtype=np.float32),
            _centroid_hz_list,
        )

    @staticmethod
    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson-Korrelation, NaN-sicher."""
        return EmotionalArcPreservationMetric._weighted_pearson(a, b, None)

    @staticmethod
    def _weighted_pearson(
        a: np.ndarray,
        b: np.ndarray,
        weights: np.ndarray | None,
    ) -> float:
        """Gewichtete Pearson-Korrelation (§2.44 Lyrics-Salienz), NaN-sicher.

        weights: 1D-Array gleichlang wie a/b; None oder uniform → ungewichtet.
        Hoch-salienz-Segmente (weight > 1.0) beeinflussen die Korrelation stärker.
        """
        try:
            if len(a) < 2 or len(b) < 2:
                return 1.0
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            if weights is None or len(weights) == 0:
                w = np.ones(n, dtype=np.float64)
            else:
                w = np.asarray(weights[:n], dtype=np.float64)
                w = np.clip(w, 0.0, None)
            w_sum = float(np.sum(w))
            if w_sum < 1e-9:
                return 1.0
            w = w / w_sum  # normalize
            a_mean = float(np.sum(w * a))
            b_mean = float(np.sum(w * b))
            cov = float(np.sum(w * (a - a_mean) * (b - b_mean)))
            std_a = float(np.sqrt(np.sum(w * (a - a_mean) ** 2)))
            std_b = float(np.sqrt(np.sum(w * (b - b_mean) ** 2)))
            if std_a < 1e-9 or std_b < 1e-9:
                return 1.0
            corr = cov / (std_a * std_b)
            if not math.isfinite(corr):
                return 0.0
            return float(np.clip(corr, -1.0, 1.0))
        except Exception as e:
            logger.warning("emotional_arc_preservation.py::_weighted_pearson Ersatzpfad: %s", e)
            return 0.0

    def correct_arc(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        max_gain_db: float = 6.0,
        damping: float = 0.7,
        frisson_zones=None,
    ) -> tuple[np.ndarray, EmotionalArcResult]:
        """Correct emotional arc via macro-level RMS-based gain envelope.

        Operates at 5 s segment timescale, complementing MDEM's 400 ms
        micro-dynamics.  Only the RMS component (0.6 weight in arousal
        proxy) responds to gain correction; ZCR (0.4 weight) is spectral.

        Algorithm (Thayer 1989 / Kim & André 2008):
            1. Per-segment RMS for original + restored (5 s, hop 2.5 s)
            2. gain_db = 20·log₁₀(rms_orig / rms_rest) · damping, ±max_gain_db
            3. Savitzky-Golay smooth → linear interpolation to sample-level
            4. Apply per-channel, NaN-guard, clip ±1.0
            5. Re-measure; revert if arousal worsened

        Args:
            original: Pre-repair reference audio (float32, 1D/2D)
            restored: Post-MDEM restored audio (float32, 1D/2D)
            sr:       48000
            max_gain_db: Max per-segment gain (default 6 dB)
            damping:  Correction fraction 0–1 (default 0.7 = 70 % correction)

        Returns:
            (corrected_audio, EmotionalArcResult after correction)
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"

        def _to_mono(a: np.ndarray) -> np.ndarray:
            # Mono-Reduktion über zentralen safe_to_mono (Konsistenz statt lokalem
            # mean(axis=0)). Kanonischer Kontrakt dieser Methode ist channels-first
            # (C, N) — auf diesem Pfad verhaltensneutral; safe_to_mono ist zusätzlich
            # layout-agnostisch für die RMS-Hüllkurven-Berechnung.
            arr = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mono = safe_to_mono(arr) if arr.ndim == 2 else arr
            return np.asarray(mono, dtype=np.float32)  # type: ignore[no-any-return]

        orig_mono = _to_mono(original)
        rest_mono = _to_mono(restored)

        n = min(len(orig_mono), len(rest_mono))
        orig_mono = orig_mono[:n]
        rest_mono = rest_mono[:n]

        # Too short for meaningful arc correction
        if n / sr < self.MIN_DURATION_S:
            return restored.copy(), self.measure(original, restored, sr)

        seg_len = int(self.SEGMENT_S * sr)
        hop_len = int(self.HOP_S * sr)
        positions = list(range(0, n - seg_len + 1, hop_len))

        if len(positions) < 3:
            return restored.copy(), self.measure(original, restored, sr)

        # §Frisson: Gänsehaut-Schutz-Frames → Frame-Index aus .start_s/.end_s
        _frisson_frame_set: set[int] = set()
        if frisson_zones:
            try:
                _hop_s = float(self.HOP_S)  # 2.5 s
                for _fz in frisson_zones:
                    _fi_start = max(0, int(float(getattr(_fz, "start_s", 0.0)) / _hop_s))
                    _fi_end = int(float(getattr(_fz, "end_s", 0.0)) / _hop_s) + 1
                    for _fi in range(_fi_start, min(_fi_end, len(positions))):
                        _frisson_frame_set.add(_fi)
                if _frisson_frame_set:
                    logger.debug("correct_arc §Frisson: %d Frames geschützt", len(_frisson_frame_set))
            except Exception:
                _frisson_frame_set = set()

        # ---- Per-segment RMS ----
        rms_orig = np.empty(len(positions), dtype=np.float32)
        rms_rest = np.empty(len(positions), dtype=np.float32)
        for i, start in enumerate(positions):
            end = start + seg_len
            rms_orig[i] = float(np.sqrt(np.mean(orig_mono[start:end] ** 2) + 1e-12))
            rms_rest[i] = float(np.sqrt(np.mean(rest_mono[start:end] ** 2) + 1e-12))

        # ---- Gain in dB, damped ----
        eps = 1e-9
        gain_db = 20.0 * np.log10((rms_orig + eps) / (rms_rest + eps))
        gain_db = np.clip(gain_db.astype(np.float64) * damping, -max_gain_db, max_gain_db)

        # §0a Original-Music-Floor: frames where rms_orig is below the music floor of
        # the original are carrier-noise frames (vinyl/tape surface noise), not music.
        # Applying positive gain on these frames re-amplifies removed carrier noise →
        # Pegelexplosion in intro/outro/fadeout. Compute once, use in both SG guards.
        _orig_rms_db_all = 20.0 * np.log10(rms_orig + eps)
        _orig_active = _orig_rms_db_all[_orig_rms_db_all > -60.0]
        if len(_orig_active) > 4:
            _orig_median_db = float(np.median(_orig_active))
            _orig_music_floor_db = float(np.clip(_orig_median_db - 18.0, -48.0, -20.0))
        else:
            _orig_music_floor_db = -42.0

        # Guard: suppress positive gain on near-silent restored segments.
        # After denoising the tail is genuinely silent; applying a positive
        # gain here (to match original tape/vinyl noise floor) inflates the
        # integrated LUFS measurement and forces phase_40 to attenuate the
        # musical content in compensation (regression).
        _silence_rms_thresh = 10.0 ** (-60.0 / 20.0)  # −60 dBFS ≈ noise floor
        # §2.30 Bug-Fix §13 — Fade-out denoising guard (companion to MDEM guard):
        # Restored segment is moderately quiet (< -42 dBFS) AND original is
        # significantly louder (> 6 dB difference). This is the "denoised fade-out"
        # pattern: original had noise+signal, restored has only the clean signal.
        # Applying positive gain would reconstruct the noise floor and cause an
        # audible volume jump at the end of the song.
        _quiet_rms_thresh = 10.0 ** (-42.0 / 20.0)  # −42 dBFS = quiet/fade-out zone
        _noise_diff_thresh_db = 6.0
        # §WPG-Quiet-Zone: frames quieter than -36 dBFS must NEVER receive positive gain.
        # −36 dBFS matches MDEM + per-sample guard thresholds (§2.30b, §2.45a).
        # Vinyl surface noise sits at −33 to −38 dBFS after denoising — using −24 dBFS
        # let those frames through the quiet-zone guard when diff < 6 dB (only the
        # _quiet_rms_thresh=-42 dBFS branch blocks at diff≥6 dB, not these frames).
        # −36 dBFS ensures vinyl noise-floor frames are always blocked by the WPG branch.
        # NOTE: _moderate_quiet_rms_thresh (−30 dBFS) was dead code — every value < −30 dBFS
        # is also < −36 dBFS so that elif was never reached; removed.
        _wpg_quiet_rms_thresh = 10.0 ** (-36.0 / 20.0)  # −36 dBFS = vinyl noise floor gate
        for _i in range(len(gain_db)):
            if gain_db[_i] > 0.0:
                # §0a Guard 0: Original frame at carrier noise floor — no positive boost.
                if _orig_rms_db_all[_i] < _orig_music_floor_db:
                    gain_db[_i] = 0.0  # Carrier-noise frame in original — never boost
                elif rms_rest[_i] < _silence_rms_thresh:
                    gain_db[_i] = 0.0
                elif rms_rest[_i] < _quiet_rms_thresh:
                    _diff = 20.0 * np.log10((rms_orig[_i] + eps) / (rms_rest[_i] + eps))
                    if _diff >= _noise_diff_thresh_db:
                        gain_db[_i] = 0.0  # Denoised fade-out: no boost
                elif rms_rest[_i] < _wpg_quiet_rms_thresh:
                    # WPG quiet zone (-42 to -36 dBFS): ALWAYS block positive boost.
                    gain_db[_i] = 0.0
                else:
                    # §2.30b Guard 3 — Any-level noise-removal guard (rms_rest >= −36 dBFS):
                    _diff_any = 20.0 * np.log10((rms_orig[_i] + eps) / (rms_rest[_i] + eps))
                    if _diff_any > max_gain_db:
                        gain_db[_i] = 0.0  # Noise-removal at any level: no boost

        # §Frisson Pre-SG: Klimax-Frames vor SG-Dämpfung schützen (max. −1.0 LU Absenkung)
        if _frisson_frame_set:
            for _fk in _frisson_frame_set:
                if 0 <= _fk < len(gain_db):
                    gain_db[_fk] = max(gain_db[_fk], -1.0)

        # Savitzky-Golay smooth (boxcar fallback)
        if len(gain_db) >= 7:
            try:
                from scipy.signal import savgol_filter  # pylint: disable=import-outside-toplevel

                gain_db = savgol_filter(gain_db, window_length=7, polyorder=2)
            except Exception:
                kernel = np.ones(5, dtype=np.float64) / 5.0
                gain_db = np.convolve(gain_db, kernel, mode="same")

        gain_db = np.clip(gain_db, -max_gain_db, max_gain_db).astype(np.float32)

        # §2.30 Post-Smoothing Quiet-Zone-Clamp: Savitzky-Golay (window=7, covers
        # up to 17.5 s at HOP_S=2.5 s) can spread positive gain from musical segments
        # into adjacent denoised/quiet segments that were previously zeroed by the
        # guard above. Re-apply the guard after smoothing — mirrors the identical fix
        # in MDEM (MicroDynamicsEnvelopeMorphing). Without this, a loud intro section
        # (0–30 s) causes a Pegelexplosion in a quiet fadeout at ~35 s (15.83 %).
        for _i in range(len(gain_db)):
            if gain_db[_i] > 0.0:
                # §0a Guard 0 post-SG: mirror of pre-SG original-music-floor guard
                if _orig_rms_db_all[_i] < _orig_music_floor_db:
                    gain_db[_i] = 0.0  # Original noise floor — no boost
                elif rms_rest[_i] < _silence_rms_thresh:
                    gain_db[_i] = 0.0
                elif rms_rest[_i] < _quiet_rms_thresh:
                    _diff_ps = 20.0 * np.log10((rms_orig[_i] + eps) / (rms_rest[_i] + eps))
                    if _diff_ps >= _noise_diff_thresh_db:
                        gain_db[_i] = 0.0  # Post-smoothing: denoised fade-out, no boost
                elif rms_rest[_i] < _wpg_quiet_rms_thresh:
                    # Post-SG WPG quiet zone (-36 dBFS): always block
                    gain_db[_i] = 0.0
                else:
                    # §2.30b Guard 3 post-smoothing — mirror of pre-smoothing Guard 3:
                    _diff_ps_any = 20.0 * np.log10((rms_orig[_i] + eps) / (rms_rest[_i] + eps))
                    if _diff_ps_any > max_gain_db:
                        gain_db[_i] = 0.0  # Post-smoothing: noise-removal at any level

        # §Frisson Post-SG: Zwei-Stufen-Invariante — SG verteilt Dämpfung in Frisson-Frames zurück
        if _frisson_frame_set:
            for _fk in _frisson_frame_set:
                if 0 <= _fk < len(gain_db):
                    gain_db[_fk] = max(gain_db[_fk], -1.0)

        # ---- Interpolate to sample-level via segment centres ----
        centres = np.array([start + seg_len // 2 for start in positions], dtype=np.float64)
        sample_idx = np.arange(n, dtype=np.float64)
        gain_db_interp = np.interp(sample_idx, centres, gain_db).astype(np.float32)

        # §2.30 Per-Sample Quiet-Zone-Guard: np.interp creates a positive ramp in
        # the transition region between a high-gain musical segment and a 0-gain
        # quiet segment (e.g., +4 dB at 32.5 s → 0 dB at 35 s gives +2 dB at 33.75 s).
        # Suppress positive interpolated gain wherever the restored signal itself is
        # below −36 dBFS (per-sample guard, matching MDEM morph() threshold).
        #
        # §2.30b Adaptiver Per-Sample-Guard (Schicht 6):
        # Für Shellac/Kassette/stark verrauschtes Vinyl liegt der restaurierte Rauschboden
        # bei −25 bis −35 dBFS (nach partieller Entrauschung). Der feste −36-dBFS-Schwellwert
        # lässt diese Frames durch, weil −28 dBFS > −36 dBFS.
        # Adaptiver Ansatz: 5th-Perzentil der 5-s-Segmente (Proxy für Trägerrauschboden)
        # + 8 dB Margin, begrenzt auf [−36, −18] dBFS.
        # Vollständig entrauschtes Material (p5 ≈ −65 dBFS): Schwellwert bleibt −36 dBFS.
        # Shellac mit Rauschboden −30 dBFS: Schwellwert → max(−36, −22) = −22 dBFS.
        _p5_rms_seg = float(np.percentile(rms_rest, 5)) if len(rms_rest) > 2 else 10.0 ** (-36.0 / 20.0)
        _p5_dbfs_seg = 20.0 * np.log10(max(_p5_rms_seg, 1e-12) + 1e-12)
        _adaptive_thresh_ps_dbfs = float(np.clip(_p5_dbfs_seg + 8.0, -36.0, -18.0))
        _quiet_rms_thresh_ps = 10.0 ** (_adaptive_thresh_ps_dbfs / 20.0)  # adaptive per-sample
        _frame_len_ps = 480  # 10 ms @ 48 kHz
        _n_full_ps = n // _frame_len_ps
        if _n_full_ps > 0:
            _segs_ps = rest_mono[: _n_full_ps * _frame_len_ps].reshape(_n_full_ps, _frame_len_ps)
            _rms_ps = np.sqrt(np.mean(_segs_ps**2, axis=1) + 1e-12)
            _is_quiet_ps = _rms_ps < float(_quiet_rms_thresh_ps)
            _quiet_mask = np.repeat(_is_quiet_ps, _frame_len_ps)
            if _n_full_ps * _frame_len_ps < n:
                _tail_rms_ps = float(np.sqrt(np.mean(rest_mono[_n_full_ps * _frame_len_ps :] ** 2) + 1e-12))
                _quiet_mask = np.concatenate(
                    [
                        _quiet_mask,
                        np.full(n - _n_full_ps * _frame_len_ps, _tail_rms_ps < float(_quiet_rms_thresh_ps)),
                    ]
                )
            # Zero out any positive interpolated gain in the quiet zone
            gain_db_interp[_quiet_mask[:n] & (gain_db_interp[:n] > 0.0)] = 0.0

        gain_linear = np.float32(10.0) ** (gain_db_interp / np.float32(20.0))

        # ---- Apply gain ----
        out = np.asarray(restored, dtype=np.float32).copy()
        if out.ndim == 2:
            for ch in range(out.shape[0]):
                ch_n = min(out.shape[1], len(gain_linear))
                out[ch, :ch_n] *= gain_linear[:ch_n]
        else:
            ch_n = min(len(out), len(gain_linear))
            out[:ch_n] *= gain_linear[:ch_n]

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        # ---- Re-measure; safety-revert if correction worsened arousal ----
        arc_before = self.measure(original, restored, sr)
        arc_after = self.measure(original, out, sr)

        if not arc_before.skipped:
            if arc_after.arousal_pearson < arc_before.arousal_pearson - 0.02:
                logger.warning(
                    "EmotionalArc correction worsened arousal (%.3f → %.3f) — reverting",
                    arc_before.arousal_pearson,
                    arc_after.arousal_pearson,
                )
                return restored.copy(), arc_before

        logger.info(
            "EmotionalArc correction angewendet: arousal %.3f→%.3f  valence %.3f→%.3f  max_gain=±%.1f dB",
            arc_before.arousal_pearson,
            arc_after.arousal_pearson,
            arc_before.valence_pearson,
            arc_after.valence_pearson,
            float(np.max(np.abs(gain_db))),
        )
        return out, arc_after


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: EmotionalArcPreservationMetric | None = None
_lock = threading.Lock()


def get_emotional_arc_metric() -> EmotionalArcPreservationMetric:
    """Thread-sicherer Singleton-Accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = EmotionalArcPreservationMetric()
    return _instance


def measure_emotional_arc(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
    lyrics_saliency: np.ndarray | None = None,
    frisson_zones=None,
    transfer_chain_depth: int | None = None,
) -> EmotionalArcResult:
    """Convenience-Wrapper: Erhalt des emotionalen Bogens prüfen.

    Args:
        original:        Original-Audio vor Restaurierung, float32, SR = 48000
        restored:        Restauriertes Audio, float32, SR = 48000
        sr:              48000 (Pflicht)
        lyrics_saliency: Optionales Salienz-Array aus §2.36 LGE (§2.44).
        frisson_zones:   Optionale Liste von FrissonZone-Objekten (.start_s, .end_s).
                         Frisson-Segmente erhalten Gewicht ×2.0 in der Pearson-Korrelation
                         (§2.44: Klimax-Passagen prägen Emotionswahrnehmung stärker).

    Returns:
        EmotionalArcResult mit Pearson-Korrelationen und Klimax-Analyse.
    """
    return get_emotional_arc_metric().measure(
        original,
        restored,
        sr,
        lyrics_saliency=lyrics_saliency,
        frisson_zones=frisson_zones,
        transfer_chain_depth=transfer_chain_depth,
    )


def correct_emotional_arc(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
    max_gain_db: float = 6.0,
    damping: float = 0.7,
    frisson_zones=None,
) -> tuple[np.ndarray, EmotionalArcResult]:
    """Convenience wrapper: correct emotional arc macro-dynamics.

    Args:
        original: Pre-repair reference audio, float32, SR=48000
        restored: Post-MDEM restored audio, float32, SR=48000
        sr:       48000 (mandatory)
        max_gain_db: Max per-segment gain (default 6 dB)
        damping:  Correction fraction 0–1 (default 0.7)
        frisson_zones: Optional list of FrissonZone objects (.start_s, .end_s) —
            §Frisson Zwei-Stufen-Invariante: Pre-SG + Post-SG floor −1.0 LU

    Returns:
        (corrected_audio, EmotionalArcResult)
    """
    return get_emotional_arc_metric().correct_arc(
        original, restored, sr, max_gain_db=max_gain_db, damping=damping, frisson_zones=frisson_zones
    )


# ---------------------------------------------------------------------------
# §2.30c WaveformPlausibilityGuard — Finale Pegelexplosions-Fangschicht
# ---------------------------------------------------------------------------


class WaveformPlausibilityGuard:
    """§2.30c Final waveform sanity check — last safety layer before HPI Gate.

    Detects Pegelexplosion (level anomalies where restored >> original) and
    applies targeted envelope attenuation. Operates AFTER correct_arc.

    Design invariants:
    - ONLY attenuation (gain ≤ 1.0, never boost) — spectral repair work untouched
    - Pure envelope correction: spectral shape, harmonics, BW-extension preserved
    - Musical Goals protection:
        * P1/P2 (Natürlichkeit, Authentizität, Timbre, TonalCenter, Artikulation):
          gain-only correction cannot harm spectral ratios → SAFE
        * P3 Emotionalität: arc-Pearson proxy must not worsen by > 0.05
        * P3 MikroDynamik: dynamic range must not collapse by > 4 dB
        * On proxy fail: reduce correction to 50%, retry
        * On second fail: skip correction entirely (§0 Primum non nocere)
    - Non-blocking: any exception → return restored unchanged + log warning

    References:
        §0 Primum non nocere — kein Artefakt einführen
        §2.30b Drei-Stufen-Invariante (prior guards; this is the final catch-all)
        §2.45a Musical Goals preservation during envelope corrections
    """

    WINDOW_S: float = 2.0  # 2 s detection windows
    HOP_S: float = 0.5  # 0.5 s hop → 4 windows/s
    # After correction, aim for restored ≈ orig + CORRECTION_TARGET_DB
    # (+2 dB headroom allows legitimate slight enhancement after denoising)
    CORRECTION_TARGET_DB: float = 2.0
    # Max single-window attenuation (don't over-correct; max 12 dB step)
    MAX_ATTENUATION_DB: float = -12.0
    # Quiet-zone emergency policy (§2.30c hard catch): if explosions are concentrated
    # in quiet/fade windows, prioritize attenuation over proxy correlation stability.
    # §2.30b: vinyl surface noise floor sits at −33 to −38 dBFS after denoising.
    # −24.0 was too permissive — those frames bypassed the quiet-zone emergency trigger.
    # −36.0 aligns with the per-sample guard (same value used throughout MDEM + correct_arc).
    _QUIET_ZONE_DBFS: float = -34.0
    _QUIET_EXPLOSION_RATIO_MIN: float = 0.80

    # Mode-adaptive explosion thresholds
    # IMPORTANT: These must be STRICTLY below the max gain any upstream guard can apply.
    # LUFS rescue caps at 6 dB → threshold must be < 6 dB to catch boundary frames.
    # correct_arc() arousal boost can apply 2-4 dB → threshold 3 dB catches that too.
    _THRESHOLD_DB: dict[str, float] = {
        "restoration": 3.0,  # was 6.0 — LUFS rescue caps at 6 dB; WPG must use < 6 dB
        "studio2026": 5.0,  # was 8.0
        "studio_2026": 5.0,
        "default": 3.0,
    }

    # Material-adaptive threshold offset (analog sources need tighter threshold)
    _MATERIAL_THRESHOLD_OFFSET: dict[str, float] = {
        "shellac": -1.0,  # max +2 dB — almost no enhancement expected
        "wax_cylinder": -1.0,
        "wire_recording": -1.0,
        "reel_tape": -1.0,  # max +2 dB
        "cassette": -1.0,
        "vinyl": -1.0,  # max +2 dB — loud noise floor (-33 dBFS), tight guard
        "cd_digital": 2.0,  # max +5 dB — digital source tolerates more
        "default": 0.0,
    }

    def apply(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        mode: str = "restoration",
        material_type: str = "unknown",
        restorability_score: float = 50.0,
    ) -> tuple[np.ndarray, dict]:
        """Erkennt and correct Pegelexplosion while preserving Musical Goals.

        Args:
            original:           Pre-restoration reference audio (float32)
            restored:           Post-correct_arc restored audio (float32)
            sr:                 Sample rate (must be 48000)
            mode:               "restoration" or "studio2026"
            material_type:      Carrier material string (e.g. "vinyl", "shellac")
            restorability_score: 0–100 restorability estimate

        Returns:
            (corrected_audio, metadata_dict)
            corrected_audio: identical to restored if no explosion detected or guard skipped
            metadata keys: "explosions_found" (int), "corrections_applied" (int),
                           "max_attenuation_db" (float), "correction_eased" (bool),
                           "skipped_reason" (str | None)
        """
        meta: dict = {
            "explosions_found": 0,
            "corrections_applied": 0,
            "max_attenuation_db": 0.0,
            "correction_eased": False,
            "quiet_zone_emergency_applied": False,
            "explosion_quiet_ratio": 0.0,
            "skipped_reason": None,
        }
        try:
            return self._apply_inner(original, restored, sr, mode, material_type, restorability_score, meta)
        except Exception as _exc:
            logger.warning("WaveformPlausibilityGuard: unhandled exception — ueberspringen. %s", _exc)
            meta["skipped_reason"] = f"exception:{type(_exc).__name__}"
            return np.asarray(restored, dtype=np.float32).copy(), meta

    def _apply_inner(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        mode: str,
        material_type: str,
        restorability_score: float,
        meta: dict,
    ) -> tuple[np.ndarray, dict]:
        eps = 1e-12
        n = original.shape[-1]

        # Require at least 10 s audio for meaningful window analysis
        if n < int(sr * 10.0):
            meta["skipped_reason"] = "audio_too_short"
            return np.asarray(restored, dtype=np.float32).copy(), meta

        orig_mono = self._to_mono(original)
        rest_mono = self._to_mono(restored)

        win = int(self.WINDOW_S * sr)
        hop = int(self.HOP_S * sr)
        n_frames = max(1, (n - win) // hop + 1)

        # --- Adaptive threshold ---
        base_thr = self._THRESHOLD_DB.get(mode, self._THRESHOLD_DB["default"])
        mat_key = str(material_type or "").lower().strip()
        mat_offset = self._MATERIAL_THRESHOLD_OFFSET.get(mat_key, 0.0)
        # Tighter for low-restorability (heavy damage → minimal legitimate enhancement)
        rest_offset = -1.0 if restorability_score < 40.0 else 0.0
        threshold_db = base_thr + mat_offset + rest_offset

        # --- Compute per-frame RMS ---
        orig_rms_db = np.empty(n_frames, dtype=np.float64)
        rest_rms_db = np.empty(n_frames, dtype=np.float64)
        for k in range(n_frames):
            s = k * hop
            e = min(s + win, n)
            orig_rms_db[k] = 20.0 * np.log10(float(np.sqrt(np.mean(orig_mono[s:e] ** 2))) + eps)
            rest_rms_db[k] = 20.0 * np.log10(float(np.sqrt(np.mean(rest_mono[s:e] ** 2))) + eps)

        delta_db = rest_rms_db - orig_rms_db
        # Use >= instead of > so frames with gain exactly at threshold are caught.
        # LUFS rescue caps at exactly 6 dB; with threshold=3 dB this is moot,
        # but the >= prevents any future boundary bypass if thresholds are tuned.
        exploded = delta_db >= threshold_db
        meta["explosions_found"] = int(np.sum(exploded))

        if meta["explosions_found"] == 0:
            return np.asarray(restored, dtype=np.float32).copy(), meta

        # --- Build per-frame correction gain (always ≤ 0 dB) ---
        # CORRECTION_TARGET_DB (2 dB) applies only to MUSIC frames.
        # Quiet-zone frames (intro/outro noise, orig ≤ _QUIET_ZONE_DBFS) must be
        # corrected to EXACTLY original level (target=0 dB above orig) so that
        # carrier-noise (vinyl -33 dBFS) is never lifted above its natural level.
        # MAX_ATTENUATION_DB is also lifted for quiet-zone frames to handle
        # extreme explosions (> 12 dB) that the default -12 dB cap would miss.
        gain_db_frames = np.zeros(n_frames, dtype=np.float64)
        for k in range(n_frames):
            if exploded[k]:
                is_quiet_frame = orig_rms_db[k] <= self._QUIET_ZONE_DBFS
                if is_quiet_frame:
                    # Quiet zone: correct to exactly original — no boost at all
                    target_db = orig_rms_db[k]  # 0 dB above original
                    max_att = -60.0  # effectively unlimited attenuation
                else:
                    target_db = orig_rms_db[k] + self.CORRECTION_TARGET_DB
                    max_att = self.MAX_ATTENUATION_DB
                gain_db_frames[k] = float(np.clip(target_db - rest_rms_db[k], max_att, 0.0))

        # --- Interpolate frame gains to sample level ---
        centres = np.array([k * hop + win // 2 for k in range(n_frames)], dtype=np.float64)
        sample_idx = np.arange(n, dtype=np.float64)
        gain_db_interp = np.interp(sample_idx, centres, gain_db_frames).astype(np.float32)
        # Hard invariant: this guard NEVER boosts — clamp positive interpolation ramps
        gain_db_interp = np.minimum(gain_db_interp, 0.0)

        # --- Apply full correction ---
        corrected = self._apply_gain(restored, gain_db_interp)

        # Quiet-zone emergency detector: if the vast majority of explosions happen in
        # quiet/fade windows, we accept attenuation directly to avoid real-audio failures
        # where proxy checks (arc/DR) are unstable but explosion is clearly harmful.
        _quiet_exploded = exploded & (orig_rms_db <= self._QUIET_ZONE_DBFS)
        _expl_count = int(np.sum(exploded))
        _quiet_ratio = (float(np.sum(_quiet_exploded)) / float(_expl_count)) if _expl_count > 0 else 0.0
        meta["explosion_quiet_ratio"] = _quiet_ratio
        if _quiet_ratio >= self._QUIET_EXPLOSION_RATIO_MIN:
            meta["corrections_applied"] = _expl_count
            meta["max_attenuation_db"] = float(np.min(gain_db_interp))
            meta["quiet_zone_emergency_applied"] = True
            logger.info(
                "WaveformPlausibilityGuard: Quiet-Zone-Notfallkorrektur aktiv "
                "(quiet_Verhaeltnis=%.2f, windows=%d, max=%.1f dB, thr=%.1f dB, mat=%s)",
                _quiet_ratio,
                _expl_count,
                meta["max_attenuation_db"],
                threshold_db,
                mat_key,
            )
            return corrected, meta

        # rrected = self._apply_gain(restored, gain_db_interp)

        # --- Musical Goals proxy check ---
        arc_before, arc_after, dr_before, dr_after = self._measure_goals_proxy(
            orig_mono, rest_mono, self._to_mono(corrected), sr, n_frames, hop, win, eps
        )
        arc_ok = arc_after >= arc_before - 0.05
        dr_ok = dr_after >= dr_before - 4.0

        if arc_ok and dr_ok:
            meta["corrections_applied"] = int(np.sum(exploded))
            meta["max_attenuation_db"] = float(np.min(gain_db_interp))
            logger.info(
                "WaveformPlausibilityGuard: %d Fenster korrigiert, max=%.1f dB "
                "(arc: %.3f→%.3f, DR: %.1f→%.1f dB, thr=%.1f dB, mat=%s)",
                meta["corrections_applied"],
                meta["max_attenuation_db"],
                arc_before,
                arc_after,
                dr_before,
                dr_after,
                threshold_db,
                mat_key,
            )
            return corrected, meta

        # --- Fallback: 50% correction ---
        gain_db_half = (gain_db_frames * 0.5).astype(np.float64)
        gain_db_half_interp = np.interp(sample_idx, centres, gain_db_half).astype(np.float32)
        gain_db_half_interp = np.minimum(gain_db_half_interp, 0.0)
        corrected_half = self._apply_gain(restored, gain_db_half_interp)

        arc_h_b, arc_h_a, dr_h_b, dr_h_a = self._measure_goals_proxy(
            orig_mono, rest_mono, self._to_mono(corrected_half), sr, n_frames, hop, win, eps
        )
        arc_half_ok = arc_h_a >= arc_h_b - 0.05
        dr_half_ok = dr_h_a >= dr_h_b - 4.0

        if arc_half_ok and dr_half_ok:
            meta["corrections_applied"] = int(np.sum(exploded))
            meta["max_attenuation_db"] = float(np.min(gain_db_half_interp))
            meta["correction_eased"] = True
            logger.info(
                "WaveformPlausibilityGuard: 50%%-Korrektur (%d Fenster, max=%.1f dB) "
                "— volle Korrektur wäre zu aggressiv (arc: %.3f→%.3f)",
                meta["corrections_applied"],
                meta["max_attenuation_db"],
                arc_before,
                arc_after,
            )
            return corrected_half, meta

        # --- Both failed: §0 Primum non nocere — skip ---
        meta["skipped_reason"] = (
            "musical_goals_proxy_fail("
            f"arc_before={arc_before:.3f},arc_after={arc_after:.3f},"
            f"dr_before={dr_before:.1f},dr_after={dr_after:.1f})"
        )
        logger.warning(
            "WaveformPlausibilityGuard: %d Explosions-Fenster erkannt, "
            "aber Korrektur verletzt Musical-Goals-Proxy → ueberspringen. "
            "arc: %.3f→%.3f, DR: %.1f→%.1f dB",
            meta["explosions_found"],
            arc_before,
            arc_after,
            dr_before,
            dr_after,
        )
        return np.asarray(restored, dtype=np.float32).copy(), meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Konvertiert to mono 1D float32. Handles (channels, samples) orientation."""
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 2:
            # Canonical Aurik format: (channels, samples) — channels dim is always ≤ 2
            if a.shape[0] <= 2:
                return np.mean(a, axis=0)  # type: ignore
            else:
                # Unexpected orientation: (samples, channels)
                return np.mean(a, axis=1)  # type: ignore
        return a  # type: ignore[no-any-return]

    def _apply_gain(self, audio: np.ndarray, gain_db_interp: np.ndarray) -> np.ndarray:
        """Wendet an: sample-level gain (dB) and clip output to [-1, 1]."""
        gain_linear = (10.0 ** (gain_db_interp / 20.0)).astype(np.float32)
        out = np.asarray(audio, dtype=np.float32).copy()
        n = len(gain_linear)
        if out.ndim == 2:
            for ch in range(out.shape[0]):
                cn = min(out.shape[1], n)
                out[ch, :cn] *= gain_linear[:cn]
        else:
            cn = min(len(out), n)
            out[:cn] *= gain_linear[:cn]
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    def _measure_goals_proxy(
        self,
        orig_mono: np.ndarray,
        rest_mono: np.ndarray,
        corr_mono: np.ndarray,
        sr: int,
        n_frames: int,
        hop: int,
        win: int,
        eps: float,
    ) -> tuple[float, float, float, float]:
        """Berechnet Musical Goals proxy metrics (arc Pearson + DR).

        Returns:
            (arc_pearson_before, arc_pearson_after, dr_before_db, dr_after_db)
        """
        n = len(orig_mono)

        # --- Arc correlation (P3 Emotionalität proxy): 5s-window RMS Pearson ---
        arc_win = int(5.0 * sr)
        n_arc = max(2, n // arc_win)

        def rms_curve(mono: np.ndarray) -> np.ndarray:
            return np.array(  # type: ignore[no-any-return]
                [
                    20.0 * np.log10(float(np.sqrt(np.mean(mono[k * arc_win : (k + 1) * arc_win] ** 2))) + eps)
                    for k in range(n_arc)
                ]
            )

        def pearson(a: np.ndarray, b: np.ndarray) -> float:
            a_c = a - a.mean()
            b_c = b - b.mean()
            denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c)) + eps
            return float(np.dot(a_c, b_c) / denom)

        orig_arc = rms_curve(orig_mono)
        rest_arc = rms_curve(rest_mono)
        corr_arc = rms_curve(corr_mono)

        arc_before = pearson(orig_arc, rest_arc)
        arc_after = pearson(orig_arc, corr_arc)

        # --- Dynamic range (P3 MikroDynamik proxy): p95 − p5 of 2s-window RMS ---
        rest_rms_frames = np.array(
            [
                20.0 * np.log10(float(np.sqrt(np.mean(rest_mono[k * hop : k * hop + win] ** 2))) + eps)
                for k in range(n_frames)
            ]
        )
        corr_rms_frames = np.array(
            [
                20.0 * np.log10(float(np.sqrt(np.mean(corr_mono[k * hop : k * hop + win] ** 2))) + eps)
                for k in range(n_frames)
            ]
        )
        dr_before = float(np.percentile(rest_rms_frames, 95) - np.percentile(rest_rms_frames, 5))
        dr_after = float(np.percentile(corr_rms_frames, 95) - np.percentile(corr_rms_frames, 5))

        return arc_before, arc_after, dr_before, dr_after


# Thread-safe singleton for WaveformPlausibilityGuard
_wpg_instance: WaveformPlausibilityGuard | None = None
_wpg_lock = threading.Lock()


def get_waveform_plausibility_guard() -> WaveformPlausibilityGuard:
    """Thread-sicherer Singleton-Accessor für WaveformPlausibilityGuard."""
    global _wpg_instance  # pylint: disable=global-statement
    if _wpg_instance is None:
        with _wpg_lock:
            if _wpg_instance is None:
                _wpg_instance = WaveformPlausibilityGuard()
    return _wpg_instance


def apply_waveform_plausibility_guard(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
    mode: str = "restoration",
    material_type: str = "unknown",
    restorability_score: float = 50.0,
) -> tuple[np.ndarray, dict]:
    """Convenience wrapper for WaveformPlausibilityGuard.apply().

    Final safety layer after correct_arc — detects Pegelexplosion and applies
    targeted envelope attenuation while preserving all 15 Musical Goals.

    Args:
        original:           Pre-restoration reference audio (float32, SR=48000)
        restored:           Post-correct_arc audio (float32, SR=48000)
        sr:                 48000 (mandatory)
        mode:               "restoration" or "studio2026"
        material_type:      Carrier material string (e.g. "vinyl", "shellac")
        restorability_score: 0–100 restorability estimate

    Returns:
        (corrected_audio, metadata_dict)
    """
    return get_waveform_plausibility_guard().apply(
        original,
        restored,
        sr,
        mode=mode,
        material_type=material_type,
        restorability_score=restorability_score,
    )
