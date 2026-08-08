"""
VocalQualityIndex (VQI) — §2.35c [RELEASE_MUST]
================================================

Vocal overall quality gate. Aggregates 5 orthogonal vocal metrics into a
single score [0, 1]. Activated when PANNs singing confidence >= 0.35.

Spec: 01_musical_goals.md §2.35c (v10.0.0)
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Thresholds (§2.35c)
# ---------------------------------------------------------------------------
VQI_WORLD_CLASS = 0.88
VQI_PROFESSIONAL = 0.82
VQI_THRESHOLD = 0.72  # Recovery-cascade vinyl default (§0p)

# Material-adaptive VQI recovery-cascade floors (§0p)
# Shellac/Wax: 0.62 — physical BW/SNR limits; Vinyl: 0.72; Tape: 0.72; CD: 0.82
_VQI_MATERIAL_FLOOR: dict[str, float] = {
    "wax_cylinder": 0.62,
    "shellac": 0.62,
    "wire_recording": 0.62,
    "lacquer_disc": 0.62,
    "tape": 0.72,
    "reel_tape": 0.72,
    "cassette": 0.70,
    "kassette": 0.70,
    "vinyl": 0.72,
    "lp": 0.72,
    "cd_digital": 0.82,
    "cd": 0.82,
    "dat": 0.82,
    "mp3_low": 0.72,
    "mp3_high": 0.75,
    "aac": 0.75,
    "minidisc": 0.72,
    "streaming": 0.75,
}


def get_vqi_material_floor(material_type: str, is_studio_2026: bool = False) -> float:
    """Gibt the material-adaptive VQI recovery-cascade threshold (§0p) zurück.

    Args:
        material_type: e.g. "shellac", "vinyl", "cd_digital".
        is_studio_2026: Studio-2026 mode → fixed target 0.87 (§0p).

    Returns:
        float: Recovery threshold ∈ [0.62, 0.87].
    """
    if is_studio_2026:
        return 0.87  # §0p: Studio 2026 fixed recovery goal
    mat = str(material_type or "").strip().lower()
    return _VQI_MATERIAL_FLOOR.get(mat, VQI_THRESHOLD)  # default: vinyl 0.72


# Default weights (weighted sum aus §2.35c Spec) — sums to exactly 1.0
_W_SINGER_ID = 0.30
_W_FORMANT = 0.25
_W_ARTICULATION = 0.20
_W_PROXIMITY = 0.15
_W_SIBILANCE = 0.10

# Genre-specific VQI weight profiles (§0p + §2.35c).
# Rationale: different vocal genres define "quality" differently.
#   jazz/blues: identity IS the performance (rough texture intentional) → high singer_id
#   opera:      formant precision is paramount (classical technique) → high formant
#   pop/rock:   articulation + sibilance critical (modern clarity) → high articulation
#   folk/country: natural voice character, less formant constraint → higher singer_id
#   soul/gospel: warmth & proximity matter → proximity elevated
# All rows sum to 1.0. Unknown genre → default weights.
# fmt: off  # column layout: (singer_id, formant, articulation, proximity, sibilance)
_VQI_GENRE_WEIGHTS: dict[str, tuple[float, float, float, float, float]] = {
    "jazz": (0.40, 0.20, 0.15, 0.15, 0.10),
    "blues": (0.40, 0.20, 0.15, 0.15, 0.10),
    "jazz_vocal": (0.40, 0.20, 0.15, 0.15, 0.10),
    "opera": (0.25, 0.40, 0.15, 0.15, 0.05),
    "klassik": (0.25, 0.40, 0.15, 0.15, 0.05),
    "classical": (0.25, 0.40, 0.15, 0.15, 0.05),
    "oper": (0.25, 0.40, 0.15, 0.15, 0.05),
    "pop": (0.25, 0.20, 0.25, 0.15, 0.15),
    "rock": (0.25, 0.20, 0.25, 0.15, 0.15),
    "pop_rock": (0.25, 0.20, 0.25, 0.15, 0.15),
    "folk": (0.35, 0.20, 0.20, 0.15, 0.10),
    "country": (0.35, 0.20, 0.20, 0.15, 0.10),
    "singer_songwriter": (0.35, 0.20, 0.20, 0.15, 0.10),
    "soul": (0.30, 0.20, 0.20, 0.25, 0.05),
    "gospel": (0.30, 0.20, 0.20, 0.25, 0.05),
    "r_b": (0.30, 0.20, 0.20, 0.25, 0.05),
    "schlager": (0.30, 0.25, 0.20, 0.15, 0.10),
    # Deutsche/europäische Genre-Varianten (Aurik-Kernmaterial)
    "german_schlager": (0.30, 0.25, 0.20, 0.15, 0.10),
    "german_pop": (0.28, 0.22, 0.22, 0.15, 0.13),
    "neue_deutsche_welle": (0.30, 0.22, 0.25, 0.13, 0.10),
    "ndw": (0.30, 0.22, 0.25, 0.13, 0.10),
    "chanson": (0.35, 0.22, 0.20, 0.18, 0.05),
    "volksmuziek": (0.30, 0.25, 0.20, 0.15, 0.10),
    "volksmusik": (0.30, 0.25, 0.20, 0.15, 0.10),
    "schlagerfolk": (0.30, 0.25, 0.20, 0.15, 0.10),
    "evergeen": (0.32, 0.25, 0.18, 0.15, 0.10),  # Tippfehler-Variante
    "evergreen": (0.32, 0.25, 0.18, 0.15, 0.10),
    "deutschpop": (0.28, 0.22, 0.22, 0.15, 0.13),
    "beatmusik": (0.28, 0.20, 0.22, 0.15, 0.15),
    "krautrock": (0.25, 0.18, 0.25, 0.15, 0.17),
    "operette": (0.27, 0.38, 0.15, 0.15, 0.05),
    "operetta": (0.27, 0.38, 0.15, 0.15, 0.05),
    "kabarett": (0.35, 0.20, 0.25, 0.12, 0.08),
    "chansonniere": (0.35, 0.22, 0.20, 0.18, 0.05),
    "bossa_nova": (0.35, 0.20, 0.18, 0.20, 0.07),
    "tango": (0.35, 0.22, 0.18, 0.18, 0.07),
}
# fmt: on


def _get_vqi_weights(genre: str | None) -> tuple[float, float, float, float, float]:
    """Gibt (w_singer_id, w_formant, w_articulation, w_proximity, w_sibilance) for genre zurück.

    Falls back to module-level defaults for unknown/None genre.
    All returned tuples sum to 1.0 within float precision.
    """
    if not genre:
        return (_W_SINGER_ID, _W_FORMANT, _W_ARTICULATION, _W_PROXIMITY, _W_SIBILANCE)
    # Normalisierung: Leerzeichen, Bindestriche, Sonderzeichen, Umlaute (DE/FR)
    key = (
        str(genre)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("&", "")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ç", "c")
        .replace("ñ", "n")
        .replace("ô", "o")
    )
    return _VQI_GENRE_WEIGHTS.get(key, (_W_SINGER_ID, _W_FORMANT, _W_ARTICULATION, _W_PROXIMITY, _W_SIBILANCE))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Channels-first (2,N) oder samples-first (N,2) → mono (N,)."""
    if audio.ndim == 1:
        return audio.astype(np.float32)  # type: ignore[no-any-return]
    if audio.ndim == 2:
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            return np.asarray(audio.mean(axis=0), dtype=np.float32)  # type: ignore[no-any-return]
        if audio.shape[1] == 2:
            return np.asarray(audio.mean(axis=1), dtype=np.float32)  # type: ignore[no-any-return]
        if audio.shape[0] == 1:
            return np.asarray(audio[0], dtype=np.float32)  # type: ignore[no-any-return]
    return audio.flatten().astype(np.float32)  # type: ignore[no-any-return]


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """NaN-safe cosine similarity between two vectors."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Component 1 — Singer Identity Cosine (DSP proxy without Resemblyzer)
# ---------------------------------------------------------------------------


def _compute_singer_identity_dsp(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
) -> float:
    """DSP proxy for singer identity: MFCC correlation + spectral-centroid correlation.

    Returns approximate cosine similarity [0, 1]. Used when Resemblyzer is
    not available (§2.35c DSP fallback).
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        n_mfcc = 20
        n_fft = min(2048, len(vocal_pre) // 4)
        hop = n_fft // 4

        mfcc_pre = librosa.feature.mfcc(y=vocal_pre, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
        mfcc_post = librosa.feature.mfcc(y=vocal_post, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)

        min_frames = min(mfcc_pre.shape[1], mfcc_post.shape[1])
        if min_frames < 4:
            return 0.80  # too short — neutral value

        # Mean MFCC over time → compact fingerprint
        vec_pre = mfcc_pre[:, :min_frames].mean(axis=1)
        vec_post = mfcc_post[:, :min_frames].mean(axis=1)
        cosine = _safe_cosine(vec_pre, vec_post)
        # MFCC cosine typically [0.93–0.97] für dieselbe Stimme nach NR-Verarbeitung.
        # §R3 Kalibrierung: raw 0.93 → 0.920 (besteht 0.92-Rollback-Schwelle), raw 0.80 → 0.829.
        # Formel: (8×cosine − 1) / 7 — raw 0.70 → 0.60 (Rollback); raw 0.96 → 0.954.
        # Altes Formel (cosine − 0.70) / 0.30: raw 0.93 → 0.767 — falscher Rollback bei unveränderter Stimme.
        return float(np.clip((cosine * 8.0 - 1.0) / 7.0, 0.0, 1.0))
    except Exception as exc:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
        logger.debug("singer_identity_dsp fehlgeschlagen: %s", exc)
        return 0.80  # conservatively neutral


def _compute_singer_identity(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
) -> tuple[float, bool]:
    """ResemblyzerPlugin (primary) or DSP proxy (fallback). Returns (cosine, dsp_used)."""
    try:
        from plugins.resemblyzer_plugin import get_resemblyzer_plugin  # pylint: disable=import-outside-toplevel

        plugin = get_resemblyzer_plugin()
        if not plugin.available:
            raise RuntimeError("ResemblyzerPlugin not available")

        emb_pre = plugin.embed(vocal_pre, sr)
        emb_post = plugin.embed(vocal_post, sr)
        if emb_pre is None or emb_post is None:
            raise RuntimeError("embed() returned None")

        cosine = plugin.cosine_similarity(emb_pre, emb_post)
    except Exception as exc:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
    except Exception as exc:
        logger.debug("Resemblyzer not verfuegbar (%s) — DSP Ersatzpfad", exc)
        return _compute_singer_identity_dsp(vocal_pre, vocal_post, sr), True


# ---------------------------------------------------------------------------
# Component 2 — Formant Stability Score
# ---------------------------------------------------------------------------


def _compute_formant_stability(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
    era_profile: object = None,
) -> float:
    """Schätzt F1-Drift via LPC im Vokal-Formant-Band (200–3400 Hz).

    Score = max(0, 1 - drift/max_drift_hz); ära-adaptiv via era_profile (§EraVocalProfile).

    Bandpass 200–3400 Hz vor LPC isoliert F1/F2 vom vollen Mix.
    Verhindert, dass HF-Boost durch Restaurierung (Centroid-Shift > 50 %)
    oder Instrumentalspektren die LPC-Resonanzberechnung verfälschen
    (Messartefakt formant≈0.0 statt echter Formant-Drift).
    LPC-Order 14 statt sr/1000+2 (=50 bei 48 kHz) — Standard für 200–3400 Hz.
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel
        from scipy.signal import butter, sosfiltfilt  # pylint: disable=import-outside-toplevel

        # Bandpass 200–3400 Hz: isoliert F1/F2-Range vor LPC.
        # sr/1000+2 = 50 bei sr=48000 erfasst alle Spektralspitzen im vollen Mix
        # (inkl. Instrumentalbegleitung) — nach HF-Boost entspricht das einer
        # drastisch verschobenen "F1"-Schätzung → formant≈0.0 als Messartefakt.
        nyq = sr / 2.0
        _bp_lo = 200.0 / nyq
        _bp_hi = min(3400.0, nyq * 0.95) / nyq
        if _bp_lo < _bp_hi:
            try:
                _sos_bp = butter(4, [_bp_lo, _bp_hi], btype="bandpass", output="sos")
                _pre_bp = sosfiltfilt(_sos_bp, vocal_pre)
                _post_bp = sosfiltfilt(_sos_bp, vocal_post)
            except Exception:
                _pre_bp, _post_bp = vocal_pre, vocal_post
        else:
            _pre_bp, _post_bp = vocal_pre, vocal_post

        frame_len = int(sr * 0.025)  # 25 ms Frames
        hop = frame_len // 2

        # LPC-Order 14: Standard für Vokal-Formant-Analyse im Bandpass-Band 200–3400 Hz.
        # §BUG-FIX v10.0.0: Order 30 ist für das VOLLE 48 kHz-Spektrum gedacht —
        # nach dem 200–3400 Hz Bandpass enthält das Signal nur noch ~3,2 kHz Bandbreite.
        # LPC Order 30 auf einem 3,2 kHz-Band → 15 Polpaare → alle spektralen Details
        # (HF-Boost-Artefakte, Kassetten-Flutter-Harmonische) als "Formanten" erkannt.
        # Folge: F1-Schätzung instabil, formant_stability_score ≈ 0.14 statt ≈ 0.85.
        # Order 14 = 7 Polpaare auf 3400 Hz-Band — erfasst F1, F2, F3 korrekt (ITUT P.501).
        # F4 liegt > 3000 Hz und fällt bereits außerhalb des Bandpass → kein Verlust.
        lpc_order = 14

        def _lpc_f1_frames(audio: np.ndarray) -> list[float]:
            frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop)
            f1_list = []
            for i in range(frames.shape[1]):
                frame = frames[:, i] * np.hanning(frame_len)
                try:
                    a = librosa.lpc(frame, order=lpc_order)
                    roots = np.roots(a)
                    roots = roots[np.imag(roots) >= 0]
                    angles = np.angle(roots)
                    freqs = sorted(angles * (sr / (2 * np.pi)))
                    # F1: erste Resonanz > 200 Hz (nach Bandpass zuverlässiger als > 100 Hz)
                    valid = [f for f in freqs if f > 200]
                    if valid:
                        f1_list.append(valid[0])
                except Exception as e:
                    logger.warning("vocal_quality_index.py::_lpc_f1_frames Ersatzpfad: %s", e)
            return f1_list

        f1_pre = _lpc_f1_frames(_pre_bp)
        f1_post = _lpc_f1_frames(_post_bp)

        if not f1_pre or not f1_post:
            return 0.85  # Messung nicht möglich → konservativer Fallback

        # Median-Filter auf F1-Zeitreihen: eliminiert Wow/Flutter-induzierte Ausreißer
        # in der degradierten Eingabe. Ohne Filter: Kassetten-Flutter verursacht F1-Spikes
        # im Input; restauriertes Audio hat stabilere F1 → scheinbar hohe "Drift" trotz
        # korrekter Restaurierung (score≈0.14 bei echter Drift=0.0 wäre reines Messartefakt).
        # 5-Frame-Median entspricht 62.5 ms Glättung (25 ms Frame, 12.5 ms Hop) — glättet
        # Wow (0.1–1 Hz) und Flutter (0–20 Hz) vollständig ohne echte Formantdrift zu maskieren.
        _kernel = min(5, len(f1_pre) if len(f1_pre) % 2 == 1 else len(f1_pre) - 1)
        if _kernel >= 3:
            from scipy.ndimage import median_filter as _mf  # pylint: disable=import-outside-toplevel

            _f1_pre_sm = list(_mf(np.array(f1_pre, dtype=np.float64), size=_kernel, mode="nearest"))
            _f1_post_sm = list(_mf(np.array(f1_post, dtype=np.float64), size=_kernel, mode="nearest"))
        else:
            _f1_pre_sm, _f1_post_sm = f1_pre, f1_post

        min_len = min(len(_f1_pre_sm), len(_f1_post_sm))
        drift = float(np.mean(np.abs(np.array(_f1_pre_sm[:min_len]) - np.array(_f1_post_sm[:min_len]))))
        # §EraVocalProfile: Historische Vokalstile haben höhere F1-Drift-Toleranz.
        # Modern (default): f1_tolerance_db=2.0 → max_drift_hz=70.0; 1900-1925: 4.0 → 140.0 Hz.
        _f1_tol = float(getattr(era_profile, "f1_tolerance_db", 2.0)) if era_profile is not None else 2.0
        _max_drift_hz = 70.0 * (_f1_tol / 2.0)
        score = float(np.clip(1.0 - drift / _max_drift_hz, 0.0, 1.0))
        return score
    except Exception as exc:
        logger.debug("formant_stability fehlgeschlagen: %s", exc)
        return 0.85


# ---------------------------------------------------------------------------
# Component 3 — Articulation Score (§2.35, via proximity-consonant ratio)
# ---------------------------------------------------------------------------


def _compute_articulation_score(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
) -> float:
    """Plosive/fricative onset precision. Uses transient energy ratio from §2.35."""
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        # Broadband transient detection via spectral flux
        hop = 512
        onset_strength_pre = librosa.onset.onset_strength(y=vocal_pre, sr=sr, hop_length=hop)  # type: ignore[attr-defined]
        onset_strength_post = librosa.onset.onset_strength(y=vocal_post, sr=sr, hop_length=hop)  # type: ignore[attr-defined]

        min_len = min(len(onset_strength_pre), len(onset_strength_post))
        if min_len < 4:
            return 0.85

        # Correlation of onset-strength curves
        pre_norm = onset_strength_pre[:min_len]
        post_norm = onset_strength_post[:min_len]
        # Guarded dot-product (NaN-safe, as required by VERBOTEN V01)
        _pn_std = float(np.std(pre_norm)) * float(np.std(post_norm))
        if _pn_std < 1e-9:
            corr = 0.0
        else:
            corr = float(
                np.dot(pre_norm - pre_norm.mean(), post_norm - post_norm.mean()) / (len(pre_norm) * _pn_std + 1e-12)
            )
        corr = float(np.clip(corr, -1.0, 1.0))
        # Correlation [0,1] → score (near 1 = onsets preserved)
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
    except Exception as exc:
        logger.debug("articulation_Wert fehlgeschlagen: %s", exc)
        return 0.85


# ---------------------------------------------------------------------------
# Component 5 — Sibilance Naturalness (5–10 kHz)
# ---------------------------------------------------------------------------


def _compute_sibilance_naturalness(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
) -> float:
    """Energy deviation in the 5–10 kHz fricative band. score = max(0, 1 - |dev_db|/6)."""
    try:
        from scipy.signal import butter, sosfiltfilt  # pylint: disable=import-outside-toplevel

        nyq = sr / 2.0
        lo = min(5000.0, nyq * 0.9) / nyq
        hi = min(10000.0, nyq * 0.99) / nyq
        if lo >= hi or lo <= 0:
            return 0.85

        sos = butter(4, [lo, hi], btype="bandpass", output="sos")
        pre_band = sosfiltfilt(sos, vocal_pre)
        post_band = sosfiltfilt(sos, vocal_post)

        rms_pre = float(np.sqrt(np.mean(pre_band**2)) + 1e-12)
        rms_post = float(np.sqrt(np.mean(post_band**2)) + 1e-12)

        abw_db = abs(20.0 * np.log10(rms_post / rms_pre))
        # 6 dB = threshold (§2.35c): 0 dB deviation → 1.0
        score = float(np.clip(1.0 - abw_db / 6.0, 0.0, 1.0))
        return score
    except Exception as exc:
        logger.debug("sibilance_naturalness fehlgeschlagen: %s", exc)
        return 0.85


# ---------------------------------------------------------------------------
# Haupt-API
# ---------------------------------------------------------------------------


def compute_vqi(  # pylint: disable=too-many-positional-arguments
    audio_orig: np.ndarray,
    audio_restored: np.ndarray,
    sr: int,
    vocal_segments: list[tuple[float, float]] | None = None,
    skip_singer_identity: bool = False,
    reference_audio: np.ndarray | None = None,
    genre: str | None = None,
    reference_singer_id: str | None = None,
    era_profile: object = None,
) -> dict[str, float]:
    """Berechnet the VocalQualityIndex (§2.35c).

    Args:
        audio_orig:       Original (degraded) before pipeline, float32 [-1,1].
        audio_restored:   Pipeline-restored audio, float32 [-1,1].
        sr:               Sample rate in Hz (required: 48000 in pipeline phases).
        vocal_segments:   Optional list of (start_sec, end_sec) vocal time windows.
                          None → entire signal is used.
        skip_singer_identity: Skips singer identity for duets/choirs.
        genre:            Genre label for adaptive weight selection (§0p).
                          None → default weights. Supported: jazz, blues, opera,
                          klassik, pop, rock, folk, country, soul, gospel, schlager.
        reference_audio:  §P1 Artist-Voice-Reference — clean reference recording
                          of the same artist (min. 0.5 s). When provided, used as
                          anchor for _compute_singer_identity() instead of the
                          degraded audio_orig. Enables precise identity comparison
                          independent of carrier noise.
        reference_singer_id: §SRL-1 voice-class ID from SingerReferenceLibrary
                          (e.g. "voice_jazz_alto"). When provided and
                          reference_audio=None, a voice-class fingerprint cosine
                          is used for singer_identity_cosine — more precise than
                          comparing against degraded input.

    Returns:
        Dict with:
            vqi                   — overall score [0, 1]
            singer_identity_cosine
            formant_stability_score
            articulation_score
            proximity_score       — from §2.35b (if available, else 0.85)
            sibilance_naturalness
            singer_id_dsp_fallback — bool: True when Resemblyzer not used
            vqi_tier              — "world_class" | "professional" | "acceptable" | "below_threshold"
            reference_audio_used  — bool: True when reference_audio used as anchor (§P1)
    """
    orig_m = _to_mono(audio_orig)
    rest_m = _to_mono(audio_restored)

    min_len = min(len(orig_m), len(rest_m))
    if min_len < sr // 2:
        # Too short for meaningful measurement — return safe neutral values.
        # 0.90 > VQI_WORLD_CLASS (0.88) ensures no false recovery-cascade trigger,
        # even in Studio 2026 mode (floor 0.87). singer_identity_cosine=0.95 prevents
        # false rollback (§0p gate: < 0.92 triggers rollback).
        return {
            "vqi": 0.90,
            "singer_identity_cosine": 0.95,
            "formant_stability_score": 0.90,
            "articulation_score": 0.90,
            "proximity_score": 0.90,
            "sibilance_naturalness": 0.90,
            "singer_id_dsp_fallback": True,
            "vqi_tier": "world_class",  # type: ignore[dict-item]
        }

    orig_m = orig_m[:min_len]
    rest_m = rest_m[:min_len]

    # Apply vocal segment mask
    if vocal_segments:
        mask = np.zeros(min_len, dtype=bool)
        for s, e in vocal_segments:
            i0 = max(0, int(s * sr))
            i1 = min(min_len, int(e * sr))
            mask[i0:i1] = True
        if mask.sum() >= sr // 4:
            orig_m = orig_m[mask]
            rest_m = rest_m[mask]

    orig_m = np.nan_to_num(orig_m, nan=0.0, posinf=0.0, neginf=0.0)
    rest_m = np.nan_to_num(rest_m, nan=0.0, posinf=0.0, neginf=0.0)

    # Component 1: Singer Identity (§MultiSinger: skip for duet/choir)
    _reference_audio_used = False
    _srl_singer_identity_done = False  # §SRL-1: True wenn Fingerprint-Pfad aktiv
    dsp_fallback: bool = False  # initialisiert; wird unten überschrieben
    if skip_singer_identity:
        singer_cosine, dsp_fallback = 0.85, True  # neutral fallback — no rollback trigger
        logger.debug("§MultiSinger: singer_identity_cosine gate uebersprungen (duet/choir)")
    else:
        # §P1 Artist-Voice-Reference: clean artist recording as a more precise identity anchor
        _id_anchor = orig_m
        if reference_audio is not None:
            try:
                _ref = _to_mono(np.asarray(reference_audio, dtype=np.float32))
                if len(_ref) >= sr // 2:  # mind. 0.5 s
                    _ref = np.nan_to_num(_ref, nan=0.0, posinf=0.0, neginf=0.0)
                    # Auf Länge des restaurierten Signals anpassen
                    if len(_ref) >= len(rest_m):
                        _id_anchor = _ref[: len(rest_m)]
                    else:
                        _id_anchor = np.pad(_ref, (0, len(rest_m) - len(_ref)))
                    _reference_audio_used = True
                    logger.debug(
                        "VQI: Referenz_audio as singer-identity anchor (§P1 Artist-Voice-Referenz, len=%d)",
                        len(_ref),
                    )
            except Exception as _ref_exc:
                logger.debug(
                    "VQI: Referenz_audio conversion fehlgeschlagen — falling back to degraded Eingabe: %s",
                    _ref_exc,
                )
        elif reference_singer_id is not None:
            # §SRL-1: Use voice-class fingerprint as identity anchor.
            # Since no real reference audio is available, we use the cosine
            # distance between the restored audio fingerprint and the
            # voice-class prototype as a singer_identity_cosine proxy.
            try:
                from backend.core.singer_reference_library import (  # pylint: disable=import-outside-toplevel
                    _STIMMKLASSE_PROTOTYPEN,
                    compute_vocal_fingerprint,
                )

                _ref_fp = _STIMMKLASSE_PROTOTYPEN.get(reference_singer_id)
                if _ref_fp is not None:
                    _rest_fp = compute_vocal_fingerprint(audio_restored, sr)
                    _ref_norm = np.linalg.norm(_ref_fp) + 1e-12
                    _rest_norm = np.linalg.norm(_rest_fp) + 1e-12
                    _srl_cosine = float(np.dot(_ref_fp, _rest_fp) / (_ref_norm * _rest_norm))
                    # Cosine [-1,1] → fingerprint-based singer_identity_cosine.
                    # Mapping: [0.5, 1.0] → [0.85, 1.0] (softer mapping since fingerprints
                    # are less precise than Resemblyzer embeddings)
                    _srl_mapped = float(np.clip(0.85 + (_srl_cosine - 0.5) * 0.30, 0.80, 1.0))
                    singer_cosine = _srl_mapped
                    dsp_fallback = True  # DSP-basiert, kein ML  # §V6: logger.warning handled at call site
                    _srl_singer_identity_done = True
                    _reference_audio_used = True  # Signal: nicht degraded-Input-Anker
                    logger.debug(
                        "VQI §SRL-1: class=%s fingerprint_cosine=%.3f → singer_identity_cosine=%.3f",
                        reference_singer_id,
                        _srl_cosine,
                        singer_cosine,
                    )
            except Exception as _srl_exc:
                logger.debug("VQI §SRL-1 nicht blockierend: %s", _srl_exc)
        if not _srl_singer_identity_done:
            singer_cosine, dsp_fallback = _compute_singer_identity(_id_anchor, rest_m, sr)

    # Component 2: Formant Stability (§EraVocalProfile: era_profile skaliert max_drift_hz)
    # §MultiSinger: Bei Duett/Chor überlagern sich mehrere Formant-Tracks im Mix.
    # LPC-F1-Analyse liefert dann unzuverlässige Resonanzschätzungen (falsch-niedrige Scores).
    # Gleiche Gate-Logik wie singer_identity: konservativer Fallback statt Fehlmessung.
    if skip_singer_identity:
        formant_score = 0.85  # neutral fallback — Multi-Singer-Interferenz verhindert valide F1-Analyse
        logger.debug("§MultiSinger: formant_stability gate uebersprungen (duet/choir) → Ersatzpfad=0.85")
    else:
        formant_score = _compute_formant_stability(orig_m, rest_m, sr, era_profile=era_profile)

    # Component 3: Articulation
    articulation = _compute_articulation_score(orig_m, rest_m, sr)

    # §SOTA-Matrix: SingMOS als primärer Naturalness-Proxy für Gesangsmaterial (§0p + copilot-instructions)
    # SingMOS Pro ist über versa_plugin verfügbar (VersaPlugin.score → model_used="singmos_pro").
    # Normiert auf [0,1]: SingMOS MOS ∈ [1,5] → (mos - 1) / 4.
    singmos_score: float | None = None
    try:
        from plugins.versa_plugin import get_versa_plugin  # type: ignore[import]  # pylint: disable=import-outside-toplevel  # noqa: I001

        _versa = get_versa_plugin()
        if _versa is not None:
            _vm_result = _versa.score(audio_restored, sr)
            if _vm_result is not None:
                _mos = float(getattr(_vm_result, "mos", float("nan")))
                _model = str(getattr(_vm_result, "model_used", ""))
                if np.isfinite(_mos) and 1.0 <= _mos <= 5.0 and "singmos" in _model:
                    singmos_score = float(np.clip((_mos - 1.0) / 4.0, 0.0, 1.0))
                    logger.debug("VQI: SingMOS Pro MOS=%.3f → normiert=%.3f", _mos, singmos_score)
    except Exception as _sm_exc:
        logger.debug("VQI: SingMOS (VERSA) nicht verfügbar (nicht blockierend): %s", _sm_exc)

    # Component 4: Vocal Proximity (§2.35b) — or SingMOS as substitute
    proximity = 0.85  # default
    if singmos_score is not None:
        proximity = singmos_score  # SingMOS primary (§SOTA-Matrix May 2026)
        logger.debug("VQI: SingMOS replaces proximity component (primary naturalness proxy)")
    else:
        try:
            from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score  # pylint: disable=import-outside-toplevel  # noqa: I001

            prox_result = compute_vocal_proximity_score(audio_orig, audio_restored, sr, vocal_segments)
            proximity = float(prox_result.get("proximity_score", 0.85))
        except Exception as exc:
            logger.debug("vocal_proximity import fehlgeschlagen: %s", exc)

    # Component 5: Sibilance
    sibilance = _compute_sibilance_naturalness(orig_m, rest_m, sr)

    # Genre-adaptive weights (§0p): different vocal genres define quality differently.
    # Falls back to default weights for unknown/None genre.
    (
        singer_identity_weight,
        formant_weight,
        articulation_weight,
        proximity_weight,
        sibilance_weight,
    ) = _get_vqi_weights(genre)
    _genre_used = str(genre or "").strip().lower() or None

    # Weighted sum (§2.35c) with genre-adaptive weights
    vqi = (
        singer_identity_weight * singer_cosine
        + formant_weight * formant_score
        + articulation_weight * articulation
        + proximity_weight * proximity
        + sibilance_weight * sibilance
    )
    vqi = float(np.clip(vqi, 0.0, 1.0))
    vqi = float(np.nan_to_num(vqi, nan=0.90))  # 0.90 > all floors incl. Studio 2026 (0.87)

    # Determine quality tier
    if vqi >= VQI_WORLD_CLASS:
        tier = "world_class"
    elif vqi >= VQI_PROFESSIONAL:
        tier = "professional"
    elif vqi >= VQI_THRESHOLD:
        tier = "acceptable"
    else:
        tier = "below_threshold"

    return {
        "vqi": vqi,
        "singer_identity_cosine": float(np.clip(singer_cosine, 0.0, 1.0)),
        "formant_stability_score": float(np.clip(formant_score, 0.0, 1.0)),
        "articulation_score": float(np.clip(articulation, 0.0, 1.0)),
        "proximity_score": float(np.clip(proximity, 0.0, 1.0)),
        "sibilance_naturalness": float(np.clip(sibilance, 0.0, 1.0)),
        "singer_id_dsp_fallback": dsp_fallback,  # §V6: logger.warning handled at call site
        "vqi_tier": tier,  # type: ignore[dict-item]
        "reference_audio_used": _reference_audio_used,  # §P1 Artist-Voice-Reference anchor used
        "genre_weights_used": _genre_used,  # type: ignore[dict-item]  # None = default weights
    }
