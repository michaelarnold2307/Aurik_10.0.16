"""
denker/exzellenz_denker.py — Exzellenz-Denker für Aurik 10.0.0
==========================================================

Verbessert restauriertes Audio bis an die physikalische Qualitätsgrenze:
  • ExcellenceOptimizer: Spectral Continuity, Micro-Dynamics, Harmonic Boost, OLA
    • MusicalGoalsChecker: Messung aller 15 musikalischen Qualitätsziele

Spec §1.2, §2.5, §2.29, §8.1 — v10.0.0
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Eager top-level import verhindert Import-Lock-Deadlock in langen Test-Läufen
# (§3.4 Lazy Imports / Spec Threading-Invariante)
try:
    from backend.core.vocal_ai_enhancement import (
        EmotionPreservationMode,
        UnifiedVocalAIEnhancer,
    )
except Exception:  # ImportError, AttributeError, etc.
    UnifiedVocalAIEnhancer = None  # type: ignore[assignment]
    EmotionPreservationMode = None  # type: ignore[assignment]

_3X_RT_LIMIT: float = 32.0  # Maximaler RT-Faktor (Spec §9.5)


# ─── Goal-Risk-Assessment (Muster D: gemeinsame Basis für prognostiziere + optimiere) ──
# Zentral definierte Goal-Thresholds: beide Pfade nutzen dieselben Werte.


_GOAL_THRESHOLDS: dict[str, float] = {
    "authentizitaet": 0.70,
    "natuerlichkeit": 0.70,  # naturalness
    "brillanz": 0.65,
    "timbre": 0.65,
    "groove": 0.60,
    "micro_dynamics": 0.60,
    "artikulation": 0.65,
    "waerme": 0.70,
    "tiefe": 0.65,  # depth / räumliche Tiefe
    "durchsetzung": 0.65,  # presence / Durchsetzung
    "transparenz": 0.65,
    "kohaerenz": 0.65,  # coherence
    "fokus": 0.60,
    "balance": 0.65,
    "stimmung": 0.60,  # emotional mood
}
"""Zentrale Bestehensgrenzen für alle 15 Musical Goals.

Ein Goal gilt als bestanden wenn sein Score ≥ diesem Wert liegt.
Ein Goal gilt als risikobehaftet (repair-würdig) wenn sein Score < Wert × 0.85.
"""


@dataclass
class GoalRiskAssessment:
    """Einheitliche Risikobewertung eines Musical Goals.

    Von ExzellenzDenker.bewerte_zielrisiko() erzeugt, sowohl von
    prognostiziere() (DSP-Proxy) als auch optimiere() (full measurement) genutzt.
    """

    goal_name: str
    """Name des Musical Goals (z. B. 'waerme', 'authentizitaet')."""

    current_score: float
    """Gemessener oder prognostizierter Score ∈ [0, 1]."""

    threshold: float
    """Bestehensgrenze aus _GOAL_THRESHOLDS."""

    risk: float
    """Risiko ∈ [0, 1]: 0 = sicher bestanden, 1 = sicher verletzt.
    Berechnet als norm(threshold - score) mit Sigmoid-Charakter.
    """

    needs_protection: bool
    """True wenn prophylaktische Phase injiziert werden sollte (risk ≥ 0.50)."""

    needs_repair: bool
    """True wenn reaktive Reparatur nötig ist (score < threshold × 0.85)."""

    @property
    def passed(self) -> bool:
        """True wenn der Score über der Bestehensgrenze liegt."""
        return self.current_score >= self.threshold


# ─── Ergebnis-Datenklasse ────────────────────────────────────────────────────


@dataclass
class ExzellenzErgebnis:
    """Ergebnis der Exzellenz-Optimierung mit Musical-Goals-Bewertung.

    Felder:
        audio:           Optimiertes Audio (float32, clip [-1, 1])
        excellence_score: Gesamt-Exzellenz-Score ∈ [0, 1]
        musical_goals:   14 Musical-Goals-Scores (Dict[str, float])
        goals_passed:    Anzahl bestandener Goals
        goals_total:     Gesamtanzahl gemessener Goals
        improvements:    Liste der angewendeten Optimierungsschritte (Deutsch)
        processing_note: Kurzbeschreibung auf Deutsch
        warnings:        Warnungsliste (technisch, Englisch)
    """

    audio: np.ndarray
    excellence_score: float
    musical_goals: dict[str, float]
    goals_passed: int
    goals_total: int
    improvements: list[str]
    processing_note: str
    warnings: list[str] = field(default_factory=list)
    versa_mos: float = 0.0
    """VERSA MOS-Score \u2208 [1, 5] gemessen am optimierten Audio (0.0 = nicht verfügbar).
    Wird von AurikDenker wiederverwendet, um doppelte VERSA-Inferenz zu vermeiden.
    """
    frisson_index: float = 0.0
    """Gänsehaut-Propensity \u2208 [0, 1] aus Goals-Proxy (Blood & Zatorre 2001, §2.53).
    0.0 = nicht berechnet oder keine Goals verfügbar.
    """
    mert_proxy_used: bool = False
    """True wenn VERSA fehlschlug und MERT als Proxy-Fallback verwendet wurde (§2.44)."""

    def as_dict(self) -> dict:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "excellence_score": float(self.excellence_score),
            "musical_goals": {k: float(v) for k, v in self.musical_goals.items()},
            "goals_passed": self.goals_passed,
            "goals_total": self.goals_total,
            "improvements": list(self.improvements),
            "processing_note": self.processing_note,
            "warnings": list(self.warnings),
            "frisson_index": float(self.frisson_index),
            "mert_proxy_used": bool(self.mert_proxy_used),
        }


# ─── Denker-Klasse ───────────────────────────────────────────────────────────


class ExzellenzDenker:
    """Optimiert restauriertes Audio auf höchstes musikalisches Niveau.

    Kombiniert den ExcellenceOptimizer (vier DSP-Pfade) mit dem
    MusicalGoalsChecker (14 Qualitätsziele) zu einer vollständigen
    Exzellenz-Bewertungseinheit.

    Verwendung::

        exd = get_exzellenz_denker()
        ergebnis = exd.optimiere(audio, sr=48_000, material="tape")
        logger.debug(ergebnis.musical_goals)
    """

    def __init__(self) -> None:
        self._optimizer: Any = None
        self._opt_lock = threading.Lock()
        self._checker: Any = None
        self._checker_lock = threading.Lock()

    # ── Öffentliche API ──────────────────────────────────────────────────────

    def optimiere(
        self,
        audio: np.ndarray,
        sr: int = 48_000,
        *,
        material: str = "auto",
        _messe_ziele_vorab: bool = True,
    ) -> ExzellenzErgebnis:
        """Optimiert Audio auf CEDAR Cambridge-Niveau und misst alle 15 Goals.

        Ablauf:
            1. NaN/Inf-Schutz
            2. Optional: Musical Goals vor Optimierung messen (Baseline)
            3. ExcellenceOptimizer: Spectral Continuity + Micro-Dynamics +
               Harmonic Boost + OLA-Crossfade
            4. Musical Goals nach Optimierung messen
            5. Excellence-Score aus Goals berechnen

        Args:
            audio:              Audio-Signal (mono/stereo, float32)
            sr:                 Sample-Rate in Hz (Standard 48 000)
            material:           Trägermedium-Bezeichnung für GP-Priors
            messe_ziele_vorab:  Ob Baseline-Goals gemessen werden sollen

        Returns:
            ExzellenzErgebnis mit optimiertem Audio und allen Metriken
        """
        assert sr == 48000, f"ExzellenzDenker.optimiere() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        # NaN/Inf-Schutz (Spec §3.1)
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if audio.size == 0:
            return self._fallback(audio, "Leeres Eingabe-Audio")

        warnings: list[str] = []
        improvements: list[str] = []
        _metadata: dict = {}  # §2.44 MERT-Proxy-Flag und interne Telemetrie

        # Schritt 1 — Excellence-Optimierung
        try:
            opt = self._get_optimizer(sr=sr, material=material)
            optimiertes_audio, result = opt.optimize(audio)
            improvements = list(getattr(result, "applied_steps", []))
            logger.info(
                "ExzellenzDenker: Optimierung abgeschlossen — %s",
                getattr(result, "summary", lambda: "ok")(),
            )
        except Exception as exc:
            warnings.append(f"ExcellenceOptimizer nicht verfügbar: {exc}")
            optimiertes_audio = audio.copy()
            logger.warning("ExzellenzDenker: Excellence-Optimierung fehlgeschlagen: %s", exc)

        # NaN/Inf nach Optimierung prüfen
        optimiertes_audio = np.nan_to_num(optimiertes_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        optimiertes_audio = np.clip(optimiertes_audio, -1.0, 1.0)

        # ── Vokal-KI-Verbesserung (material-abhängig) ─────────────────────────
        _VOCAL_MATERIALS = {"gesang", "sprache", "vocal", "singer", "speech", "voice"}
        if material.lower().strip() in _VOCAL_MATERIALS:
            try:
                if UnifiedVocalAIEnhancer is None or EmotionPreservationMode is None:
                    raise ImportError("core.vocal_ai_enhancement nicht verfügbar")
                _vocal_enh = UnifiedVocalAIEnhancer(sample_rate=sr)
                _vocal_result = _vocal_enh.enhance(
                    optimiertes_audio,
                    emotion_mode=EmotionPreservationMode.BALANCED,
                    breath_preservation=0.7,
                    sibilance_reduction=True,
                )
                optimiertes_audio = _vocal_result.audio
                logger.info(
                    "ExzellenzDenker Vokal-KI: Qualität +%.3f, Phasen=%s",
                    _vocal_result.quality_improvement,
                    _vocal_result.processing_applied,
                )
            except Exception as _exc:
                logger.warning("ExzellenzDenker VocalAIEnhancer: %s", _exc)

        # ── VERSA MOS: Qualitätssignal für Entscheidung über Nachbearbeitung (§4.4) ──
        # VERSA 2024 (non-reference MOS) gibt ein unabhängiges Klangsignal,
        # das entscheidet ob eine weitere Verarbeitungsrunde sinnvoll ist.
        versa_mos: float = 0.0
        try:
            from plugins.versa_plugin import score_mos

            _mono = optimiertes_audio if optimiertes_audio.ndim == 1 else optimiertes_audio.mean(axis=-1)
            _mono = np.nan_to_num(_mono.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            _vr = score_mos(_mono, sr)
            versa_mos = float(_vr.mos)
            logger.info(
                "ExzellenzDenker VERSA MOS=%.3f (%s) — Entscheidungssignal",
                versa_mos,
                _vr.model_used,
            )
            if versa_mos < 3.5:
                warnings.append(
                    f"VERSA MOS={versa_mos:.2f} < 3.5 — Qualität unter Mindestniveau, Nachbearbeitung empfohlen"
                )
            elif versa_mos >= 4.3:
                improvements.append(f"VERSA MOS={versa_mos:.2f} ≥ 4.3 — Studioqualität erreicht")
        except Exception as _ve:
            logger.debug("ExzellenzDenker: VERSA MOS nicht verfügbar: %s", _ve)
            # §2.44 VERBOTEN: MERT darf nicht primary sein wenn VERSA verfügbar.
            # Hier: VERSA fehlgeschlagen → MERT als Proxy-Fallback (§2.44).
            try:
                from plugins.mert_plugin import get_mert_plugin as _get_mert

                _mert = _get_mert()
                _mono_mert = optimiertes_audio if optimiertes_audio.ndim == 1 else optimiertes_audio.mean(axis=-1)
                _mono_mert = np.nan_to_num(_mono_mert.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _mert_analysis = _mert.analyze(_mono_mert, sr)
                versa_mos = float(_mert_analysis.naturalness_score) * 5.0  # MERT [0,1] → MOS-Skala
                logger.info(
                    "ExzellenzDenker MERT-Proxy MOS=%.3f (VERSA unavailable, §2.44 fallback)",
                    versa_mos,
                )
                _metadata["mert_proxy_used"] = True  # §2.44 VERBOTEN-Invariante
            except Exception as _mert_exc:
                logger.debug("ExzellenzDenker: MERT-Proxy MOS nicht verfügbar: %s", _mert_exc)

        # Schritt 2 — Musical Goals messen
        goals = self.messe_ziele(optimiertes_audio, sr)

        # Excellence-Score: Mittelwert aller Goals
        # §09.2 Material-adaptive Mindest-Score (nicht hardcoded 0.75 — Shellac-Ceiling 0.70,
        # Vinyl 0.88, CD 0.95 → 0.75 wäre für schwierige Materialien systematisch falsch).
        try:
            from backend.core.calibration_matrix import _MATERIAL_QUALITY_CEILING as _MQC

            _mat_norm_ed = str(material or "auto").strip().lower()
            _goal_min_raw = _MQC.get(_mat_norm_ed, _MQC.get("vinyl", 0.75)) * 0.85
            _GOAL_MIN: float = float(np.clip(_goal_min_raw, 0.50, 0.75))
        except Exception:
            _GOAL_MIN = 0.75
        if goals:
            finite_vals = [v for v in goals.values() if np.isfinite(v)]
            score = float(np.mean(finite_vals)) if finite_vals else 0.0
            passed = sum(1 for v in finite_vals if v >= _GOAL_MIN)
        else:
            score = 0.0
            passed = 0

        # Schritt 3 — Musical Goals Re-Pass bei Violations (max. 1 Runde)
        # v10.0.0: Von 3 auf 1 Re-Pass reduziert — wissenschaftliche Begründung:
        # Kaskadierte identische Verarbeitung akkumuliert STFT-Rundungsfehler
        # (Ephraim & Malah 1984) und verschiebt ML-Modelle in untrainierte Domains.
        # Ein einzelner korrigierender Pass mit degressiver Intensität ist optimal.
        _violations = [k for k, v in goals.items() if math.isfinite(v) and v < _GOAL_MIN] if goals else []
        _MAX_RE_PASSES = 1
        for _re_pass_i in range(1, _MAX_RE_PASSES + 1):
            _violations = [k for k, v in goals.items() if math.isfinite(v) and v < _GOAL_MIN] if goals else []
            if not _violations:
                break
            try:
                _opt_rp = self._get_optimizer(sr=sr, material=material)
                # Degressive intensity: reduce harmonic boost + modulation each pass
                _opt_rp._harm_boost_db *= max(0.3, 1.0 - 0.3 * _re_pass_i)
                _opt_rp._modulation_strength *= max(0.3, 1.0 - 0.25 * _re_pass_i)
                _rp_audio, _rp_result = _opt_rp.optimize(optimiertes_audio)
                _rp_audio = np.clip(
                    np.nan_to_num(_rp_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                    -1.0,
                    1.0,
                )
                _rp_goals = self.messe_ziele(_rp_audio, sr)
                _rp_passed = sum(1 for v in _rp_goals.values() if math.isfinite(v) and v >= _GOAL_MIN)
                # Re-Pass nur übernehmen wenn er Goals verbessert
                if _rp_passed >= passed:
                    optimiertes_audio = _rp_audio
                    goals = _rp_goals
                    finite_vals = [v for v in goals.values() if math.isfinite(v)]
                    score = float(np.mean(finite_vals)) if finite_vals else score
                    passed = _rp_passed
                    improvements.append(
                        f"Re-Pass {_re_pass_i}: {len(_violations)} Violations → {passed}/{len(goals)} Goals bestanden"
                    )
                    logger.info(
                        "ExzellenzDenker Re-Pass %d: %d Violations korrigiert → %d/%d Goals",
                        _re_pass_i,
                        len(_violations),
                        passed,
                        len(goals),
                    )
                else:
                    logger.debug(
                        "ExzellenzDenker Re-Pass %d: keine Verbesserung (%d→%d Goals), abgebrochen",
                        _re_pass_i,
                        passed,
                        _rp_passed,
                    )
                    break  # No improvement → stop re-passing
            except Exception as _rp_exc:
                logger.debug("ExzellenzDenker Re-Pass %d fehlgeschlagen: %s", _re_pass_i, _rp_exc)
                break

        # §2.53a VERSA Re-measurement nach Goal-Re-Pass: optimiertes_audio könnte durch
        # Re-Pass geändert worden sein. VERSA neu messen für post-repair MOS.
        # (Die initiale VERSA-Messung L186–200 war VOR Re-Pass → hier re-messen.)
        if optimiertes_audio.shape != audio.shape or not np.array_equal(optimiertes_audio, audio):
            try:
                from plugins.versa_plugin import score_mos as _score_mos_post

                _mono_post = optimiertes_audio if optimiertes_audio.ndim == 1 else optimiertes_audio.mean(axis=-1)
                _mono_post = np.nan_to_num(_mono_post.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _vr_post = _score_mos_post(_mono_post, sr)
                _mos_post = float(getattr(_vr_post, "mos", 1.0))
                if math.isfinite(_mos_post) and 1.0 <= _mos_post <= 5.0:
                    logger.info(
                        "ExzellenzDenker VERSA post-repair re-measurement: MOS=%.3f (pre-repair war %.3f)",
                        _mos_post,
                        versa_mos,
                    )
                    versa_mos = _mos_post  # Überschreibe pre-repair Wert
                else:
                    logger.debug("ExzellenzDenker: VERSA post-repair MOS ungültig: %.1f", _mos_post)
            except Exception as _ve:
                logger.debug("ExzellenzDenker: VERSA post-repair Messung fehlgeschlagen: %s", _ve)

        # §2.53 Frisson-Index (Gänsehaut-Propensity) aus Goals-Proxy
        # ── §v10 Gänsehaut-Faktor aus psychoakustischem Modell ──
        # Ersetzt den alten frisson_index (Musical-Goal-Proxy), der nur technische
        # Metriken kombinierte. Jetzt: echtes psychoakustisches Modell mit
        # Dynamic Contrast, Harmonic Surprise, Spectral Shimmer, Temporal Breath,
        # Frequency Warmth — basierend auf Sloboda 1991, Blood & Zatorre 2001.
        frisson_index = 0.0
        _goose_label = ""
        try:
            from backend.core.goosebumps_factor import compute_goosebumps

            _goose_r = compute_goosebumps(optimiertes_audio, sr)
            frisson_index = float(_goose_r.score)
            _goose_label = _goose_r.label
            logger.info(
                "ExzellenzDenker frisson_index (Goosebumps)=%.3f (%s): "
                "dynamic=%.2f harmonic=%.2f shimmer=%.2f breath=%.2f warmth=%.2f",
                frisson_index,
                _goose_label,
                _goose_r.dynamic_contrast,
                _goose_r.harmonic_surprise,
                _goose_r.spectral_shimmer,
                _goose_r.temporal_breath,
                _goose_r.frequency_warmth,
            )
        except Exception:
            # Fallback: Musical-Goal-Proxy (alt, aber besser als 0.0)
            _fi_micro = float(goals.get("micro_dynamics", 0.0)) if goals else 0.0
            _fi_emo = float(goals.get("emotionalitaet", 0.0)) if goals else 0.0
            _fi_arc = 0.5 * _fi_emo + 0.5 * _fi_micro
            _fi_art = float(goals.get("artikulation", 0.0)) if goals else 0.0
            _fi_spa = float(goals.get("spatial_depth", 0.0)) if goals else 0.0
            _fi_trans = float(goals.get("transparenz", 0.0)) if goals else 0.0
            _fi_tonal = float(goals.get("tonal_center", 0.0)) if goals else 0.0
            frisson_index = float(
                np.clip(
                    0.26 * _fi_arc
                    + 0.18 * _fi_micro
                    + 0.14 * _fi_emo
                    + 0.14 * _fi_art
                    + 0.10 * _fi_spa
                    + 0.08 * _fi_trans
                    + 0.10 * _fi_tonal,
                    0.0,
                    1.0,
                )
            )

        note = (
            f"Exzellenz-Optimierung abgeschlossen: Score {score:.3f}, "
            f"{passed}/{len(goals)} Ziele erfüllt"
            + (f", VERSA MOS={versa_mos:.2f}" if versa_mos > 0.0 else "")
            + (f", Gänsehaut={frisson_index:.0%}" if frisson_index > 0.0 else "")
            + "."
        )

        return ExzellenzErgebnis(
            audio=optimiertes_audio,
            excellence_score=score,
            musical_goals=goals,
            goals_passed=passed,
            goals_total=len(goals),
            improvements=improvements,
            processing_note=note,
            warnings=warnings,
            versa_mos=versa_mos,
            frisson_index=frisson_index,
            mert_proxy_used=bool(_metadata.get("mert_proxy_used", False)),
        )

    def messe_ziele(self, audio: np.ndarray, sr: int, reference: np.ndarray | None = None) -> dict[str, float]:
        """Misst alle 15 Musical Goals für das übergebene Audio.

        Args:
            audio:     Audio-Signal (mono/stereo, float32)
            sr:        Sample-Rate in Hz
            reference: Optionales Original-Audio (vor Restaurierung) — verbessert
                       Präzision von tonal_center, timbre_authentizitaet, authentizitaet,
                       separation_fidelity und artikulation erheblich (§S6).

        Returns:
            Dict mit Goal-Namen → Score ∈ [0, 1].
            Leerer Dict bei Fehler (NaN-sicher).
        """
        # NaN/Inf-Schutz
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if audio.size == 0:
            return {}

        import concurrent.futures as _cf_mz

        try:
            checker = self._get_checker()
            # Bug-Fix §10c: thread-based timeout — prevents O(N²) or deadlock
            # in any metric from blocking the pipeline forever.  120 s covers
            # all 15 metrics even on long tracks; normal run < 30 s.
            with _cf_mz.ThreadPoolExecutor(max_workers=1) as _exec_mz:
                _fut_mz = _exec_mz.submit(checker.measure_all, audio, sr, reference)
                try:
                    raw = _fut_mz.result(timeout=120.0)
                except _cf_mz.TimeoutError:
                    logger.warning(
                        "ExzellenzDenker: messe_ziele() Timeout (120 s) — "
                        "leeres Dict zurückgegeben (Goal-Messung abgebrochen)"
                    )
                    return {}
            # Nur finite Werte behalten
            return {k: float(v) for k, v in raw.items() if np.isfinite(v)}
        except Exception as exc:
            logger.warning("ExzellenzDenker: Goal-Messung fehlgeschlagen: %s", exc)
            return {}

    def _get_surgical_zones(self, ctx: dict) -> list:
        """§2.59: Bereits chirurgisch behandelte Zonen — nicht erneut anfassen."""
        return ctx.get("surgical_defect_types", [])

    def messe_und_repariere(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        mode: str = "restoration",
        material: str = "auto",
        reference_audio: np.ndarray | None = None,
        inapplicable_goals: frozenset[str] | set[str] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Misst Goals und führt konservative Ziel-Reparatur für P3-P5-Verletzungen durch.

        Strategie (§0 + §2.45):
          1. Goals messen — sofort zurück wenn alle ≥ 0.75 (Minimal-Interventions-Prinzip).
          2. Nur P3-P5-Verletzungen mit Defizit ≥ 0.03 adressieren.
          3. Zeit-Domain-Reparatur (ExcellenceOptimizer, micro_dynamics + ola_edges):
             Kein STFT-Roundtrip-Fehler — sicher nach beliebig vielen Pipeline-STFT-Phasen
             (Ephraim & Malah 1984 gilt nur für spektrale Subtraktion, nicht für
             Hüllkurven-Modulation oder Overlap-Add-Crossfade).
          4. Frequenz-Blend-Reparatur (α ≥ 0.95, falls reference_audio vorhanden):
             Für waerme/brillanz/bass_kraft ohne neue STFT-Kaskade.
             Minimalste Rückmischung (max. 7 % Originalanteil).
          5. Bestes Kandidaten-Audio nur übernehmen wenn:
             goals_passed_after ≥ goals_passed_before UND
             kein Ziel um mehr als 0.02 schlechter als vor der Reparatur (§0).

        Warum sicher nach UV3-FeedbackChain (Abgrenzung zu v10.0.0):
          - v10.0.0 deaktivierte den ExcellenceOptimizer *global* nach UV3, weil alle
            vier Schritte auf STFT-Basis liefen und kaskadieren würden.
          - Hier werden *ausschließlich* Zeit-Domain-Schritte (micro_dynamics, ola_edges)
            verwendet — diese akkumulieren per Definition keine STFT-Rundungsfehler.
          - Der Blend-Schritt mischt Linear-PCM (kein Spektrum) → ebenfalls STFT-frei.

        Args:
            audio:           Verarbeitetes Audio (nach UV3 + FeedbackChain), float32.
            sr:              Sample-Rate (muss 48000 sein).
            material:        Trägermedium für ExcellenceOptimizer-Profile.
            reference_audio: Original-Audio vor der Restaurierung (für Blend-Reparatur).

        Returns:
            (audio_out, goals_dict): Im Erfolgsfall ggf. verbessertes Audio + Goals.
            Im Fehlerfall: (audio unverändert, ursprüngliche Goals oder {}).
        """
        # Mode-adaptive PMGG-Canonical-Schwellen (§9.10.77) — ersetzen den
        # vereinfachten 0.75-Floor. P1-Verletzungen (z.B. natuerlichkeit=0.82 < 0.90)
        # werden jetzt korrekt erkannt. P2: tonal_center < 0.95 wird als Verletzung
        # klassifiziert; Zeit-Domain-Ops helfen nur begrenzt, aber Erkennung ist Pflicht.
        _is_studio = str(mode).lower() in {"studio", "studio_2026", "studio2026"}
        _FALLBACK_MIN: float = 0.75  # Fallback für unbekannte Goals
        try:
            from backend.core.per_phase_musical_goals_gate import _get_canonical_thresholds as _gct

            _thresholds: dict[str, float] = _gct(is_studio_2026=_is_studio)
        except Exception:
            _thresholds = {}  # Import-Fehler → Fallback
        _MIN_DEFICIT: float = 0.03  # Reparatur nur wenn Score < threshold − 0.03
        _REPAIR_TRIGGER_FLOOR: float = 0.75  # §2.45: Borderline nahe 0.75 nicht nachbearbeiten

        # P3-P5 goals addressable by time-domain or blend repair.
        # P1/P2 goals werden in Step 4 via ultra-konservativem Blend (α=0.98/0.96) abgedeckt.
        _P3P5_REPAIR_TARGETS: frozenset[str] = frozenset(
            {
                # P3 — time-domain micro_dynamics helps directly
                "micro_dynamics",
                "groove",
                "emotionalitaet",
                # P4 — frequency blend helps (spectral envelope closer to reference)
                "waerme",
                "bass_kraft",
                "transparenz",
                "separation_fidelity",
                # P5 — frequency blend helps
                "brillanz",
                "spatial_depth",
            }
        )
        # Goals that benefit from micro_dynamics injection (time-domain envelope)
        _MICRO_DYN_GOALS: frozenset[str] = frozenset({"micro_dynamics", "groove", "emotionalitaet"})
        # Goals that benefit from OLA-crossfade edge smoothing (time-domain)
        _OLA_GOALS: frozenset[str] = frozenset({"artikulation", "natuerlichkeit"})
        # Goals that benefit from blend with reference (restores spectral warmth/brightness/spatiality)
        # spatial_depth: Blend zurück mit Original stellt übergetrocknete Raumcues wieder her
        # (phase_49 over-drying); Gate: _is_improvement() verhindert jede Regression (§0).
        # separation_fidelity: Blend mit Original stellt spektrales Gleichgewicht wieder her,
        # das Kreuz-Kontamination zwischen Komponenten durch Over-Processing reduziert.
        _BLEND_GOALS: frozenset[str] = frozenset(
            {"waerme", "bass_kraft", "brillanz", "transparenz", "spatial_depth", "separation_fidelity"}
        )
        # P1/P2 goals addressable by ultra-conservative blend only (≤ 4 % Originalanteil).
        # Letztes Sicherheitsnetz nach UV3+APR — keine STFT, keine Dynamik-Änderung.
        # Nur aktiv wenn: reference_audio vorhanden + Shape-Match + kein Hochrausch-Träger.
        # timbre (P2): strukturell identisch mit timbre_authentizitaet — konservativer Blend
        # stellt spektrale Formidentität wieder her, ohne harmonische Anteile hinzuzufügen.
        _P1P2_BLEND_GOALS: frozenset[str] = frozenset(
            {"authentizitaet", "timbre_authentizitaet", "timbre", "tonal_center", "transient_energie"}
        )
        # Modus-kritische Ziele für robuste Gesamtbalance über stark unterschiedliche Songs.
        # Restoration priorisiert Klangwahrheit + Defektfreiheit, Studio 2026 priorisiert
        # zusätzlich räumliche/moderne Präsenzziele.
        _MODE_CORE_GOALS: frozenset[str] = (
            frozenset(
                {
                    "natuerlichkeit",
                    "authentizitaet",
                    "timbre_authentizitaet",
                    "tonal_center",
                    "artikulation",
                    "transient_energie",
                    "spatial_depth",
                }
            )
            if not _is_studio
            else frozenset(
                {
                    "natuerlichkeit",
                    "authentizitaet",
                    "timbre_authentizitaet",
                    "tonal_center",
                    "artikulation",
                    "transient_energie",
                    "spatial_depth",
                    "brillanz",
                    "separation_fidelity",
                    "micro_dynamics",
                }
            )
        )

        # §2.32 Inapplicable-Filter: Goals die GAF als physikalisch nicht messbar
        # markiert hat (z.B. brillanz bei BW<8kHz ohne AudioSR, bass_kraft bei
        # vocal-dominanter Quelle) aus Reparatur UND Messung ausschließen.
        _inappl: frozenset[str] = frozenset(inapplicable_goals) if inapplicable_goals else frozenset()
        _eff_repair_targets: frozenset[str] = _P3P5_REPAIR_TARGETS - _inappl
        _eff_p1p2_blend_targets: frozenset[str] = _P1P2_BLEND_GOALS - _inappl

        # NaN/Inf-Schutz
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            return audio, {}

        # Step 1: Basis-Messung
        goals_initial = self.messe_ziele(audio, sr, reference=reference_audio)
        if not goals_initial:
            return audio, {}

        def _count_passed(goals: dict[str, float]) -> int:
            return sum(
                1
                for _k, _v in goals.items()
                if _k not in _inappl  # §S5: inapplicable Goals nicht mitzählen
                and math.isfinite(_v)
                and _v >= _thresholds.get(_k, _FALLBACK_MIN)
            )

        def _deficit_sum(goals: dict[str, float], *, focus: frozenset[str] | None = None) -> float:
            _acc = 0.0
            for _k, _v in goals.items():
                if focus is not None and _k not in focus:
                    continue
                if not math.isfinite(_v):
                    continue
                _thr = float(_thresholds.get(_k, _FALLBACK_MIN))
                _acc += max(0.0, _thr - float(_v))
            return float(_acc)

        _passed_initial = _count_passed(goals_initial)
        _total = len(goals_initial) - len(_inappl)  # §S5: inapplicable aus Gesamtzahl herausrechnen

        # §v10.303.7 Degradation-Convergence: Maximal 3 Reparatur-Versuche.
        # Auf degradiertem Material (Cassette, bw_loss=1.0) sind brillanz,
        # spatial_depth, waerme oft physikalisch unerreichbar. Der ExzellenzDenker
        # würde sonst 6×9s = 54s erfolglos loopieren.
        _repair_attempts: int = 0
        # §v10.306: Per-Goal Konvergenz-Tracking — Oszillations-Erkennung.
        # Wenn ein Goal in 3 aufeinanderfolgenden Versuchen keine Verbesserung
        # zeigt, wird es als "inapplicable" markiert und nicht weiter verfolgt.
        _goal_stall_count: dict[str, int] = {}
        _goal_best_score: dict[str, float] = {}
        _STALL_LIMIT: int = 3  # Max aufeinanderfolgende Stagnationen pro Goal
        _MAX_REPAIR_ATTEMPTS: int = 3

        # §v10.306 Konvergenz-Tracking: Goals die 3× stagnieren → inapplicable
        def _track_convergence(_cand_goals: dict[str, float]) -> None:
            for _gk, _gv in _cand_goals.items():
                if not math.isfinite(_gv):
                    continue
                _prev = _goal_best_score.get(_gk)
                if _prev is not None and _gv <= _prev + 0.005:
                    _goal_stall_count[_gk] = _goal_stall_count.get(_gk, 0) + 1
                else:
                    _goal_stall_count[_gk] = 0
                if _prev is None or _gv > _prev:
                    _goal_best_score[_gk] = _gv

        def _prune_stalled(_violations: set[str]) -> set[str]:
            _stalled = {k for k, c in _goal_stall_count.items() if c >= _STALL_LIMIT}
            if _stalled:
                logger.info("ExzellenzDenker: %d Goals inapplicable (3× stagniert): %s",
                            len(_stalled), ", ".join(sorted(_stalled)))
            return _violations - _stalled

        # P3-P5 violations with meaningful deficit (avoids micro-adjustments on borderline goals)
        _p35_violations: set[str] = {
            k
            for k, v in goals_initial.items()
            if k in _eff_repair_targets
            and math.isfinite(v)
            and v < min(_thresholds.get(k, _FALLBACK_MIN), _REPAIR_TRIGGER_FLOOR) - _MIN_DEFICIT
        }
        # P1/P2 violations für ultra-konservativen Blend-Fallback (Step 4)
        _p12_violations: set[str] = {
            k
            for k, v in goals_initial.items()
            if k in _eff_p1p2_blend_targets
            and math.isfinite(v)
            and v < min(_thresholds.get(k, _FALLBACK_MIN), _REPAIR_TRIGGER_FLOOR) - _MIN_DEFICIT
        }
        if not _p35_violations and not _p12_violations:
            # §2.45: Minimal-Intervention — kein Eingriff wenn Goals erfüllt / Grenzfall
            return audio, goals_initial

        logger.info(
            "ExzellenzDenker messe_und_repariere: %d/%d Goals | %d P3-P5-Verletzungen: %s%s",
            _passed_initial,
            _total,
            len(_p35_violations),
            ", ".join(sorted(_p35_violations)),
            (
                f" | {len(_p12_violations)} P1/P2-Verletzungen: {', '.join(sorted(_p12_violations))}"
                if _p12_violations
                else ""
            ),
        )

        _best_audio: np.ndarray = audio
        _best_goals: dict[str, float] = dict(goals_initial)
        _best_passed: int = _passed_initial

        def _is_improvement(candidate_goals: dict[str, float]) -> bool:
            """Accept if net improvement OR no goal regresses beyond tolerance (§v10.306 recalibrated).

            §v10.306: Vorher _max_drop=0.01/0.015/0.02 war innerhalb Messrauschen
            (spektrale Metriken haben ~0.02 Varianz). Jede echte Verbesserung wurde
            abgelehnt. Neue Schwellen: 0.03/0.04/0.05 + Net-Improvement-Gate.
            """
            _cand_passed = _count_passed(candidate_goals)

            # §v10.306 Net-Improvement-Gate: Erlaube Kandidat wenn Summe der
            # Verbesserungen > Summe der Verschlechterungen (über ALLE Goals).
            _net_gain = 0.0
            _regression_count = 0
            for _g, _v_init in goals_initial.items():
                if not math.isfinite(_v_init):
                    continue
                _v_cand = candidate_goals.get(_g, _v_init)
                if not math.isfinite(_v_cand):
                    continue
                _delta = _v_cand - _v_init
                _net_gain += _delta
                # §v10.306: Toleranz-Schwelle 5× höher als vorher — oberhalb Messrauschen
                _max_drop = 0.05  # general: 5% Toleranz (vorher 2%)
                if _g in _MODE_CORE_GOALS:
                    _max_drop = 0.04  # core: 4% (vorher 1.5%)
                if _g in _P1P2_BLEND_GOALS:
                    _max_drop = 0.03  # blend: 3% (vorher 1%)
                if _delta < -_max_drop:
                    _regression_count += 1

            # Net-Improvement: positive Gesamtbilanz → akzeptieren
            if _net_gain > 0.0:
                return True

            # Keine Net-Verbesserung aber auch keine schwere Regression
            if _regression_count == 0 and _cand_passed >= _best_passed:
                return True

            # Zu viele Regressionen → ablehnen
            if _regression_count > 0 and _net_gain <= -0.02:
                return False

            # Alte Guards als Safety-Net
            if _cand_passed < _best_passed - 1:  # Toleranz: 1 Goal weniger passed ist ok
                return False

        # Step 2: Zeit-Domain-Reparatur (micro_dynamics + ola_edges — kein STFT)
        _needs_td = bool(_p35_violations & (_MICRO_DYN_GOALS | _OLA_GOALS))
        if _needs_td and _repair_attempts < _MAX_REPAIR_ATTEMPTS:
            _repair_attempts += 1
            try:
                from backend.core.excellence_optimizer import ExcellenceOptimizer

                _apply_md = bool(_p35_violations & _MICRO_DYN_GOALS)
                _apply_ola = bool(_p35_violations & _OLA_GOALS)
                _opt_td = ExcellenceOptimizer(
                    sample_rate=sr,
                    apply_continuity=False,  # Kein STFT — Ephraim & Malah 1984
                    apply_micro_dynamics=_apply_md,
                    apply_harmonic_boost=False,  # Kein STFT — Ephraim & Malah 1984
                    apply_ola_edges=_apply_ola,
                    material=material,
                )
                # Konservative Stärke: §0 Minimal-Intervention
                _opt_td._modulation_strength = (  # pylint: disable=protected-access
                    getattr(_opt_td, "_modulation_strength", 0.3) * 0.55
                )
                _td_out, _ = _opt_td.optimize(_best_audio)
                _td_out = np.clip(
                    np.nan_to_num(_td_out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                    -1.0,
                    1.0,
                )
                _td_goals = self.messe_ziele(_td_out, sr, reference=reference_audio)
                if _td_goals and _is_improvement(_td_goals):
                    _td_passed = _count_passed(_td_goals)
                    _best_audio = _td_out
                    _best_goals = _td_goals
                    _best_passed = _td_passed
                    logger.info(
                        "ExzellenzDenker Goal-Repair (Zeit-Domain): %d → %d Goals bestanden",
                        _passed_initial,
                        _best_passed,
                    )
                else:
                    logger.debug(
                        "ExzellenzDenker Goal-Repair (Zeit-Domain): keine Verbesserung (%d Goals)",
                        _passed_initial,
                    )
            except Exception as _td_exc:
                logger.debug("ExzellenzDenker Goal-Repair (Zeit-Domain) fehlgeschlagen: %s", _td_exc)

        # Step 3: Frequenz-Blend-Reparatur für waerme/brillanz/bass_kraft/transparenz
        _needs_blend = bool(_p35_violations & _BLEND_GOALS)
        # §0: Rauschbehaftetes Original NICHT remischen — Blend hebt Träger-Rauschen zurück.
        # shellac/wax_cylinder/wire_recording haben SNR < 20 dB → max. 4 % remixte Rauschenergie
        # wäre hörbarer Qualitätsverlust (Primum non nocere).
        _HIGH_NOISE_MATERIALS: frozenset[str] = frozenset({"shellac", "wax_cylinder", "wire_recording"})
        if str(material).lower() in _HIGH_NOISE_MATERIALS:
            _needs_blend = False
            logger.debug("ExzellenzDenker Blend-Reparatur übersprungen — Hochrausch-Träger: %s", material)
        if _needs_blend and reference_audio is not None and reference_audio.shape == _best_audio.shape:
            _repair_attempts += 1
            try:
                _ref_f32 = np.nan_to_num(reference_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                for _alpha in (0.96, 0.93):
                    _blended = np.clip(
                        _alpha * _best_audio + (1.0 - _alpha) * _ref_f32,
                        -1.0,
                        1.0,
                    )
                    _blend_goals = self.messe_ziele(_blended, sr, reference=reference_audio)
                    if not _blend_goals:
                        continue
                    _blend_passed = _count_passed(_blend_goals)
                    if _is_improvement(_blend_goals) and _blend_passed >= _best_passed:
                        _best_audio = _blended
                        _best_goals = _blend_goals
                        _best_passed = _blend_passed
                        logger.info(
                            "ExzellenzDenker Goal-Repair (Blend α=%.2f): %d → %d Goals bestanden",
                            _alpha,
                            _passed_initial,
                            _best_passed,
                        )
                        break  # Erste Verbesserung genügt (§2.45 Minimal-Intervention)
            except Exception as _bl_exc:
                logger.debug("ExzellenzDenker Goal-Repair (Blend) fehlgeschlagen: %s", _bl_exc)

        # Step 4: Ultra-konservativer P1/P2-Blend (max 4 % Originalanteil)
        # Letztes Sicherheitsnetz für authentizitaet/timbre_authentizitaet/tonal_center/
        # transient_energie wenn UV3+APR nicht ausreichten.
        # §2.45: alpha ∈ {0.98, 0.96} → max. 4 % Original. Kein STFT, kein Dynamik-Eingriff.
        # §0: Primum non nocere — _is_improvement() blockiert jede Regression.
        # §0h: Hochrausch-Träger (shellac/wax/wire) ausgeschlossen — Rausch-Reinmix wäre hörbar.
        _needs_p12_blend = (
            bool(_p12_violations)
            and reference_audio is not None
            and reference_audio.shape == _best_audio.shape
            and str(material).lower() not in _HIGH_NOISE_MATERIALS
        )
        if _needs_p12_blend and _repair_attempts < _MAX_REPAIR_ATTEMPTS:
            _repair_attempts += 1
            try:
                _ref_f32_p12 = np.nan_to_num(reference_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                for _alpha_p12 in (0.98, 0.96):
                    _p12_blended = np.clip(
                        _alpha_p12 * _best_audio + (1.0 - _alpha_p12) * _ref_f32_p12,
                        -1.0,
                        1.0,
                    )
                    _p12_goals = self.messe_ziele(_p12_blended, sr, reference=reference_audio)
                    if not _p12_goals:
                        continue
                    _p12_passed = _count_passed(_p12_goals)
                    if _is_improvement(_p12_goals) and _p12_passed >= _best_passed:
                        _best_audio = _p12_blended
                        _best_goals = _p12_goals
                        _best_passed = _p12_passed
                        logger.info(
                            "ExzellenzDenker P1/P2-Blend (α=%.2f): %d → %d Goals bestanden (%s)",
                            _alpha_p12,
                            _passed_initial,
                            _best_passed,
                            ", ".join(sorted(_p12_violations)),
                        )
                        break
            except Exception as _p12_exc:
                logger.debug("ExzellenzDenker P1/P2-Blend fehlgeschlagen: %s", _p12_exc)

        # Step 5: Lokaler End-Gate-Backoff für spatial_depth/waerme
        # Ziel: Restliche Defizite in Raumtiefe/Wärme glätten, ohne globale
        # Kernziele zu verschlechtern. Akzeptanz nur wenn _is_improvement()
        # AND lokaler Defizitvektor tatsächlich sinkt.
        _LOCAL_RESCUE_GOALS: frozenset[str] = frozenset({"spatial_depth", "waerme"})
        _needs_local_rescue = (
            reference_audio is not None
            and reference_audio.shape == _best_audio.shape
            and str(material).lower() not in _HIGH_NOISE_MATERIALS
        )
        if _needs_local_rescue and _repair_attempts < _MAX_REPAIR_ATTEMPTS:
            _repair_attempts += 1
            try:
                _rescue_targets = {
                    _g
                    for _g in _LOCAL_RESCUE_GOALS
                    if math.isfinite(float(_best_goals.get(_g, 1.0)))
                    and float(_best_goals.get(_g, 1.0)) < float(_thresholds.get(_g, _FALLBACK_MIN))
                }
                if _rescue_targets:
                    _ref_f32_local = np.nan_to_num(reference_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    _best_local_def = _deficit_sum(_best_goals, focus=frozenset(_rescue_targets))
                    _waerme_focus = "waerme" in _rescue_targets
                    _spa_now = float(_best_goals.get("spatial_depth", 1.0))
                    _wae_now = float(_best_goals.get("waerme", 1.0))
                    for _alpha_local in (0.985, 0.975, 0.965):
                        _local_blended = np.clip(
                            _alpha_local * _best_audio + (1.0 - _alpha_local) * _ref_f32_local,
                            -1.0,
                            1.0,
                        )
                        _local_goals = self.messe_ziele(_local_blended, sr, reference=reference_audio)
                        if not _local_goals:
                            continue
                        _spa_cand = float(_local_goals.get("spatial_depth", _spa_now))
                        _wae_cand = float(_local_goals.get("waerme", _wae_now))
                        _cand_local_def = _deficit_sum(_local_goals, focus=frozenset(_rescue_targets))
                        _cand_passed = _count_passed(_local_goals)
                        # Bei waerme-Rettung erlauben wir kleinen Spatial-Tradeoff,
                        # aber nur wenn waerme spuerbar steigt und spatial_depth nicht kippt.
                        if _waerme_focus:
                            _spa_drop_local = _spa_now - _spa_cand
                            _wae_gain_local = _wae_cand - _wae_now
                            if _spa_drop_local > 0.02:
                                continue
                            if _wae_gain_local < 0.015:
                                continue
                        if _cand_local_def + 1e-6 < _best_local_def and _is_improvement(_local_goals):
                            _best_audio = _local_blended
                            _best_goals = _local_goals
                            _best_passed = _cand_passed
                            logger.info(
                                "ExzellenzDenker Local-Rescue (α=%.3f): local_def %.4f→%.4f (%s)",
                                _alpha_local,
                                _best_local_def,
                                _cand_local_def,
                                ", ".join(sorted(_rescue_targets)),
                            )
                            break
            except Exception as _local_exc:
                logger.debug("ExzellenzDenker Local-Rescue fehlgeschlagen: %s", _local_exc)

        return _best_audio, _best_goals

    def prognostiziere(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        defect_result: object | None = None,
        material: str = "auto",
    ) -> dict[str, float]:
        """Schätzt Goal-Risikowahrscheinlichkeiten VOR der Restaurierung.

        Schnelle DSP-Proxies (≤ 200 ms) als Frühwarnsystem für den
        PhaseInteractionDenker. Ermöglicht prophylaktische Phasen-Aktivierung
        bevor UV3 läuft — statt P1/P2-Verletzungen erst danach zu heilen (§2.45a).

        Kein ML, kein STFT-Roundtrip — reine Spektralstatistik auf Kurzfenster.

        Gemessene Risiken:
            Rauschboden       → Natuerlichkeit / Authentizitaet
            HF-Verlust        → Brillanz / Timbre
            Transient-Armut   → Groove / MicroDynamics
            Dropout-Schwere   → Artikulation

        Returns:
            Dict goal_name → Risikowahrscheinlichkeit ∈ [0, 1].
            0.0 = sicher, 1.0 = sicher verletzt ohne schützende Phase.
            Leeres Dict bei internem Fehler (fail-safe).
        """
        try:
            audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if audio.size == 0:
                return {}

            mono = audio if audio.ndim == 1 else audio.mean(axis=0)
            risk: dict[str, float] = {}

            # ── Rauschboden → Natuerlichkeit / Authentizitaet ─────────────────────
            # 5. Perzentil der FFT-Magnituden als Rauschboden-Proxy.
            # Referenz: -70 dBFS = sauber; -40 dBFS = hohes Rauschen.
            try:
                n_fft = min(4096, mono.size)
                if n_fft >= 128:
                    fft_mag = np.abs(np.fft.rfft(mono[:n_fft]))
                    noise_floor_db = float(np.percentile(20.0 * np.log10(fft_mag + 1e-10), 5))
                    # Lineares Risiko zwischen -70 dBFS (0.0) und -40 dBFS (1.0)
                    noise_risk = float(np.clip((-noise_floor_db - 40.0) / 30.0, 0.0, 1.0))
                    risk["natuerlichkeit"] = noise_risk
                    risk["authentizitaet"] = round(noise_risk * 0.80, 3)
            except Exception:
                logger.debug("prognostiziere: silent except suppressed", exc_info=True)

            # ── HF-Verlust → Brillanz / Timbre ───────────────────────────
            # Welch-PSD: Energie 8 kHz+ / Breitband-Energie.
            # Referenz: CD (1990er) ≅ 0.10-0.25; Shellac ≅ 0.01-0.03.
            try:
                if sr > 0 and mono.size >= sr // 8:
                    from scipy.signal import welch as _welch

                    _nperseg = min(1024, mono.size)
                    freqs, psd = _welch(mono, fs=sr, nperseg=_nperseg)
                    bb_energy = float(np.sum(psd[freqs >= 20])) + 1e-10
                    hf_energy = float(np.sum(psd[freqs >= 8000]))
                    hf_ratio = hf_energy / bb_energy
                    # Risiko steigt wenn hf_ratio < 0.08 (≥ Kassetten/Shellac-Niveau)
                    hf_risk = float(np.clip(1.0 - hf_ratio / 0.08, 0.0, 1.0))
                    risk["brillanz"] = hf_risk
                    risk["timbre"] = round(hf_risk * 0.70, 3)
            except Exception:
                logger.debug("prognostiziere: silent except suppressed", exc_info=True)

            # ── Transient-Armut → Groove / MicroDynamics ─────────────────
            # Onset-Rate in 10-ms-Fenstern. Referenz: lebendige Musik ≥ 1.5 Onsets/s.
            # Tiefes Rauschen oder starker Dropout senken die Rate.
            try:
                if mono.size >= 4800:
                    hop = int(sr * 0.010)  # 10 ms
                    max_frames = min(len(mono) // hop, 1000)  # max 10 s analysieren
                    env = np.abs(mono[: max_frames * hop])
                    frames = env.reshape(max_frames, hop)
                    frame_rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-8
                    onsets = float(np.sum(frame_rms[1:] > 3.0 * frame_rms[:-1]))
                    duration_s = max_frames * hop / max(sr, 1)
                    onset_rate = onsets / max(duration_s, 0.001)
                    # Risiko steigt wenn Onset-Rate < 0.8/s (stark gedämpftes Signal)
                    transient_risk = float(np.clip(1.0 - onset_rate / 0.8, 0.0, 1.0))
                    risk["groove"] = round(transient_risk * 0.70, 3)
                    risk["micro_dynamics"] = round(transient_risk * 0.80, 3)
            except Exception:
                logger.debug("prognostiziere: silent except suppressed", exc_info=True)

            # ── Dropout-Schwere → Artikulation ──────────────────────────
            # Dropout-Severity direkt aus DefectResult (falls verfügbar).
            try:
                if defect_result is not None:
                    scores = getattr(defect_result, "scores", {}) or {}
                    dropout_sev = 0.0
                    for k, v in scores.items():
                        if "dropout" in str(k).lower():
                            sev = float(getattr(v, "severity", 0.0) if not isinstance(v, (int, float)) else v)
                            dropout_sev = max(dropout_sev, sev)
                    if dropout_sev > 0.0:
                        risk["artikulation"] = float(np.clip(dropout_sev, 0.0, 1.0))
            except Exception:
                logger.debug("prognostiziere: silent except suppressed", exc_info=True)

            logger.debug(
                "ExzellenzDenker.prognostiziere(): material=%s %d Risikofelder: %s",
                material,
                len(risk),
                {k: f"{v:.2f}" for k, v in risk.items()},
            )
            return risk
        except Exception as exc:
            logger.debug("ExzellenzDenker.prognostiziere() fehlgeschlagen: %s", exc)
            return {}

    def bewerte_zielrisiko(
        self,
        current_scores: dict[str, float],
    ) -> dict[str, GoalRiskAssessment]:
        """Einheitliche Risikobewertung aller Goals aus gemessenen/prognostizierten Scores.

        Verwendet von:
          - prognostiziere() für die prophylaktische Risikovorhersage (Stufe 5b)
          - optimiere() für die reaktive Reparaturentscheidung (Stufe 9)

        Beide Pfade nutzen DIESELBEN _GOAL_THRESHOLDS — keine abweichenden
        Schwellen mehr zwischen Vorhersage und Messung.

        Args:
            current_scores: dict goal_name → Score ∈ [0, 1].
                            Kann aus prognostiziere() (DSP-Proxy) oder
                            messe_ziele() (full ML) stammen.

        Returns:
            dict goal_name → GoalRiskAssessment mit risk, needs_protection,
            needs_repair und passed.
        """
        assessments: dict[str, GoalRiskAssessment] = {}
        for goal_name, score in current_scores.items():
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                continue
            score = float(np.clip(score, 0.0, 1.0))
            threshold = _GOAL_THRESHOLDS.get(goal_name, 0.65)

            # Risiko mit logistischer Sigmoid-Charakteristik:
            # Bei score = threshold → risk ≈ 0.50
            # Bei score = 0 → risk ≈ 0.95
            # Bei score = 1 → risk ≈ 0.01
            deficit = threshold - score
            if deficit <= 0:
                risk = 0.05 * abs(deficit)  # minimal risk bei Überschreitung
            else:
                risk = float(np.clip(deficit / threshold, 0.0, 0.95))

            risk = float(np.clip(risk, 0.0, 1.0))
            repair_threshold = threshold * 0.85

            assessments[goal_name] = GoalRiskAssessment(
                goal_name=goal_name,
                current_score=score,
                threshold=threshold,
                risk=risk,
                needs_protection=risk >= 0.50,
                needs_repair=score < repair_threshold,
            )
        return assessments

    # ── Interne Hilfsmethoden ────────────────────────────────────────────────

    def _get_optimizer(self, sr: int = 48_000, material: str = "auto") -> Any:
        """Double-Checked Locking — lazy ExcellenceOptimizer-Init.

        Der Optimizer wird je sr+material neu erstellt wenn sr/material sich ändern.
        Für die häufigsten Fälle (konstantes sr=48000, material="auto") wird
        die bestehende Instanz wiederverwendet.
        """
        if self._optimizer is not None:
            # Quick-check ob sample_rate und material noch stimmen
            existing_sr = getattr(self._optimizer, "sample_rate", None)
            existing_mat = getattr(self._optimizer, "material", None)
            if existing_sr == sr and existing_mat == material.lower().strip():
                return self._optimizer

        with self._opt_lock:
            # Erneuter Check unter Lock
            if self._optimizer is not None:
                existing_sr = getattr(self._optimizer, "sample_rate", None)
                existing_mat = getattr(self._optimizer, "material", None)
                if existing_sr == sr and existing_mat == material.lower().strip():
                    return self._optimizer
            self._optimizer = self._build_optimizer(sr, material)
        return self._optimizer

    @staticmethod
    def _build_optimizer(sr: int, material: str) -> Any:
        """Erstellt einen neuen ExcellenceOptimizer."""
        from backend.core.excellence_optimizer import ExcellenceOptimizer  # lazy import

        return ExcellenceOptimizer(
            sample_rate=sr,
            apply_continuity=True,
            apply_micro_dynamics=True,
            apply_harmonic_boost=True,
            apply_ola_edges=True,
            material=material,
            use_mert=material.lower().strip()
            in {  # M-7b: MERT nur für Vokal-Material
                "gesang",
                "sprache",
                "vocal",
                "singer",
                "speech",
                "voice",
            },
        )

    def _get_checker(self) -> Any:
        """Double-Checked Locking — lazy MusicalGoalsChecker-Init."""
        if self._checker is None:
            with self._checker_lock:
                if self._checker is None:
                    self._checker = self._build_checker()
        return self._checker

    @staticmethod
    def _build_checker() -> Any:
        """Erstellt MusicalGoalsChecker."""
        from backend.core.musical_goals.musical_goals_metrics import (  # lazy import
            MusicalGoalsChecker,
        )

        return MusicalGoalsChecker()

    @staticmethod
    def _fallback(audio: np.ndarray, grund: str) -> ExzellenzErgebnis:
        """Fallback-Ergebnis bei leerem oder fehlerhaftem Audio."""
        return ExzellenzErgebnis(
            audio=audio,
            excellence_score=0.0,
            musical_goals={},
            goals_passed=0,
            goals_total=0,
            improvements=[],
            processing_note=f"Keine Optimierung möglich: {grund}",
            warnings=[grund],
        )


# ─── Modul-Level-Singleton (Double-Checked Locking — Spec §3.2) ─────────────

_instance: ExzellenzDenker | None = None
_lock = threading.Lock()


def get_exzellenz_denker() -> ExzellenzDenker:
    """Thread-sicherer Singleton-Accessor für den ExzellenzDenker."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ExzellenzDenker()
    return _instance


def optimiere(
    audio: np.ndarray,
    sr: int = 48_000,
    *,
    material: str = "auto",
) -> ExzellenzErgebnis:
    """Convenience-Wrapper: Exzellenz-Optimierung über Singleton."""
    return get_exzellenz_denker().optimiere(audio, sr, material=material)


def messe_ziele(audio: np.ndarray, sr: int = 48_000) -> dict[str, float]:
    """Convenience-Wrapper: Musical Goals messen über Singleton."""
    return get_exzellenz_denker().messe_ziele(audio, sr)
