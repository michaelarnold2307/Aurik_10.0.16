"""
PhaseConductor — Inter-Phase Adaptive Controller (Aurik 10.0.0.x, §Hebel-3)
=========================================================================

Intelligente Wetness-Steuerung zwischen Phasen:
- Misst nach jeder Phase den Residual-Defekt-Zustand (Spektralvarianz, Rauschboden, Transienten)
- Schätzt ob die nächste Phase noch Gewinn bringt (vorhersage_nutzen)
- Passt `strength`/`wet` der nächsten Phase dynamisch an (kein Over-Processing)

Architekturprinzip: Kein ML-Modell. Leichtgewichtiger Encoder auf Basis DSP-Merkmale:
  - Noise-Floor (5. Perzentil PSD)
  - HF-Energy-Ratio (8 kHz–Nyquist / Breitband)
  - Transient-Density (Onset-Rate per Sekunde)
  - Harmonic-Coherence (Autocorrelation-Peak-Ratio)

Diese 4 Merkmale bilden einen 4D-State-Vektor. Die Empfehlung erfolgt durch
Nearest-Neighbor in einem Per-Material-Referenzgitter (Mahal.-Distanz).

Alle Operationen: CPU-only, < 50 ms für 1 min Audio.

Singleton via `get_phase_conductor()`.

Author: Aurik Development Team
Version: 1.0.0 (Hebel-3 §Adaptive-Intelligence v10.0.0)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_INSTANCE_HOLDER: list[PhaseConductor | None] = [None]
_lock = threading.Lock()


@dataclass
class PhaseState:
    """Aktueller Defektzustand nach einer Phase — Eingang für Conductor-Entscheidung."""

    noise_floor_db: float  # 5. Perzentil PSD [dBFS], ≤ −60 = sauber
    hf_energy_ratio: float  # Energie 8 kHz – Nyquist / Breitband [0–1]
    transient_density: float  # Onset-Rate [Events/s], 0 = keine
    harmonic_coherence: float  # Autokorr.-Peak-Ratio F0 [0–1], 1 = rein tonisch
    rms_db: float  # RMS-Pegel [dBFS]
    phase_id: str = ""  # Letzter abgeschlossener Phase-Identifier

    def as_vec(self) -> np.ndarray:
        """4D-Merkmals-Vektor für Nearest-Neighbor-Lookup."""
        return np.array(  # type: ignore[no-any-return]
            [
                float(np.clip((self.noise_floor_db + 80.0) / 80.0, 0.0, 1.0)),  # 0=sauber, 1=laut
                float(np.clip(self.hf_energy_ratio, 0.0, 1.0)),
                float(np.clip(self.transient_density / 20.0, 0.0, 1.0)),  # norm 20 Events/s
                float(np.clip(self.harmonic_coherence, 0.0, 1.0)),
            ],
            dtype=np.float64,
        )


@dataclass
class ConductorRecommendation:
    """Empfehlung für die nächste Phase."""

    next_phase_id: str
    recommended_strength: float  # 0.0–1.0: empfohlener Wet-Level
    skip_recommended: bool  # True wenn Conductor keinen Nutzen erwartet
    skip_reason: str = ""
    confidence: float = 0.5  # Konfidenz [0–1]
    state_snapshot: PhaseState | None = None


# ─── Referenz-Gitter: per-Materialklasse ────────────────────────────────
# Jede Row: [noise_norm, hf_ratio, transient_norm, harm_coh, ideal_strength]
# Calibrated on internal Golden-Samples (2026-04).
_REFERENCE_GRID: dict[str, np.ndarray] = {
    "vinyl": np.array(
        [
            # noise  hf    trans  harm  strength
            [0.20, 0.30, 0.70, 0.85, 0.90],  # high crackle, good harmonics → treat heavily
            [0.10, 0.50, 0.30, 0.90, 0.50],  # moderate noise, lots of HF → moderate strength
            [0.05, 0.70, 0.10, 0.95, 0.20],  # nearly clean → gentle pass
            [0.40, 0.10, 0.90, 0.60, 0.75],  # heavy crackle, transient-rich → moderate
        ]
    ),
    "tape": np.array(
        [
            [0.30, 0.20, 0.50, 0.80, 0.85],
            [0.15, 0.40, 0.30, 0.85, 0.55],
            [0.05, 0.60, 0.10, 0.92, 0.20],
            [0.50, 0.10, 0.40, 0.70, 0.80],
        ]
    ),
    "shellac": np.array(
        [
            [0.55, 0.10, 0.80, 0.70, 0.90],
            [0.35, 0.20, 0.60, 0.75, 0.75],
            [0.15, 0.30, 0.40, 0.80, 0.45],
            [0.05, 0.40, 0.20, 0.88, 0.20],
        ]
    ),
    "cd_digital": np.array(
        [
            [0.05, 0.75, 0.20, 0.92, 0.15],  # clean digital → very gentle
            [0.10, 0.60, 0.40, 0.85, 0.30],
        ]
    ),
    "unknown": np.array(
        [
            [0.20, 0.40, 0.50, 0.80, 0.60],
            [0.10, 0.50, 0.30, 0.85, 0.40],
            [0.05, 0.65, 0.15, 0.90, 0.20],
            [0.35, 0.25, 0.70, 0.72, 0.75],
        ]
    ),
}

# Phasen, die nie übersprungen werden dürfen (§6.2a Material-Pflicht-Phasen)
_NEVER_SKIP = frozenset(
    {
        "phase_01_click_removal",
        "phase_09_crackle_removal",
        "phase_12_wow_flutter_fix",
        "phase_14_phase_correction",
        "phase_15_stereo_balance",
    }
)

# Mindeststärken je Phase-Typ (verhindert Bypass durch Over-Confidence)
_MIN_STRENGTH: dict[str, float] = {
    "phase_03_denoise": 0.35,
    "phase_29_tape_hiss_reduction": 0.12,
    "phase_01_click_removal": 0.30,
    "phase_07_harmonic_restoration": 0.10,
    "phase_06_frequency_restoration": 0.10,
}
_DEFAULT_MIN_STRENGTH = 0.25  # §v10.44: angehoben von 0.05 auf Joint-Calibration-Minimum (§G73)


class PhaseConductor:
    """Inter-Phase Adaptive Controller.

    Misst Restdefekt-Zustand nach jeder Phase und gives recommendations
    für Stärke / Skip der nächsten Phase.

    Thread-sicher (alle öffentlichen Methoden unter _lock).
    """

    def __init__(self) -> None:
        self._op_lock = threading.Lock()
        self._history: list[tuple[str, PhaseState]] = []  # (phase_id, state)
        logger.info("PhaseConductor initialisiert (§Hebel-3 Adaptive-Conductor v1.0)")

    # ── Public API ──────────────────────────────────────────────────────

    def measure_state(self, audio: np.ndarray, sr: int, phase_id: str) -> PhaseState:
        """Messe Defektzustand des Audiosignals nach einer Phase.

        Parameters
        ----------
        audio : np.ndarray
            Mono oder Stereo (N,) oder (N, 2) — Float32/64 in [-1, 1]
        sr : int
            Sample-Rate (48000)
        phase_id : str
            Bezeichner der gerade abgeschlossenen Phase (für Protokoll)

        Returns
        -------
        PhaseState
            4D-Merkmals-Snapshot
        """
        mono = _to_mono(audio)
        noise_floor_db = _estimate_noise_floor(mono, sr)
        hf_energy_ratio = _estimate_hf_energy_ratio(mono, sr)
        transient_density = _estimate_transient_density(mono, sr)
        harmonic_coherence = _estimate_harmonic_coherence(mono, sr)
        rms_db = _rms_db(mono)

        state = PhaseState(
            noise_floor_db=noise_floor_db,
            hf_energy_ratio=hf_energy_ratio,
            transient_density=transient_density,
            harmonic_coherence=harmonic_coherence,
            rms_db=rms_db,
            phase_id=phase_id,
        )
        with self._op_lock:
            self._history.append((phase_id, state))
        logger.debug(
            "PhaseConductor: %s → noise=%.1f dBFS hf=%.2f trans=%.1f/s harm=%.2f",
            phase_id,
            noise_floor_db,
            hf_energy_ratio,
            transient_density,
            harmonic_coherence,
        )
        return state

    def recommend(
        self,
        next_phase_id: str,
        current_state: PhaseState,
        material_type: str = "unknown",
        current_strength: float = 1.0,
        goal_weights: dict[str, float] | None = None,
        song_goal_targets: dict[str, float] | None = None,
        current_goal_scores: dict[str, float] | None = None,
        current_phase_id: str | None = None,
        active_phase_coalitions: dict[str, tuple[str, ...]] | None = None,
    ) -> ConductorRecommendation:
        """Empfehle Stärke / Skip-Entscheidung für die nächste Phase.

        Parameters
        ----------
        next_phase_id : str
            Phase-Identifier der nächsten Phase
        current_state : PhaseState
            Zustand NACH der letzten Phase
        material_type : str
            Materialklasse (z.B. "vinyl", "tape", …)
        current_strength : float
            Aktuell geplante Stärke der nächsten Phase
        goal_weights : dict[str, float] | None
            §2.52a Per-Song Goal-Gewichte (aus SongGoalImportance §2.56).
            Moduliert Strength ±10 % basierend auf gewichteter Relevanz.
        song_goal_targets : dict[str, float] | None
            §2.31 Per-Song Studio-Day-Targets aus estimate_song_goal_targets().
            Stopp-Signal: wenn aktuelle Scores ≥ Targets − 0.03 → Strength 0.0.
        current_goal_scores : dict[str, float] | None
            Aktuelle Musical-Goal-Scores (nach letzter Phase).
        current_phase_id : str | None
            Zuletzt abgeschlossene Phase; ermöglicht Koalitions-Kontinuität.
        active_phase_coalitions : dict[str, tuple[str, ...]] | None
            Aktive Phase-Koalitionen aus dem Restorer; verhindert vorzeitige
            Skip- oder Dämpfungs-Empfehlungen innerhalb zusammengehöriger Gruppen.

        Returns
        -------
        ConductorRecommendation
        """
        # §6.2a: Pflicht-Phasen nie überspringen
        if next_phase_id in _NEVER_SKIP:
            return ConductorRecommendation(
                next_phase_id=next_phase_id,
                recommended_strength=current_strength,
                skip_recommended=False,
                skip_reason="",
                confidence=1.0,
                state_snapshot=current_state,
            )

        current_coalition = _coalition_name_for_phase(current_phase_id, active_phase_coalitions)
        next_coalition = _coalition_name_for_phase(next_phase_id, active_phase_coalitions)
        coalition_continuation = bool(current_coalition and current_coalition == next_coalition)

        grid_key = _canonical_material(material_type)
        grid = _REFERENCE_GRID.get(grid_key, _REFERENCE_GRID["unknown"])
        state_vec = current_state.as_vec()

        # Psychoacoustic weighting: noise-floor and harmonic coherence dominate
        # perceived restoration headroom more strongly than HF ratio alone.
        # Scientific basis: Zwicker & Fastl (2007) masking/noise salience;
        # Virtanen et al. (2007) harmonicity as a restoration quality prior.
        _distance_weights = np.array([1.35, 0.85, 1.00, 1.25], dtype=np.float64)
        dists = np.linalg.norm((grid[:, :4] - state_vec) * _distance_weights, axis=1)
        nn_idx = int(np.argmin(dists))
        nn_dist = float(dists[nn_idx])
        ideal_strength = float(grid[nn_idx, 4])

        # Interpolation: wenn Zustand sauber (Distanz zu "clean" < Rand)
        # → lineare Absorption des Ideals
        confidence = float(np.clip(1.0 - nn_dist / 1.5, 0.2, 0.9))
        recommended_strength = float(
            np.clip(ideal_strength * confidence + current_strength * (1.0 - confidence), 0.0, 1.0)
        )

        # §2.52a Goal-Weight-Modulation: ±10 % bounded adjustment.
        # High-priority goals (weight > 1.0) push strength up; low-priority push down.
        if goal_weights:
            try:
                _gw_vals = [v for v in goal_weights.values() if isinstance(v, (int, float)) and np.isfinite(v)]
                if _gw_vals:
                    _gw_mean = float(np.mean(_gw_vals))
                    # map mean weight to [-0.10, +0.10] range (1.0 = neutral)
                    _gw_mod = float(np.clip((_gw_mean - 1.0) * 0.10, -0.10, 0.10))
                    recommended_strength = float(np.clip(recommended_strength + _gw_mod, 0.0, 1.0))
            except Exception as e:
                logger.warning("Verarbeitungsschritt_conductor.py::unbekannter Ersatzpfad: %s", e)
                pass  # Non-blocking: goal_weights integration failure → neutral

        # §2.31 Per-Song Studio-Day-Target Stopp-Signal: Phasen über Ziel hinaus verhindern
        # (Over-Processing-Schutz ohne PMGG-Notbremse).
        if song_goal_targets and current_goal_scores and next_phase_id not in _NEVER_SKIP:
            try:
                # Bug-Fix: Deutsche Goal-Keys aus studio_goal_targets.py (natuerlichkeit, authentizitaet,
                # timbre_authentizitaet, artikulation) — nicht englische Namen. tonal_center ist in beiden gleich.
                _P1P2_GOALS = {
                    "natuerlichkeit",
                    "authentizitaet",
                    "tonal_center",
                    "timbre_authentizitaet",
                    "artikulation",
                }
                _goals_at_target = sum(
                    1
                    for g, target_val in song_goal_targets.items()
                    if g in current_goal_scores and float(current_goal_scores[g]) >= float(target_val) - 0.03
                )
                _total_goals = len(song_goal_targets)
                # Stopp wenn ≥ 80 % aller Ziele erreicht (inkl. P1/P2)
                _p1p2_at_target = sum(
                    1
                    for g in _P1P2_GOALS
                    if g in song_goal_targets
                    and g in current_goal_scores
                    and float(current_goal_scores[g]) >= float(song_goal_targets[g]) - 0.03
                )
                _p1p2_total = sum(1 for g in _P1P2_GOALS if g in song_goal_targets)
                if (
                    _total_goals > 0
                    and _goals_at_target >= int(0.80 * _total_goals)
                    and (_p1p2_total == 0 or _p1p2_at_target >= _p1p2_total)
                ):
                    # Bug-Fix: skip_recommended=True statt recommended_strength=0.0.
                    # recommended_strength=0.0 wurde sofort durch max(..., _DEFAULT_MIN_STRENGTH=0.05)
                    # auf 0.05 gehoben → Stop-Signal wirkungslos. skip_recommended=True bypasses
                    # den min_strength-Floor und verhindert dass UV3 einen Strength-Hint speichert.
                    _stop_reason = (
                        f"§2.31 Studio-Day-Target Stopp: {next_phase_id} — "
                        f"{_goals_at_target}/{_total_goals} Ziele erreicht, "
                        f"P1/P2 {_p1p2_at_target}/{_p1p2_total}"
                    )
                    logger.debug(_stop_reason)
                    return ConductorRecommendation(
                        next_phase_id=next_phase_id,
                        recommended_strength=0.0,
                        skip_recommended=True,
                        skip_reason=_stop_reason,
                        confidence=confidence,
                        state_snapshot=current_state,
                    )
            except Exception as e:
                logger.warning("Verarbeitungsschritt_conductor.py::unbekannter Ersatzpfad: %s", e)
                pass  # Non-blocking — Stopp-Signal-Fehler nie pipeline-blockierend

        if coalition_continuation:
            # Koalitions-Mitglieder dürfen nicht von einem einzelnen State-Snapshot
            # auf Skip oder zu starke Dämpfung gedrückt werden; Gruppe wird erst
            # nach Abschluss der Koalition beurteilt.
            recommended_strength = float(np.clip(max(recommended_strength, 0.55), 0.0, 1.0))
            confidence = float(np.clip(max(confidence, 0.75), 0.0, 1.0))

        # Mindest-Stärke aus Invariante
        min_str = _MIN_STRENGTH.get(next_phase_id, _DEFAULT_MIN_STRENGTH)
        recommended_strength = max(recommended_strength, min_str)

        # Skip-Empfehlung: nur wenn Noise-Floor bereits sehr gut UND HF voll
        skip = False
        skip_reason = ""
        if (
            current_state.noise_floor_db < -68.0
            and current_state.hf_energy_ratio > 0.55
            and current_state.harmonic_coherence > 0.88
            and recommended_strength < 0.12
            and not coalition_continuation
        ):
            # Signal bereits sehr sauber — Phase bringt kaum Gewinn
            skip = True
            skip_reason = (
                f"Conductor: noise_floor={current_state.noise_floor_db:.1f} dBFS, "
                f"hf={current_state.hf_energy_ratio:.2f} → kaum Restdefekt für {next_phase_id}"
            )
            logger.debug("PhaseConductor ueberspringen-Empfehlung: %s", skip_reason)

        return ConductorRecommendation(
            next_phase_id=next_phase_id,
            recommended_strength=recommended_strength,
            skip_recommended=skip and not coalition_continuation,
            skip_reason=skip_reason,
            confidence=confidence,
            state_snapshot=current_state,
        )

    def reset(self) -> None:
        """Setzt Histories zurück (nach jeder Song-Session aufrufen)."""
        with self._op_lock:
            self._history.clear()


# ── DSP-Merkmals-Extraktion (CPU, < 5 ms / Aufruf) ─────────────────────


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1) if audio.shape[1] <= 8 else np.mean(audio, axis=0)
    else:
        mono = audio
    return np.asarray(mono, dtype=np.float64)  # type: ignore[no-any-return]


def _rms_db(mono: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    return float(20.0 * np.log10(max(rms, 1e-9)))


def _estimate_noise_floor(mono: np.ndarray, sr: int) -> float:
    """5. Perzentil der Frame-RMS als Rauschboden-Schätzung [dBFS]."""
    frame_len = max(256, sr // 100)  # ~10 ms
    n = len(mono) // frame_len
    if n < 4:
        return float(20.0 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-9))
    rms_frames = np.array(
        [np.sqrt(np.mean(mono[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n)],
        dtype=np.float64,
    )
    p5 = float(np.percentile(rms_frames[rms_frames > 1e-9], 5)) if np.any(rms_frames > 1e-9) else 1e-9
    return float(20.0 * np.log10(max(p5, 1e-9)))


def _estimate_hf_energy_ratio(mono: np.ndarray, sr: int) -> float:
    """Anteil Hochfrequenzenergie (8 kHz – Nyquist) am Breitbandsignal."""
    fft_n = min(4096, len(mono))
    if fft_n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(mono[:fft_n] * np.hanning(fft_n))) ** 2
    freqs = np.fft.rfftfreq(fft_n, d=1.0 / sr)
    hf_mask = freqs >= 8000.0
    total = float(np.sum(spec) + 1e-12)
    hf = float(np.sum(spec[hf_mask]) + 1e-12)
    return float(np.clip(hf / total, 0.0, 1.0))


def _estimate_transient_density(mono: np.ndarray, sr: int) -> float:
    """Onset-Rate [Events/s] via einfache Flux-Methode."""
    if len(mono) < sr // 10:
        return 0.0
    hop = sr // 200  # 5 ms hop
    n_frames = len(mono) // hop
    rms = np.array(
        [np.sqrt(np.mean(mono[i * hop : (i + 1) * hop] ** 2)) for i in range(n_frames)],
        dtype=np.float64,
    )
    flux = np.maximum(0.0, np.diff(rms))
    threshold = float(np.mean(flux) + 1.5 * np.std(flux) + 1e-10)
    n_onsets = int(np.sum(flux > threshold))
    duration_s = len(mono) / sr
    return float(n_onsets / max(duration_s, 0.01))


def _estimate_harmonic_coherence(mono: np.ndarray, sr: int) -> float:
    """Harmonizitäts-Schätzung via normalisierter Autokorrelation am F0-Lag."""
    # Keep this bounded for realtime-safety in long E2E runs.
    # A 2048-sample window is sufficient for robust lag-peak estimation
    # while avoiding O(n^2) autocorrelation costs.
    max_samples = min(int(sr), 2048)
    segment = mono[: min(len(mono), max_samples)]
    if len(segment) < 64:
        return 0.5
    # FFT-based autocorrelation: O(n log n) instead of O(n^2).
    n = len(segment)
    n_fft = 1 << ((2 * n - 1).bit_length())
    spec = np.fft.rfft(segment, n=n_fft)
    ac = np.fft.irfft(spec * np.conj(spec), n=n_fft)[:n]
    ac_norm = ac / (ac[0] + 1e-12)
    # F0 search: 50–1000 Hz → lags sr/1000 .. sr/50
    lag_min = max(int(sr / 1000), 1)
    lag_max = min(int(sr / 50), len(ac_norm) - 1)
    if lag_min >= lag_max:
        return 0.5
    peak = float(np.max(ac_norm[lag_min:lag_max]))
    return float(np.clip(peak, 0.0, 1.0))


def _canonical_material(material_type: str) -> str:
    """Normalisiert diverse Material-Strings auf Grid-Keys."""
    m = str(material_type).lower()
    if "vinyl" in m or "lacquer" in m:
        return "vinyl"
    if "shellac" in m or "wax" in m or "wire" in m:
        return "shellac"
    if "tape" in m or "cassette" in m or "dat" in m or "reel" in m:
        return "tape"
    if "cd" in m or "digital" in m or "stream" in m or "mp3" in m or "aac" in m:
        return "cd_digital"
    return "unknown"


def _coalition_name_for_phase(
    phase_id: str | None,
    active_phase_coalitions: dict[str, tuple[str, ...]] | None,
) -> str:
    if not phase_id or not active_phase_coalitions:
        return ""
    for coalition_name, members in active_phase_coalitions.items():
        if phase_id in members:
            return str(coalition_name)
    return ""


# ── Singleton ──────────────────────────────────────────────────────────


def get_phase_conductor() -> PhaseConductor:
    """Gibt die globale PhaseConductor-Instanz zurück (thread-sicher)."""
    if _INSTANCE_HOLDER[0] is None:
        with _lock:
            if _INSTANCE_HOLDER[0] is None:
                _INSTANCE_HOLDER[0] = PhaseConductor()
    conductor = _INSTANCE_HOLDER[0]
    assert conductor is not None
    return conductor
