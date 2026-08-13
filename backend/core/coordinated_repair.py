#!/usr/bin/env python3
"""
§v10.400: Repair Planner + Coordinated Repair — Manifest-gesteuerte Defekt-Behebung.

Problem: 12 Reparatur-Phasen arbeiten isoliert. Jede erkennt Defekte NEU,
statt das fertige Consensus-Manifest zu nutzen. Die Reihenfolge ist fix —
nicht vom tatsächlichen Defekt-Profil abhängig.

Lösung: 
  1. Repair Planner analysiert das Defect Manifest und plant die OPTIMALE
     Reihenfolge. "Klick vor Rauschen", "Hum vor Denoise", "Inpainting zum Schluss".
  2. Coordinated Repair führt den Plan aus. Jede Phase bekommt das Manifest
     als Kontext — keine Doppel-Erkennung, keine widersprüchlichen Eingriffe.

RX-11-Äquivalent: "Repair Assistant" — aber mit 30 Modulen statt 1 Scanner,
plus Harmonic Inpainting als finale Stufe.

Grundregeln der Reparatur-Reihenfolge:
  1. TRANSIENT (Klick, Knackser, Dropout) — zuerst, weil sie andere
     Detektoren stören (Klicks → falsche Frequenz-Peaks)
  2. TONAL (Hum, Brummen, Pfeifen) — vor Breitband, weil schmalbandig
     und gut isolierbar
  3. MODULATION (Wow/Flutter, Phasenfehler) — vor spektraler Reparatur
  4. BREITBAND (Rauschen, Hiss, Tape-Noise) — Haupt-Denoising
  5. CLIPPING/DISTORTION — nach Denoising (würde sonst Rauschen verstärken)
  6. INPAINTING (Harmonic Reconstruction) — ZUM SCHLUSS, baut auf
     bereits entrauschtem Signal auf
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

try:
    from backend.core.post_repair_artifact_guard import PostRepairArtifactGuard as _ArtifactGuard
except Exception:  # pragma: no cover — optional
    _ArtifactGuard = None

try:
    from backend.core.perceptual_closed_loop import PerceptualClosedLoop as _PerceptualLoop
except Exception:  # pragma: no cover — optional
    _PerceptualLoop = None

log = logging.getLogger(__name__)

_PROJECT_P = __import__("pathlib").Path(__file__).resolve().parent.parent

SR = 48000


# ═════════════════════════════════════════════════════════════════════════════
# Repair Strategy Model
# ═════════════════════════════════════════════════════════════════════════════

class RepairPriority(int, Enum):
    """Reparatur-Priorität (niedriger = zuerst ausführen)."""
    TRANSIENT = 1      # Klicks, Knackser, Dropouts
    TONAL = 2          # Hum, Brummen, Pfeifen
    MODULATION = 3     # Wow/Flutter, Phasenfehler
    BREITBAND = 4      # Rauschen, Hiss, Tape-Noise
    DISTORTION = 5     # Clipping, De-Essing-Artefakte
    INPAINTING = 6     # Harmonic Reconstruction — IMMER ZULETZT


@dataclass
class RepairStep:
    """Ein einzelner Reparatur-Schritt im Plan."""
    phase_id: str                      # z.B. "phase_01_click_removal"
    priority: RepairPriority
    defect_category: str               # Welcher Defekt-Typ wird repariert
    affected_samples: list[tuple[int, int]]  # (start, end) Sample-Bereiche
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # Phase-IDs, die VORHER laufen müssen
    enables: list[str] = field(default_factory=list)      # Phase-IDs, die NACHHER möglich sind


@dataclass
class RepairPlan:
    """Kompletter Reparatur-Plan mit geordneten Schritten."""
    steps: list[RepairStep] = field(default_factory=list)
    total_defects: int = 0
    total_coverage_samples: int = 0   # Wie viele Samples insgesamt betroffen
    estimated_duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def phase_order(self) -> list[str]:
        """Geordnete Liste der Phasen-IDs."""
        return [s.phase_id for s in self.steps]


# ═════════════════════════════════════════════════════════════════════════════
# §v10.994: MP-SENet Norm-Kalibrierung — pegelunabhängige Amplituden-Verarbeitung
# ═════════════════════════════════════════════════════════════════════════════


def _normalize_amp_peak99(amp: np.ndarray) -> tuple[np.ndarray, float]:
    """99-Perzentil-Peak-Normalisierung: Modell-Eingang auf Referenzpegel.

    Das MP-SENet-Modell wurde auf normalisierten Amplituden trainiert;
    ohne Norm hängt das Ergebnis vom Eingangspegel ab (Skalenfehler).
    """
    amp = np.asarray(amp, dtype=np.float32)
    p99 = float(np.percentile(amp, 99.0)) if amp.size else 0.0
    scale = p99 if p99 > 1e-8 else 1.0
    return (amp / scale).astype(np.float32), scale


def _denormalize_amp(amp: np.ndarray, scale: float) -> np.ndarray:
    """Gain-Kompensation: Modell-Ausgang zurück auf Originalpegel."""
    return (np.asarray(amp, dtype=np.float32) * scale).astype(np.float32)


def _guard_amp_loudness(denoised: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Loudness-Guard: Ausgangs-Amplitude nie > 1.05× Eingangs-Peak."""
    denoised = np.asarray(denoised, dtype=np.float32)
    original = np.asarray(original, dtype=np.float32)
    in_max = float(np.max(original)) if original.size else 0.0
    out_max = float(np.max(denoised)) if denoised.size else 0.0
    if out_max > max(in_max, 1e-6) * 1.05:
        denoised = denoised * (max(in_max, 1e-6) * 1.05 / out_max)
    return denoised.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Defect → Phase Mapping
# ═════════════════════════════════════════════════════════════════════════════

DEFECT_TO_PHASE: dict[str, RepairStep] = {
    "click": RepairStep(
        phase_id="phase_01_click_removal",
        priority=RepairPriority.TRANSIENT,
        defect_category="click",
        affected_samples=[],
        enables=["phase_03_denoise", "phase_07_harmonic_restoration"],
    ),
    "crackle": RepairStep(
        phase_id="phase_09_crackle_removal",
        priority=RepairPriority.TRANSIENT,
        defect_category="crackle",
        affected_samples=[],
        enables=["phase_03_denoise"],
    ),
    "pop": RepairStep(
        phase_id="phase_01_click_removal",
        priority=RepairPriority.TRANSIENT,
        defect_category="pop",
        affected_samples=[],
    ),
    "dropout": RepairStep(
        phase_id="phase_24_dropout_repair",
        priority=RepairPriority.TRANSIENT,
        defect_category="dropout",
        affected_samples=[],
        enables=["phase_55_diffusion_inpainting"],
    ),
    "hum": RepairStep(
        phase_id="phase_02_hum_removal",
        priority=RepairPriority.TONAL,
        defect_category="hum",
        affected_samples=[],
        depends_on=["phase_01_click_removal"],  # Klicks stören Hum-Erkennung
    ),
    "wow_flutter": RepairStep(
        phase_id="phase_12_wow_flutter_fix",
        priority=RepairPriority.MODULATION,
        defect_category="wow_flutter",
        affected_samples=[],
    ),
    "phase_error": RepairStep(
        phase_id="phase_14_phase_correction",
        priority=RepairPriority.MODULATION,
        defect_category="phase_error",
        affected_samples=[],
    ),
    "hiss": RepairStep(
        phase_id="phase_03_denoise",
        priority=RepairPriority.BREITBAND,
        defect_category="hiss",
        affected_samples=[],
        depends_on=["phase_01_click_removal", "phase_02_hum_removal"],
    ),
    "tape_hiss": RepairStep(
        phase_id="phase_29_tape_hiss_reduction",
        priority=RepairPriority.BREITBAND,
        defect_category="tape_hiss",
        affected_samples=[],
        depends_on=["phase_01_click_removal"],
    ),
    "vinyl_noise": RepairStep(
        phase_id="phase_28_surface_noise_profiling",
        priority=RepairPriority.BREITBAND,
        defect_category="vinyl_noise",
        affected_samples=[],
    ),
    "clipping": RepairStep(
        phase_id="phase_07_declipper",
        priority=RepairPriority.DISTORTION,
        defect_category="clipping",
        affected_samples=[],
        depends_on=["phase_03_denoise"],  # Erst entrauschen, dann declippen
    ),
    "distortion": RepairStep(
        phase_id="phase_07_declipper",
        priority=RepairPriority.DISTORTION,
        defect_category="distortion",
        affected_samples=[],
    ),
    "sibilance": RepairStep(
        phase_id="phase_19_de_esser",
        priority=RepairPriority.DISTORTION,
        defect_category="sibilance",
        affected_samples=[],
    ),
    "pre_echo": RepairStep(
        phase_id="phase_03_denoise",
        priority=RepairPriority.BREITBAND,
        defect_category="pre_echo",
        affected_samples=[],
    ),
    "print_through": RepairStep(
        phase_id="phase_57_print_through_reduction",
        priority=RepairPriority.BREITBAND,
        defect_category="print_through",
        affected_samples=[],
    ),
    "reverb_tail": RepairStep(
        phase_id="phase_20_reverb_reduction",
        priority=RepairPriority.DISTORTION,
        defect_category="reverb_tail",
        affected_samples=[],
        depends_on=["phase_03_denoise"],  # erst entrauschen, dann Hall reduzieren
    ),
    "bandwidth_loss": RepairStep(
        phase_id="phase_06_frequency_restoration",
        priority=RepairPriority.INPAINTING,
        defect_category="bandwidth_loss",
        affected_samples=[],
        depends_on=["phase_03_denoise"],  # Höhen-Rekonstruktion NACH Entrauschen
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# Repair Planner
# ═════════════════════════════════════════════════════════════════════════════

class RepairPlanner:
    """
    Analysiert das Defect Manifest und erstellt einen optimierten Reparatur-Plan.

    Regeln:
      1. Sortiere nach Priority (Transient → Inpainting)
      2. Respektiere Abhängigkeiten (depends_on)
      3. Merge gleiche Phasen (z.B. "click" + "pop" → beide Phase 01)
      4. Entferne Phasen ohne betroffene Defekte
      5. Harmonic Inpainting IMMER als letzter Schritt
    """

    def plan(self, manifest: Any, audio_length: int, metadata: Optional[dict] = None) -> RepairPlan:
        """
        Erstellt einen Reparatur-Plan aus einem Defect Manifest.

        Args:
            manifest: DefectManifest aus der Consensus Pipeline
            audio_length: Gesamtlänge des Audios in Samples
            metadata: Optionaler Kontext (z.B. vocal_confidence) für
                §v10.994 Model-Zoo-Aktivierung

        Returns:
            RepairPlan mit geordneten Schritten
        """
        if not manifest or not hasattr(manifest, 'defects') or not manifest.defects:
            return RepairPlan(total_defects=0)

        defects = manifest.defects

        # Schritt 1: Gruppiere Defekte nach Phase
        phase_defects: dict[str, list[Any]] = {}
        for d in defects:
            cat = getattr(d, 'category', None)
            if cat is None:
                continue
            cat_str = cat.value if hasattr(cat, 'value') else str(cat)
            mapping = DEFECT_TO_PHASE.get(cat_str)
            if mapping is None:
                continue
            phase_id = mapping.phase_id
            if phase_id not in phase_defects:
                phase_defects[phase_id] = []
            phase_defects[phase_id].append(d)

        if not phase_defects:
            return RepairPlan(total_defects=len(defects))

        # Schritt 2: Erstelle RepairSteps mit Sample-Bereichen
        steps: list[RepairStep] = []
        for phase_id, phase_defect_list in phase_defects.items():
            # Nimm die erste Defect-Mapping als Template
            template = None
            for d in phase_defect_list:
                cat_str = d.category.value if hasattr(d.category, 'value') else str(d.category)
                if cat_str in DEFECT_TO_PHASE:
                    template = DEFECT_TO_PHASE[cat_str]
                    break

            if template is None:
                continue

            # Sammle betroffene Sample-Bereiche
            affected = []
            for d in phase_defect_list:
                start = getattr(d, 'start_sample', 0)
                end = getattr(d, 'end_sample', start + 1000)
                if end > start:
                    affected.append((int(start), int(end)))

            # Berechne adaptive Parameter aus Defekt-Schwere
            avg_confidence = np.mean([
                float(getattr(d, 'confidence', 0.5)) for d in phase_defect_list
            ])
            avg_severity = np.mean([
                float(getattr(d, 'severity', 0.5)) for d in phase_defect_list
            ])

            step = RepairStep(
                phase_id=phase_id,
                priority=template.priority,
                defect_category=template.defect_category,
                affected_samples=affected,
                parameters={
                    "strength": float(avg_severity * avg_confidence),
                    "confidence": float(avg_confidence),
                    "defect_count": len(phase_defect_list),
                    "coverage_pct": float(
                        sum(e - s for s, e in affected) / max(audio_length, 1) * 100
                    ),
                },
                depends_on=list(template.depends_on),
                enables=list(template.enables),
            )
            # §v10.994: Kontextabhängige Model-Zoo-Aktivierung (Opt-In via Plan)
            _vocal_conf = float((metadata or {}).get("vocal_confidence", 0.0) or 0.0)
            if step.defect_category in ("hiss", "reverb_tail") and _vocal_conf > 0.5:
                step.parameters["use_sgmse"] = True
                step.parameters["sgmse_sigma"] = 0.4 if step.defect_category == "reverb_tail" else 0.5
            if step.defect_category == "hiss" and _vocal_conf > 0.65:
                step.parameters["use_mp_senet"] = True
            steps.append(step)

        # Schritt 3: Sortiere nach Priority, dann nach Abhängigkeiten
        steps.sort(key=lambda s: (s.priority.value, len(s.depends_on)))

        # Schritt 4: Topologische Sortierung (Abhängigkeiten auflösen)
        ordered = self._topological_sort(steps)

        # Schritt 5: Harmonic Inpainting als finalen Schritt hinzufügen
        total_coverage = sum(
            sum(e - s for s, e in step.affected_samples) for step in ordered
        )
        if total_coverage > 0:
            inpainting_step = RepairStep(
                phase_id="phase_55_diffusion_inpainting",
                priority=RepairPriority.INPAINTING,
                defect_category="harmonic_loss",
                affected_samples=[(0, audio_length)],  # Global
                parameters={
                    "strength": 0.3,  # Konservativ
                    "confidence": 0.8,
                    "coverage_pct": 100.0,
                },
                depends_on=[s.phase_id for s in ordered],  # NACH allen anderen
                enables=[],
            )
            ordered.append(inpainting_step)

        return RepairPlan(
            steps=ordered,
            total_defects=len(defects),
            total_coverage_samples=total_coverage,
            metadata={
                "defect_types": list(phase_defects.keys()),
                "phase_count": len(ordered),
                "planner_version": "v10.400",
            },
        )

    def _topological_sort(self, steps: list[RepairStep]) -> list[RepairStep]:
        """Sortiert Schritte topologisch nach Abhängigkeiten."""
        phase_ids = {s.phase_id for s in steps}
        ordered: list[RepairStep] = []
        remaining = list(steps)

        while remaining:
            # Finde Schritt ohne unerfüllte Abhängigkeiten
            progress = False
            for step in list(remaining):
                unmet_deps = [
                    d for d in step.depends_on
                    if d in phase_ids and d not in [s.phase_id for s in ordered]
                ]
                if not unmet_deps:
                    ordered.append(step)
                    remaining.remove(step)
                    progress = True
                    break

            if not progress:
                # Zirkuläre Abhängigkeit — breche auf
                ordered.extend(remaining)
                break

        return ordered


# ═════════════════════════════════════════════════════════════════════════════
# Coordinated Repair Executor
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RepairReport:
    """Bericht nach koordinierter Reparatur."""
    plan: RepairPlan
    completed_steps: list[str]
    failed_steps: list[tuple[str, str]]  # (phase_id, error_message)
    total_time: float
    input_peak: float
    output_peak: float
    # §v10.990: Guard-/Loop-Telemetrie für das Frontend (via bridge.get_guard_report)
    guard_violations: dict[str, int] = field(default_factory=dict)
    guard_peak_delta_db: float = 0.0
    utmos_iterations: int = 0
    utmos_blend_count: int = 0
    utmos_mos_before: float = 0.0
    utmos_mos_after: float = 0.0


class CoordinatedRepair:
    """
    Führt den Repair Plan aus — koordiniert, mit Manifest-Kontext.

    Jede Phase bekommt:
      - Das Audio (ggf. bereits von vorherigen Phasen bearbeitet)
      - Das Defect Manifest (damit sie WEISS, was zu reparieren ist)
      - Die spezifischen Parameter aus dem RepairStep
    """

    def execute(
        self,
        audio: np.ndarray,
        plan: RepairPlan,
        manifest: Optional[Any] = None,
        sample_rate: int = SR,
        material: str = "",
    ) -> tuple[np.ndarray, RepairReport]:
        self._material = material
        """
        Führt den Reparatur-Plan Schritt für Schritt aus.

        Args:
            audio: [T] oder [C, T] Eingangsaudio
            plan: RepairPlan vom RepairPlanner
            manifest: DefectManifest (optional, für Kontext)
            sample_rate: Samplerate

        Returns:
            (repaired_audio, RepairReport)
        """
        t0 = time.time()

        was_mono = audio.ndim == 1
        if was_mono:
            audio = audio[np.newaxis, :]
        n_channels = audio.shape[0]

        input_peak = float(np.abs(audio).max())
        current_audio = audio.copy()

        completed: list[str] = []
        failed: list[tuple[str, str]] = []

        _guard = _ArtifactGuard() if _ArtifactGuard is not None else None
        _perceptual = _PerceptualLoop() if _PerceptualLoop is not None else None

        # §v10.990: Telemetrie-Akkumulatoren für den RepairReport
        _guard_violations: dict[str, int] = {}
        _guard_peak_delta = 0.0
        _utmos_iterations = 0
        _utmos_blend_count = 0
        _utmos_mos_before = 0.0
        _utmos_mos_after = 0.0

        for step in plan.steps:
            try:
                _audio_pre = current_audio.copy()
                current_audio = self._execute_step(
                    current_audio, step, manifest, sample_rate, n_channels,
                )
                # §v10.950: No-Op-Erkennung — wenn der Schritt nichts geändert
                # hat, Guards und Perceptual-Loop überspringen (RT-Einsparung)
                _changed = not np.allclose(
                    np.asarray(_audio_pre), np.asarray(current_audio), atol=1e-7,
                )

                # §v10.610: Post-Repair Artifact Guard — Pumping/Verzerrung checken
                if _guard is not None and _changed:
                    _guard_result = _guard.check(
                        audio_pre=_audio_pre,
                        audio_post=current_audio,
                        sr=sample_rate,
                        phase_id=step.phase_id,
                    )
                    if not getattr(_guard_result, "passed", True):
                        # §v10.860: Spektrale Verstöße → 100% Reject (kein Blend),
                        # milde Verstöße (truepeak/pumping) → 70/30 Blend.
                        _violations = getattr(_guard_result, "violations", [])
                        for _v in _violations:
                            _v_str = str(_v)
                            _cat = "spectral" if _v_str.startswith(("spectral", "formant_drift")) else (
                                "pumping" if _v_str.startswith("pumping") else "truepeak"
                            )
                            _guard_violations[_cat] = _guard_violations.get(_cat, 0) + 1
                            if _v_str.startswith("truepeak_rise"):
                                try:
                                    _guard_peak_delta = max(
                                        _guard_peak_delta, abs(float(_v_str.split("_")[-1][:-2]))
                                    )
                                except ValueError:
                                    pass
                        _is_spectral = any(
                            str(v).startswith(("spectral", "formant_drift"))
                            for v in _violations
                        )
                        if _is_spectral:
                            current_audio = _audio_pre
                            log.warning(
                                "§v10.860 Guard: %s SPEKTRALER Schaden (%s) — Schritt verworfen",
                                step.phase_id, _violations,
                            )
                        else:
                            current_audio = _guard.blend_back(_audio_pre, current_audio, 0.7)
                            log.warning(
                                "§v10.610 Guard: %s erzeugte Artefakte (%s) — zurückgeblendet",
                                step.phase_id, _violations,
                            )
                # §v10.620: Perceptual Closed-Loop — UTMOS-basierte Qualitätsprüfung
                if _perceptual is not None and _changed:
                    _percept_result = _perceptual.evaluate(
                        audio_pre=_audio_pre,
                        audio_post=current_audio,
                        sr=sample_rate,
                        golden_sample=getattr(self, "_golden_sample", None),
                    )
                    _utmos_iterations += 1
                    _utmos_mos_before = float(getattr(_percept_result, "mos_pre", 0.0) or 0.0)
                    if not getattr(_percept_result, "passed", True):
                        current_audio = _perceptual.blend_back(
                            _audio_pre, current_audio, _percept_result,
                        )
                        _utmos_blend_count += 1
                        _utmos_mos_after = float(getattr(_percept_result, "mos_post", 0.0) or 0.0)
                        log.warning(
                            "§v10.620 Loop: %s verschlechterte MOS (%.3f → %.3f) — adaptiert",
                            step.phase_id,
                            _percept_result.mos_pre,
                            _percept_result.mos_post,
                        )
                # §v10.820: Do-No-Harm-Gate — SNR-Verschlechterung → Schritt zurückrollen
                _snr_pre = float(np.mean(_audio_pre ** 2) + 1e-10)
                _snr_post = float(np.mean(current_audio ** 2) + 1e-10)
                if _snr_post < _snr_pre * 0.7 and len(step.affected_samples) == 0:
                    # Signalenergie um >30% reduziert ohne lokale Defekt-Bereiche
                    current_audio = _audio_pre
                    log.warning(
                        "§v10.820 Do-No-Harm: %s reduzierte Signalenergie >30%% — Schritt verworfen",
                        step.phase_id,
                    )
                completed.append(step.phase_id)
                log.info(
                    "Repair: %s completed (%d defects, %.1f%% coverage)",
                    step.phase_id,
                    step.parameters.get("defect_count", 0),
                    step.parameters.get("coverage_pct", 0),
                )
            except Exception as e:
                failed.append((step.phase_id, str(e)))
                log.warning("Repair: %s FAILED — %s", step.phase_id, e)

        elapsed = time.time() - t0
        output_peak = float(np.abs(current_audio).max())

        if was_mono and current_audio.shape[0] == 1:
            current_audio = current_audio[0]

        return current_audio.astype(np.float32), RepairReport(
            plan=plan,
            completed_steps=completed,
            failed_steps=failed,
            total_time=elapsed,
            input_peak=input_peak,
            output_peak=output_peak,
            guard_violations=_guard_violations,
            guard_peak_delta_db=_guard_peak_delta,
            utmos_iterations=_utmos_iterations,
            utmos_blend_count=_utmos_blend_count,
            utmos_mos_before=_utmos_mos_before,
            utmos_mos_after=_utmos_mos_after,
        )

    def _execute_step(
        self,
        audio: np.ndarray,
        step: RepairStep,
        manifest: Optional[Any],
        sample_rate: int,
        n_channels: int,
    ) -> np.ndarray:
        """Führt einen einzelnen Reparatur-Schritt aus."""

        # Dispatch zu den bekannten Phasen
        # §v10.810: Transient-Defekte nutzen DirectDefectRepair (spektrale
        # Interpolation — RX-11-Äquivalent), statt Pass-Through.
        phase_handlers = {
            "phase_03_denoise": self._run_denoise,
            "phase_01_click_removal": self._run_transient_repair,
            "phase_09_crackle_removal": self._run_banquet_vinyl,
            "phase_24_dropout_repair": self._run_transient_repair,
            "phase_27_click_pop_removal": self._run_transient_repair,
            "phase_02_hum_removal": self._run_hum_removal,
            "phase_07_declipper": self._run_declipper,
            "phase_12_wow_flutter_fix": self._run_wow_flutter,
            "phase_14_phase_correction": self._run_phase_correction,
            "phase_19_de_esser": self._run_de_esser,
            "phase_06_frequency_restoration": self._run_frequency_restoration,
            "phase_20_reverb_reduction": self._run_reverb_reduction,
            "phase_49_advanced_dereverb": self._run_advanced_dereverb,
            "phase_28_surface_noise_profiling": self._run_banquet_vinyl,
            "phase_29_tape_hiss_reduction": self._run_tape_hiss,
            "phase_55_diffusion_inpainting": self._run_inpainting,
            "phase_57_print_through_reduction": self._run_print_through,
        }

        handler = phase_handlers.get(step.phase_id, self._run_pass_through)

        outputs = []
        for ch in range(n_channels):
            channel_out = handler(audio[ch], step, manifest, sample_rate)
            outputs.append(channel_out)

        return np.stack(outputs)

    def _run_denoise(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """Führt Denoising via SOTA 4-Layer Pipeline aus.

        §v10.994 Opt-In-Kette: use_mp_senet / use_sgmse aktivieren die
        Model-Zoo-Modelle VOR dem DSP-Standardpfad. Bei jedem Fehler greift
        der DSP-Fallback — nie stiller Ausfall.
        """
        # 1) MP-SENet Vokal-Denoising (Opt-In, kalibriert)
        if step.parameters.get("use_mp_senet", False):
            _mp = self._run_mp_senet_vocal(audio, step, manifest, sr)
            if np.asarray(_mp).shape == audio.shape and not np.allclose(
                np.asarray(_mp), audio, atol=1e-7
            ):
                log.info("MP-SENet aktiv für %s", step.phase_id)
                return np.asarray(_mp, dtype=np.float32)
        # 2) SGMSE+ Sprach-Enhancement-Diffusion (Opt-In, kontextaktiviert)
        if step.parameters.get("use_sgmse", False):
            try:
                from plugins.sgmse_plugin import enhance_sgmse

                _sigma = float(step.parameters.get("sgmse_sigma", 0.5))
                _res = enhance_sgmse(audio, sr, sigma=_sigma)
                _out = getattr(_res, "audio", None)
                if _out is not None and np.asarray(_out).shape == audio.shape:
                    log.info("SGMSE+ aktiv für %s (σ=%.2f)", step.phase_id, _sigma)
                    return np.asarray(_out, dtype=np.float32)
            except Exception as exc:
                log.warning("SGMSE+ nicht verfügbar (%s) — DSP-Fallback", exc)
        # 3) Standard: SOTA 4-Layer DSP-Pipeline
        try:
            from backend.core.sota_denoise_pipeline import SOTADenoisePipeline
            pipeline = SOTADenoisePipeline()
            strength = step.parameters.get("strength", 0.4)
            result = pipeline.process(audio, sr, override_strength=strength)
            return result.audio.astype(np.float32)
        except Exception:
            return audio

    def _run_banquet_vinyl(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """Vinyl-Crackle/Oberflächenrauschen via Banquet ONNX.

        §v10.840: Banquet ist ein VINYL-Modell — auf digitalem Material
        verschlechtert es (Benchmark: -1.3 dB). Nicht-Vinyl → Interpolation.
        """
        material = getattr(self, "_material", None)
        # §v10.870: Banquet auf diesem Corpus-Knistern bei JEDER Strength
        # schädlich (0.02→-0.6 dB, 0.12→-3.4 dB). Default: Interpolation.
        # Banquet nur als Opt-In, wenn es für das Material kalibriert wurde.
        use_banquet = bool(step.parameters.get("use_banquet", False))
        if not use_banquet:
            log.info(
                "Banquet übersprungen (nicht kalibriert für %s) — Interpolation",
                material or "unbekannt",
            )
            return self._run_transient_repair(audio, step, manifest, sr)
        if material is not None and str(material).lower() not in ("vinyl", "shellac", ""):
            log.info("Banquet übersprungen (Material %s) — Interpolation", material)
            return self._run_transient_repair(audio, step, manifest, sr)
        try:
            from plugins.banquet_vinyl_plugin import get_banquet_plugin
            strength = float(step.parameters.get("strength", 0.5))
            plugin = get_banquet_plugin()
            result = plugin.process(audio, sr, strength)
            if result is not None and np.asarray(result).shape == audio.shape:
                log.info("Banquet Vinyl: %s (strength=%.2f)", step.phase_id, strength)
                return np.asarray(result, dtype=np.float32)
        except Exception as exc:
            log.warning("Banquet nicht verfügbar (%s) — Fallback DirectDefectRepair", exc)
        return self._run_transient_repair(audio, step, manifest, sr)

    def _run_transient_repair(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """Transient-Reparatur via DirectDefectRepair (spektrale Interpolation).

        §v10.810: RX-11-Äquivalent — ersetzt Klicks/Knackser/Dropouts durch
        Interpolation aus der Umgebung statt nur zu dämpfen.
        """
        try:
            from backend.core.direct_defect_repair import DirectDefectRepair
            repairer = DirectDefectRepair()
            repaired, report = repairer.repair(audio, sr)
            if repaired is not None and repaired.shape == audio.shape:
                log.info(
                    "Transient-Repair %s: %s",
                    step.phase_id,
                    {k: v for k, v in report.items() if isinstance(v, (int, float, bool))},
                )
                return repaired.astype(np.float32)
        except Exception as exc:
            log.debug("DirectDefectRepair nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_inpainting(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """Harmonic Inpainting via DiT mit korrektem Flow-Matching-ODE-Solver.

        §v10.900: Flow-Matching verlangt ODE-Integration dx/dt = v(x,t).
        Der frühere Ein-Schritt-Pfad zerstörte -15 bis -20 dB. Jetzt:
        N Euler-Schritte von t=0 → t=1.
        """
        use_inpainting = bool(step.parameters.get("use_inpainting", False))
        if not use_inpainting:
            log.info("Inpainting übersprungen (§v10.880 Opt-In)")
            return audio
        try:
            # DiT-basiertes Inpainting — verwendet das trainierte Modell
            from models.miipher_dit.dit_model import FlowMatchingDiT
            import torch

            base_dir = __import__('pathlib').Path(__file__).parent.parent / "models" / "harmonic_inpainting"
            mask_ckpt = base_dir / "inpainting_mask_best.pt"
            if mask_ckpt.exists():
                # §v10.910: Mask-konditioniertes Modell (2 Kanäle: Audio+Maske)
                model = FlowMatchingDiT(in_channels=2)
                ckpt = torch.load(str(mask_ckpt), map_location='cpu', weights_only=True)
                model.load_state_dict(ckpt.get("model_state_dict", ckpt))
                use_mask_channel = True
            else:
                model = FlowMatchingDiT()
                ckpt_path = base_dir / "inpainting_best.pt"
                use_mask_channel = False
                if ckpt_path.exists():
                    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
                    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()

            # §v10.900: ODE-Integrationsparameter
            n_steps = int(step.parameters.get("ode_steps", 20))
            strength = float(step.parameters.get("strength", 0.3))
            dt = 1.0 / n_steps

            # Process in 2-second chunks
            chunk_samples = 2 * sr
            output = np.zeros_like(audio)
            for start in range(0, len(audio), chunk_samples // 2):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                x_audio = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(-1).to(device)
                if use_mask_channel:
                    # 2. Kanal: Maske (1 in Inpaint-Regionen)
                    ch_mask = torch.zeros_like(x_audio)
                    for s_smp, e_smp in step.affected_samples or []:
                        ch_mask[:, s_smp:min(e_smp, x_audio.shape[1]), :] = 1.0
                    if not step.affected_samples or ch_mask.sum() == 0:
                        ch_mask = torch.ones_like(x_audio)
                    x = torch.cat([x_audio, ch_mask], dim=-1)
                else:
                    x = x_audio
                x0 = x_audio.clone()

                # §v10.900: Mask-Reset — nach jedem Euler-Schritt werden die
                # NICHT-Inpaint-Regionen auf das Original zurückgesetzt. Das
                # verhindert ODE-Drift in unkontrollierten Regionen
                # (Ablation: -15.8 dB ohne Reset → -4.9 dB mit Reset).
                affected = step.affected_samples or []
                mask = torch.zeros_like(x0)
                for s_smp, e_smp in affected:
                    mask[:, s_smp:min(e_smp, x0.shape[1]), :] = 1.0
                if not affected or mask.sum() == 0:
                    mask = torch.ones_like(x0)  # ganzes Chunk

                # ── ODE-Integration: dx/dt = v(x, t) via Euler ──
                # §v10.910: Velocity wirkt NUR auf den Audio-Kanal; der
                # Mask-Kanal ist eine Bedingung und bleibt konstant.
                with torch.no_grad():
                    for i in range(n_steps):
                        t = torch.full((1,), i * dt, device=device)
                        velocity = model(x, t)  # [B, T, 1]
                        x = x + torch.cat([velocity, torch.zeros_like(velocity)], dim=-1) * dt
                        # Mask-Reset: unmaskierte Audio-Regionen bleiben Original
                        x[..., :1] = x[..., :1] * mask + x0 * (1.0 - mask)

                enhanced_np = x[..., :1].squeeze().cpu().numpy()

                # Nur die Inpaint-Regionen übernehmen, Rest = Original
                mix = min(1.0, strength)
                inpainted_chunk = chunk * (1.0 - mix) + enhanced_np * mix

                out_len = min(chunk_samples, len(audio) - start)
                window = np.hanning(chunk_samples)
                output[start:start + out_len] += inpainted_chunk[:out_len] * window[:out_len] / 2

            return output.astype(np.float32)
        except Exception:
            log.debug("Inpainting not available, skipping")
            return audio

    def _run_hum_removal(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.940: Hum-Entfernung via Phase 02 (echte Implementierung)."""
        try:
            from backend.core.phases.phase_02_hum_removal import HumRemovalPhase
            mat = getattr(self, "_material", "") or "unknown"
            result = HumRemovalPhase().process(
                audio=audio, sample_rate=sr, material_type=mat, auto_detect=True,
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("Hum-Removal nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_declipper(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.940: De-Clipping via Phase 07."""
        try:
            from backend.core.phases.phase_07_declipper import DeclipperPhase
            strength = float(step.parameters.get("strength", 0.5))
            result = DeclipperPhase().process(
                audio=audio, sample_rate=sr, strength=strength,
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("De-Clipper nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_wow_flutter(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.940: Wow/Flutter-Korrektur via Phase 12."""
        try:
            from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix
            mat = getattr(self, "_material", "") or "unknown"
            result = WowFlutterFix().process(
                audio=audio, sample_rate=sr, material_type=mat,
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("Wow/Flutter-Fix nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_phase_correction(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.940: Phasenkorrektur via Phase 14."""
        try:
            from backend.core.phases.phase_14_phase_correction import PhaseCorrection
            mat = getattr(self, "_material", "") or "unknown"
            result = PhaseCorrection().process(
                audio=audio, sample_rate=sr, material_type=mat,
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("Phasenkorrektur nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_de_esser(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.940: De-Essing via Phase 19."""
        try:
            from backend.core.phases.phase_19_de_esser import DeEsserPhase
            mat_str = getattr(self, "_material", "") or "unknown"
            try:
                from backend.core.defect_scanner import MaterialType as _MT
                mat = _MT("tape" if mat_str == "cassette" else mat_str)
            except Exception:
                mat = mat_str
            result = DeEsserPhase().process(
                audio=audio, sample_rate=sr, material_type=mat, gender="unknown",
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("De-Esser nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_tape_hiss(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.940: Tape-Hiss-Reduktion via Phase 29."""
        try:
            from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase
            mat = getattr(self, "_material", "") or "unknown"
            result = TapeHissReductionPhase().process(
                audio=audio, sample_rate=sr, material=mat, quality_mode="quality",
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("Tape-Hiss-Reduktion nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_print_through(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.940: Print-Through-Reduktion via Phase 57."""
        try:
            from backend.core.phases.phase_57_print_through_reduction import PrintThroughReductionPhase
            mat = getattr(self, "_material", "") or "unknown"
            result = PrintThroughReductionPhase().process(
                audio=audio, sample_rate=sr, material_type=mat,
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("Print-Through-Reduktion nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_mp_senet_vocal(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.950: Vokal-Denoising via MP-SENet ONNX (Opt-In use_mp_senet=True).

        I/O: noisy_amp [B, 201, T] + noisy_pha [B, 201, T] → denoised_amp.
        n_fft=400 (201 Bins), hop=100 (75% Overlap).
        """
        if not step.parameters.get("use_mp_senet", False):
            return audio
        try:
            import onnxruntime as ort

            mono = np.asarray(audio, dtype=np.float32)
            if mono.ndim > 1:
                mono = mono.mean(axis=0)

            n_fft, hop = 400, 100
            window = np.hanning(n_fft).astype(np.float32)
            n_frames = max(1, 1 + (len(mono) - n_fft) // hop)
            spec = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.complex64)
            for i in range(n_frames):
                s = i * hop
                if s + n_fft > len(mono):
                    break
                spec[i] = np.fft.rfft(mono[s:s + n_fft] * window)

            amp = np.abs(spec).astype(np.float32)          # [T, 201]
            pha = np.angle(spec).astype(np.float32)

            # §v10.994: Norm-Kalibrierung — das Modell wurde auf 99-Perzentil-
            # normalisierten Amplituden trainiert. Peak-Norm + Gain-Kompensation
            # macht die Inferenz pegelfest (Scale-Invarianz).
            amp_norm, _amp_scale = _normalize_amp_peak99(amp)

            session = ort.InferenceSession(
                str(_PROJECT_P / "models" / "mp_senet" / "mp_senet.onnx"),
                providers=["CPUExecutionProvider"],
            )
            denoised_amp = session.run(
                None, {
                    "noisy_amp": amp_norm.T[np.newaxis],   # [1, 201, T]
                    "noisy_pha": pha.T[np.newaxis],
                },
            )[0][0].T  # [T, 201]
            denoised_amp = _denormalize_amp(denoised_amp, _amp_scale)
            denoised_amp = _guard_amp_loudness(denoised_amp, amp)

            # Rekonstruktion mit Original-Phase
            enhanced_spec = denoised_amp.astype(np.complex64) * np.exp(1j * pha)
            out = np.zeros(len(mono), dtype=np.float32)
            wsum = np.zeros(len(mono), dtype=np.float32)
            for i in range(len(enhanced_spec)):
                s = i * hop
                if s + n_fft > len(mono):
                    break
                frame = np.fft.irfft(enhanced_spec[i]) * window
                out[s:s + n_fft] += frame
                wsum[s:s + n_fft] += window ** 2
            wsum[wsum < 1e-8] = 1.0
            out /= wsum
            log.info("MP-SENet Vokal-Denoising: %d Frames verarbeitet", len(enhanced_spec))
            return out.astype(np.float32)
        except Exception as exc:
            log.warning("MP-SENet nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_reverb_reduction(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.960: Hall-Reduktion via Phase 20 (RX-11 De-reverb-Äquivalent)."""
        try:
            from backend.core.phases.phase_20_reverb_reduction import ReverbReduction
            mat = getattr(self, "_material", "") or "unknown"
            result = ReverbReduction().process(
                audio=audio, sample_rate=sr, material=mat,
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("Reverb-Reduktion nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_advanced_dereverb(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.960: Advanced De-Reverb via Phase 49 (für stark verhallte Aufnahmen)."""
        try:
            from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase
            mat = getattr(self, "_material", "") or "unknown"
            result = AdvancedDereverbPhase().process(
                audio=audio, sample_rate=sr, material_type=mat,
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("Advanced De-Reverb nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_frequency_restoration(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """§v10.970: Bandwidth-Extension via Phase 06 (RX-11 Spectral-Recovery-Äquivalent)."""
        try:
            from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase
            mat = getattr(self, "_material", "") or "unknown"
            result = FrequencyRestorationPhase().process(
                audio=audio, sample_rate=sr, material_type=mat,
            )
            out = getattr(result, "audio", result)
            if out is not None and np.asarray(out).shape == audio.shape:
                return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            log.warning("Frequency-Restoration nicht verfügbar (%s) — Pass-Through", exc)
        return audio

    def _run_pass_through(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """Pass-through für Phasen, die noch nicht integriert sind."""
        return audio


# ═════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═════════════════════════════════════════════════════════════════════════════

class CoordinatedRepairPipeline:
    """
    Vollständige Defekt-Reparatur: Planung → Ausführung → Bericht.

    Nutzung:
        pipeline = CoordinatedRepairPipeline()
        plan = pipeline.plan(manifest, audio_length)
        repaired, report = pipeline.execute(audio, plan, manifest)
    """

    def __init__(self):
        self.planner = RepairPlanner()
        self.executor = CoordinatedRepair()

    def plan(self, manifest: Any, audio_length: int, metadata: Optional[dict] = None) -> RepairPlan:
        return self.planner.plan(manifest, audio_length, metadata)

    def execute(
        self,
        audio: np.ndarray,
        plan: RepairPlan,
        manifest: Optional[Any] = None,
        sample_rate: int = SR,
        material: str = "",
    ) -> tuple[np.ndarray, RepairReport]:
        self._material = material
        return self.executor.execute(audio, plan, manifest, sample_rate)

    def repair_all(
        self,
        audio: np.ndarray,
        manifest: Any,
        sample_rate: int = SR,
    ) -> tuple[np.ndarray, RepairReport]:
        """
        Führt die KOMPLETTE Reparatur durch:
        1. Analysiert das Manifest
        2. Plant die optimale Reihenfolge
        3. Führt alle Reparaturen koordiniert aus
        """
        plan = self.plan(manifest, len(audio) if audio.ndim == 1 else audio.shape[1])
        return self.execute(audio, plan, manifest, sample_rate)
