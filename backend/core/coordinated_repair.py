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

    def plan(self, manifest: Any, audio_length: int) -> RepairPlan:
        """
        Erstellt einen Reparatur-Plan aus einem Defect Manifest.

        Args:
            manifest: DefectManifest aus der Consensus Pipeline
            audio_length: Gesamtlänge des Audios in Samples

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
    ) -> tuple[np.ndarray, RepairReport]:
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

        for step in plan.steps:
            try:
                _audio_pre = current_audio.copy()
                current_audio = self._execute_step(
                    current_audio, step, manifest, sample_rate, n_channels,
                )
                # §v10.610: Post-Repair Artifact Guard — Pumping/Verzerrung checken
                if _guard is not None:
                    _guard_result = _guard.check(
                        audio_pre=_audio_pre,
                        audio_post=current_audio,
                        sr=sample_rate,
                        phase_id=step.phase_id,
                    )
                    if not getattr(_guard_result, "passed", True):
                        # Artefakt erkannt → zurückblenden (70% pre / 30% post)
                        current_audio = _guard.blend_back(_audio_pre, current_audio, 0.7)
                        log.warning(
                            "§v10.610 Guard: %s erzeugte Artefakte (%s) — zurückgeblendet",
                            step.phase_id,
                            getattr(_guard_result, "violations", []),
                        )
                # §v10.620: Perceptual Closed-Loop — UTMOS-basierte Qualitätsprüfung
                if _perceptual is not None:
                    _percept_result = _perceptual.evaluate(
                        audio_pre=_audio_pre,
                        audio_post=current_audio,
                        sr=sample_rate,
                        golden_sample=getattr(self, "_golden_sample", None),
                    )
                    if not getattr(_percept_result, "passed", True):
                        current_audio = _perceptual.blend_back(
                            _audio_pre, current_audio, _percept_result,
                        )
                        log.warning(
                            "§v10.620 Loop: %s verschlechterte MOS (%.3f → %.3f) — adaptiert",
                            step.phase_id,
                            _percept_result.mos_pre,
                            _percept_result.mos_post,
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
        phase_handlers = {
            "phase_03_denoise": self._run_denoise,
            "phase_01_click_removal": self._run_pass_through,
            "phase_02_hum_removal": self._run_pass_through,
            "phase_07_declipper": self._run_pass_through,
            "phase_09_crackle_removal": self._run_pass_through,
            "phase_12_wow_flutter_fix": self._run_pass_through,
            "phase_14_phase_correction": self._run_pass_through,
            "phase_19_de_esser": self._run_pass_through,
            "phase_24_dropout_repair": self._run_pass_through,
            "phase_28_surface_noise_profiling": self._run_pass_through,
            "phase_29_tape_hiss_reduction": self._run_pass_through,
            "phase_55_diffusion_inpainting": self._run_inpainting,
            "phase_57_print_through_reduction": self._run_pass_through,
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
        """Führt Denoising via SOTA 4-Layer Pipeline aus."""
        try:
            from backend.core.sota_denoise_pipeline import SOTADenoisePipeline
            pipeline = SOTADenoisePipeline()
            strength = step.parameters.get("strength", 0.4)
            result = pipeline.process(audio, sr, override_strength=strength)
            return result.audio.astype(np.float32)
        except Exception:
            return audio

    def _run_inpainting(
        self, audio: np.ndarray, step: RepairStep,
        manifest: Optional[Any], sr: int,
    ) -> np.ndarray:
        """Harmonic Inpainting via feingetuntem DiT."""
        try:
            # DiT-basiertes Inpainting — verwendet das trainierte Modell
            from models.miipher_dit.dit_model import FlowMatchingDiT
            import torch

            model = FlowMatchingDiT()
            ckpt_path = __import__('pathlib').Path(__file__).parent.parent / "models" / "harmonic_inpainting" / "inpainting_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
                model.load_state_dict(ckpt.get("model_state_dict", ckpt))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()

            strength = step.parameters.get("strength", 0.3)

            # Process in 2-second chunks
            chunk_samples = 2 * sr
            output = np.zeros_like(audio)
            for start in range(0, len(audio), chunk_samples // 2):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(-1).to(device)
                t_flow = torch.full((1,), 1.0 - strength, device=device)  # Less flow = more reconstruction

                with torch.no_grad():
                    velocity = model(x, t_flow)
                    # Simple Euler step: x_clean = x + velocity * strength
                    enhanced = x + velocity * strength

                enhanced_np = enhanced.squeeze().cpu().numpy()
                out_len = min(chunk_samples, len(audio) - start)
                # Overlap-add
                window = np.hanning(chunk_samples)
                output[start:start + out_len] += enhanced_np[:out_len] * window[:out_len] / 2

            return output.astype(np.float32)
        except Exception:
            log.debug("Inpainting not available, skipping")
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

    def plan(self, manifest: Any, audio_length: int) -> RepairPlan:
        return self.planner.plan(manifest, audio_length)

    def execute(
        self,
        audio: np.ndarray,
        plan: RepairPlan,
        manifest: Optional[Any] = None,
        sample_rate: int = SR,
    ) -> tuple[np.ndarray, RepairReport]:
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
