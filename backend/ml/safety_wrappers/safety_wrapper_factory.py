"""
safety_wrapper_factory.py - Automated Safety Wrapper Deployment

Automatically wraps all DSP modules with appropriate HIPS safety wrappers.
Uses module classification to assign generic wrappers.

Usage:
    from backend.ml.safety_wrappers.safety_wrapper_factory import wrap_all_modules

    wrapped_modules = wrap_all_modules()
    dehum = wrapped_modules['dehum']
    processed, report = dehum.process(audio, sr, strength=0.8)

Author: AURIK Team
Version: 1.0.0
Date: 8. Februar 2026
"""

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.core.musical_goals.processing_modes import ProcessingMode

from .generic_safety_wrapper import GenericNoiseReductionSafety, GenericRestorationSafety

logger = logging.getLogger(__name__)
from .generic_safety_wrapper_extended import GenericDynamicsSafety, GenericSpatialSafety, GenericSpectralSafety

# ============================================================================
# MODULE CLASSIFICATION
# ============================================================================

DSP_MODULE_CLASSIFICATION = {
    # Noise Reduction modules
    "noise_reduction": [
        "adaptive_imcra",
        "adaptive_mcra",
        "adaptive_mmse_lsa",
        "adaptive_mmse_stsa",
        "adaptive_mmse_noise_psd",
        "adaptive_omlsa",
        "adaptive_spectral_subtraction",
        "adaptive_spectral_gating",
        "adaptive_wiener_filter",
        "adaptive_noise_profiling",
        "adaptive_noise_profile_learning",
        "adaptive_histogram_noise",
        "adaptive_minimum_statistics",
        "adaptive_musical_noise_reduction",
        "spectral_denoiser",
        "spectral_gate",
        "spectral_subtractor",
        "sota_denoiser",
        "dehiss",
        "dehiss_multiband",
        "automatic_denoiser",
        "tape_noise_reduction",
        "reel_to_reel_noise_reduction",
    ],
    # Restoration modules
    "restoration": [
        "adaptive_spectral_inpainting",
        "adaptive_spectral_peak_removal",
        "adaptive_janssen_iterative",
        "adaptive_deconvolution",
        "automatic_declicker",
        "automatic_declicker_multiband",
        "automatic_declipper",
        "automatic_declipper_bass",
        "automatic_declipper_chain",
        "automatic_declipper_classic",
        "automatic_declipper_experimental",
        "automatic_declipper_instrument",
        "automatic_declipper_legacy",
        "automatic_declipper_low_latency",
        "automatic_declipper_multiband",
        "automatic_declipper_music",
        "automatic_declipper_percussive",
        "automatic_declipper_realtime",
        "automatic_declipper_reference",
        "automatic_declipper_stereo",
        "automatic_declipper_streaming",
        "automatic_declipper_ultra_low_latency",
        "automatic_declipper_voice",
        "automatic_decrackler",
        "automatic_debuzzer",
        "automatic_plosive_remover",
        "clickpop_remover",
        "declipper",
        "decrackler",
        "shellac_declicker",
        "riaa_declicker",
        "cd_error_correction",
        "adaptive_intermodulation_remover",
        "bandwidth_artifact_remover",
        "masking_remover",
        "hum_remover",
        "wow_flutter_remover",
    ],
    # Dynamics processing modules
    "dynamics": [
        "multiband_compressor",
        "multiband_expander",
        "multiband_gate",
        "multiband_limiter",
        "multiband_master",
        "custom_compressor",
        "intelligent_limiter",
        "limiter",
        "dynamic_range_expander",
        "broadband_dynamics_stabilizer",
        "adaptive_gain_rider",
        "gain_staging",
        "loudness_matching",
        "masking_aware_dynamic_eq",
    ],
    # Spectral processing modules
    "spectral": [
        "auto_eq",
        "perceptual_eq",
        "shellac_equalizer",
        "shellac_eq_contract",
        "tape_equalizer",
        "reel_to_reel_equalizer",
        "riaa_equalizer",
        "vinyl_emulation",
        "cd_deemphasis",
        "classic_filters",
        "allpass_filter",
        "linearphase_filter",
        "rumble_filter",
        "bandwidth_extender",
        "bandwidth_extension",
        "audio_super_resolution",
        "harmonic_exciter",
        "automatic_harmonics",
        "artifact_transient_enhancer",
        "transient_enhancer",
        "transient_shaper",
        "speaker_enhancement",
        "dynamic_spectral_tilt",
        "adaptive_spectral_centroid",
        "adaptive_spectral_flux",
        "adaptive_spectral_rolloff",
    ],
    # Spatial processing modules
    "spatial": [
        "stereo_widener",
        "stereo_enhancer",
        "stereo_image_correction",
        "stereo_matrix",
        "auto_panner_gain_ducker",
        "balance",
    ],
}


# ============================================================================
# CUSTOM WRAPPERS (already implemented)
# ============================================================================

CUSTOM_WRAPPERS = {
    "dehum": "dehum_safety.DeHumSafety",
    "declick": "declick_safety.DeClickSafety",
    "deesser": "deesser_safety.DeesserSafety",
    "context_aware_deesser": "context_aware_deesser_safety.ContextAwareDeesserSafety",
    "denoise": "denoise_safety.DenoiseSafety",
    "formant_shifter": "formant_shifter_safety.FormantShifterSafety",
    "harmonic_exciter": "harmonic_exciter_safety.HarmonicExciterSafety",
    "pitch_correction": "pitch_correction_safety.PitchCorrectionSafety",
    "stereo_widener": "stereo_widener_safety.StereoWidenerSafety",
    "vocal_declipping": "vocal_declipping_safety.VocalDeclippingSafety",
    "vocal_separation": "vocal_separation_safety.VocalSeparationSafety",
}


# ============================================================================
# SAFETY WRAPPER FACTORY
# ============================================================================


class SafetyWrapperFactory:
    """
    Factory für Erstellung und Verwaltung von HIPS-Sicherheits-Wrappern.

    Automatically:
    - Classifies DSP modules by type
    - Assigns appropriate generic wrapper
    - Uses custom wrapper if available
    - Tracks wrapper statistics
    """

    def __init__(self, processing_mode: ProcessingMode = ProcessingMode.RESTORATION, dsp_module_path: str = "dsp"):
        """
        Initialisiert factory.

        Args:
            processing_mode: Default processing mode for Musical Goals
            dsp_module_path: Path to DSP modules directory
        """
        self.processing_mode = processing_mode
        self.dsp_module_path = Path(dsp_module_path)
        self.wrapped_modules: dict[str, Any] = {}
        self.wrapper_stats: dict[str, int] = {
            "custom": 0,
            "noise_reduction": 0,
            "restoration": 0,
            "dynamics": 0,
            "spectral": 0,
            "spatial": 0,
            "unclassified": 0,
        }

    def classify_module(self, module_name: str) -> str:
        """
        Classify DSP module by type.

        Args:
            module_name: Name of DSP module

        Returns:
            Module type ('noise_reduction', 'restoration', etc.)
        """
        # Check if custom wrapper exists
        if module_name in CUSTOM_WRAPPERS:
            return "custom"

        # Check classification dict
        for module_type, modules in DSP_MODULE_CLASSIFICATION.items():
            if module_name in modules or any(module_name.startswith(mod) for mod in modules):
                return module_type

        # Fallback: heuristic classification based on name
        name_lower = module_name.lower()

        if "noise" in name_lower or "hiss" in name_lower or "gate" in name_lower:
            return "noise_reduction"
        elif "click" in name_lower or "clip" in name_lower or "crack" in name_lower:
            return "restoration"
        elif "compress" in name_lower or "limit" in name_lower or "gate" in name_lower or "expand" in name_lower:
            return "dynamics"
        elif "eq" in name_lower or "filter" in name_lower or "harmonic" in name_lower:
            return "spectral"
        elif "stereo" in name_lower or "pan" in name_lower or "spatial" in name_lower:
            return "spatial"

        return "unclassified"

    def create_wrapper_for_module(
        self, module_name: str, processor_func: Callable | None = None, module_version: str = "1.0.0"
    ) -> Any:
        """
        Erstellt appropriate safety wrapper for DSP module.

        Args:
            module_name: Name of DSP module
            processor_func: Processing function (if None, will try to import)
            module_version: Module version

        Returns:
            Safety wrapper instance
        """
        module_type = self.classify_module(module_name)
        self.wrapper_stats[module_type] += 1

        # Use custom wrapper if available
        if module_type == "custom":
            wrapper_path = CUSTOM_WRAPPERS[module_name]
            module_path, class_name = wrapper_path.rsplit(".", 1)

            try:
                wrapper_module = importlib.import_module(f"backend.ml.safety_wrappers.{module_path}")
                wrapper_class = getattr(wrapper_module, class_name)

                return wrapper_class(
                    module_name=module_name,
                    module_version=module_version,
                    processor_func=processor_func,
                    processing_mode=self.processing_mode,
                )
            except Exception as e:
                logger.warning("Warning: Could not laden custom wrapper for %s: %s", module_name, e)
                # Fall through to generic wrapper

        # Create generic wrapper
        if module_type == "noise_reduction":
            return GenericNoiseReductionSafety(
                module_name=module_name,
                module_version=module_version,
                processor_func=processor_func,  # type: ignore[arg-type]
                processing_mode=self.processing_mode,
            )

        elif module_type == "restoration":
            return GenericRestorationSafety(
                module_name=module_name,
                module_version=module_version,
                processor_func=processor_func,  # type: ignore[arg-type]
                processing_mode=self.processing_mode,
            )

        elif module_type == "dynamics":
            return GenericDynamicsSafety(
                module_name=module_name,
                module_version=module_version,
                processor_func=processor_func,  # type: ignore[arg-type]
                processing_mode=self.processing_mode,
            )

        elif module_type == "spectral":
            return GenericSpectralSafety(
                module_name=module_name,
                module_version=module_version,
                processor_func=processor_func,  # type: ignore[arg-type]
                processing_mode=self.processing_mode,
            )

        elif module_type == "spatial":
            return GenericSpatialSafety(
                module_name=module_name,
                module_version=module_version,
                processor_func=processor_func,  # type: ignore[arg-type]
                processing_mode=self.processing_mode,
            )

        else:
            # Unclassified: default to noise reduction (most conservative)
            return GenericNoiseReductionSafety(
                module_name=module_name,
                module_version=module_version,
                processor_func=processor_func,  # type: ignore[arg-type]
                processing_mode=self.processing_mode,
            )

    def wrap_all_modules(self, module_list: list | None = None) -> dict[str, Any]:
        """
        Kapselt all DSP modules with safety wrappers.

        Args:
            module_list: List of module names to wrap (if None, wrap all classified)

        Returns:
            Dict mapping module names to wrapped instances
        """
        if module_list is None:
            # Use all classified modules
            module_list = []
            for modules in DSP_MODULE_CLASSIFICATION.values():
                module_list.extend(modules)
            module_list.extend(CUSTOM_WRAPPERS.keys())

        wrapped = {}

        for module_name in module_list:
            try:
                wrapper = self.create_wrapper_for_module(module_name)
                wrapped[module_name] = wrapper
            except Exception as e:
                logger.warning("Warning: Could not wrap module %s: %s", module_name, e)

        self.wrapped_modules = wrapped
        return wrapped

    def get_statistics(self) -> dict[str, Any]:
        """Gibt zurück: wrapper deployment statistics."""
        return {
            "total_wrapped": sum(self.wrapper_stats.values()),
            "by_type": self.wrapper_stats.copy(),
            "coverage": {
                "noise_reduction": f"{self.wrapper_stats['noise_reduction']}/{len(DSP_MODULE_CLASSIFICATION['noise_reduction'])}",
                "restoration": f"{self.wrapper_stats['restoration']}/{len(DSP_MODULE_CLASSIFICATION['restoration'])}",
                "dynamics": f"{self.wrapper_stats['dynamics']}/{len(DSP_MODULE_CLASSIFICATION['dynamics'])}",
                "spectral": f"{self.wrapper_stats['spectral']}/{len(DSP_MODULE_CLASSIFICATION['spectral'])}",
                "spatial": f"{self.wrapper_stats['spatial']}/{len(DSP_MODULE_CLASSIFICATION['spatial'])}",
                "custom": f"{self.wrapper_stats['custom']}/{len(CUSTOM_WRAPPERS)}",
            },
        }

    def print_report(self):
        """Gibt aus: deployment report."""
        stats = self.get_statistics()

        logger.info(str("=" * 70))
        logger.info("HIPS SAFETY WRAPPER DEPLOYMENT REPORT")
        logger.info(str("=" * 70))
        logger.info("Total Modules Wrapped: %s", stats["total_wrapped"])
        logger.info("")
        logger.info("Coverage by Type:")
        for wrapper_type, coverage in stats["coverage"].items():
            logger.info("  %s: %s", wrapper_type, coverage)
        logger.info("")
        logger.info("Wrapper Type Distribution:")
        for wrapper_type, count in stats["by_type"].items():
            pct = 100 * count / max(1, stats["total_wrapped"])
            logger.info("  %s: %3d (%5.1f%%)", wrapper_type, count, pct)
        logger.info(str("=" * 70))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def wrap_all_modules(processing_mode: ProcessingMode = ProcessingMode.RESTORATION) -> dict[str, Any]:
    """
    Convenience function to wrap all DSP modules.

    Args:
        processing_mode: Processing mode for Musical Goals

    Returns:
        Dict mapping module names to wrapped instances
    """
    factory = SafetyWrapperFactory(processing_mode=processing_mode)
    wrapped = factory.wrap_all_modules()
    factory.print_report()
    return wrapped


def wrap_module(
    module_name: str,
    processor_func: Callable | None = None,
    processing_mode: ProcessingMode = ProcessingMode.RESTORATION,
    module_version: str = "1.0.0",
) -> Any:
    """
    Convenience function to wrap a single DSP module.

    Args:
        module_name: Name of DSP module
        processor_func: Processing function
        processing_mode: Processing mode for Musical Goals
        module_version: Module version

    Returns:
        Safety wrapper instance
    """
    factory = SafetyWrapperFactory(processing_mode=processing_mode)
    return factory.create_wrapper_for_module(
        module_name=module_name, processor_func=processor_func, module_version=module_version
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logger.info("Deploying HIPS Safety Wrappers to all DSP modules...")

    wrapped_modules = wrap_all_modules(ProcessingMode.RESTORATION)

    logger.info("\nerfolgreich wrapped %s modules!", len(wrapped_modules))
    logger.info("\nExample usage:")
    logger.info("  from backend.ml.safety_wrappers.safety_wrapper_factory import wrap_module")
    logger.info("  ")
    logger.info("  dehum = wrap_module('automatic_dehum')")
    logger.info("  verarbeitet, report = dehum.verarbeiten(audio, sr, strength=0.8)")
    logger.info("  ")
    logger.info("  if report.decision == ProcessingDecision.ROLLBACK_REQUIRED:")
    logger.info("      logger.info('Processing abgebrochen - returning Originalsignal')")
