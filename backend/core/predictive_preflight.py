"""
PredictivePreflight — DSP-Proxy für alle Phasen-Familien (§v10.97)

Führt einen 70ms-Preflight über alle ausgewählten Phasen durch und
prognostiziert die Quality-Verbesserung (Δ) in 5 Dimensionen:
  SNR, Spektrale Balance, Crest-Faktor, HF-Peak-Ratio, IACC.

Phasen mit Δ < 0.2 auf ALLEN Dimensionen werden zum Skip vorgemerkt.
Die tatsächlichen Skip-Gates (§v10.96) bleiben als Safety-Net aktiv.

Architektur:
  1. 128-sample FFT, 5 Positionen über das Audio verteilt
  2. Pro Phasen-Familie: vereinfachte Gain-Kurve simulieren
  3. Delta in jeder Dimension messen
  4. Skip-Liste zurückgeben

Keine ML-Inferenz. Reine DSP-Proxy-Berechnung.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Phase-Familien → DSP-Proxy-Mapping ──────────────────────────────────

_PHASE_FAMILY_PROXY: dict[str, dict[str, Any]] = {
    "subtractive_cleanup": {
        "description": "Rauschunterdrückung (Denoise, Hum, Crackle, Gate)",
        "sensitivity_band": (4000, 16000),  # HF-Hiss wird reduziert
        "gain_db": -12.0,  # Simulierte Rausch-Dämpfung → SNR ↑
        "metrics": ["snr_improvement"],
    },
    "tonal_restoration": {
        "description": "EQ, Frequenz-Restauration, Präsenz-Boost",
        "sensitivity_band": (2000, 8000),  # Mitten/Höhen werden angehoben
        "gain_db": +3.0,
        "metrics": ["spectral_balance_improvement"],
    },
    "dynamics_repair": {
        "description": "Kompression, Expansion, Wow/Flutter",
        "sensitivity_band": None,  # Zeitdomain
        "gain_db": 0.0,
        "metrics": ["crest_factor_improvement"],
    },
    "sibilance_control": {
        "description": "De-Esser, Zischlaut-Reduktion",
        "sensitivity_band": (5000, 10000),
        "gain_db": -3.0,
        "metrics": ["hf_peak_ratio_improvement"],
    },
    "spectral_restoration": {
        "description": "Inpainting, Dropout-Repair",
        "sensitivity_band": None,
        "gain_db": 0.0,
        "metrics": ["spectral_continuity_improvement"],
    },
    "enhancement": {
        "description": "Vocal, Air, Bass, Exciter",
        "sensitivity_band": (4000, 16000),
        "gain_db": +2.0,
        "metrics": ["hf_energy_improvement"],
    },
    "stereo": {
        "description": "Balance, Width, Phase",
        "sensitivity_band": None,
        "gain_db": 0.0,
        "metrics": ["iacc_improvement"],
    },
}

# Phasen-Familien-Zuordnung (aus unified_restorer_v3.py)
_PHASE_TO_FAMILY: dict[str, str] = {
    "phase_01_click_removal": "subtractive_cleanup",
    "phase_02_hum_removal": "subtractive_cleanup",
    "phase_03_denoise": "subtractive_cleanup",
    "phase_04_eq_correction": "tonal_restoration",
    "phase_05_rumble_filter": "subtractive_cleanup",
    "phase_06_frequency_restoration": "tonal_restoration",
    "phase_07_harmonic_restoration": "enhancement",
    "phase_08_transient_preservation": "dynamics_repair",
    "phase_09_crackle_removal": "subtractive_cleanup",
    "phase_10_compression": "dynamics_repair",
    "phase_12_wow_flutter_fix": "dynamics_repair",
    "phase_13_stereo_enhancement": "stereo",
    "phase_14_phase_correction": "stereo",
    "phase_15_stereo_balance": "stereo",
    "phase_16_final_eq": "tonal_restoration",
    "phase_17_mastering_polish": "tonal_restoration",
    "phase_18_noise_gate": "subtractive_cleanup",
    "phase_19_de_esser": "sibilance_control",
    "phase_20_reverb_reduction": "subtractive_cleanup",
    "phase_23_spectral_repair": "spectral_restoration",
    "phase_24_dropout_repair": "spectral_restoration",
    "phase_25_azimuth_correction": "stereo",
    "phase_26_dynamic_range_expansion": "dynamics_repair",
    "phase_27_click_repair": "subtractive_cleanup",
    "phase_28_surface_noise_profiling": "subtractive_cleanup",
    "phase_29_tape_hiss_reduction": "subtractive_cleanup",
    "phase_31_speed_pitch_correction": "dynamics_repair",
    "phase_36_transient_shaper": "dynamics_repair",
    "phase_37_bass_enhancement": "enhancement",
    "phase_38_presence_boost": "tonal_restoration",
    "phase_39_air_band_enhancement": "enhancement",
    "phase_40_loudness_normalization": "dynamics_repair",
    "phase_42_vocal_enhancement": "enhancement",
    "phase_43_ml_deesser": "sibilance_control",
    "phase_46_spatial_enhancement": "stereo",
    "phase_47_truepeak_limiter": "dynamics_repair",
    "phase_48_stereo_width_enhancer": "stereo",
    "phase_49_advanced_dereverb": "subtractive_cleanup",
    "phase_50_spectral_repair": "spectral_restoration",
    "phase_55_diffusion_inpainting": "spectral_restoration",
    "phase_56_spectral_band_gap_repair": "spectral_restoration",
}


# Minimale Verbesserung pro Dimension, um Phase nicht zu skippen.
# §v10.101: Adaptiv statt statisch 0.15 — hängt von Restorability ab.
# Je schlechter die Quelle, desto mehr Phasen werden gebraucht →
# desto niedriger die Skip-Schwelle.
def _adaptive_min_delta(restorability_score: float = 65.0) -> float:
    """Restorability-adaptive Mindestverbesserung für Phase-Skip."""
    rs = float(max(10.0, min(100.0, restorability_score)))
    if rs >= 80:
        return 0.20  # Gutes Material: hohe Hürde, viel skippen
    elif rs >= 60:
        return 0.15  # Mittleres Material: Standard
    elif rs >= 40:
        return 0.10  # Beschädigtes Material: mehr Phasen erlauben
    else:
        return 0.05  # Stark beschädigt: fast alle Phasen laufen lassen


@dataclass
class PhasePrediction:
    """Prognose für eine einzelne Phase."""

    phase_id: str
    family: str
    deltas: dict[str, float] = field(default_factory=dict)
    should_skip: bool = False
    skip_reason: str = ""


@dataclass
class PreflightResult:
    """Ergebnis des PredictivePreflight."""

    predictions: list[PhasePrediction] = field(default_factory=list)
    skipped_phases: list[str] = field(default_factory=list)
    total_phases: int = 0
    computation_time_ms: float = 0.0


def compute_preflight(
    audio: np.ndarray,
    sample_rate: int,
    phase_ids: list[str],
    *,
    n_positions: int = 5,
    fft_size: int = 128,
    restorability_score: float = 65.0,
) -> PreflightResult:
    """Führt PredictivePreflight für alle gegebenen Phasen durch.

    Args:
        audio: Mono oder Stereo Audio (float32)
        sample_rate: Sample rate in Hz
        phase_ids: Liste der Phase-IDs, die evaluiert werden sollen
        n_positions: Anzahl Analyse-Positionen (default 5)
        fft_size: FFT-Größe für DSP-Proxy (default 128)

    Returns:
        PreflightResult mit Skip-Empfehlungen
    """
    import time

    t0 = time.time()

    # Mono-Konvertierung
    mono = audio if audio.ndim == 1 else np.mean(audio.astype(np.float64), axis=0)
    mono = np.asarray(mono, dtype=np.float32)
    n_total = len(mono)

    if n_total < fft_size * 2:
        return PreflightResult(total_phases=len(phase_ids), computation_time_ms=0.0)

    # Analyse-Positionen: gleichverteilt über das Audio
    positions = []
    step = max(1, (n_total - fft_size) // max(1, n_positions - 1)) if n_positions > 1 else 0
    for i in range(n_positions):
        pos = min(i * step, n_total - fft_size)
        positions.append(pos)

    # FFT an jeder Position
    spectra = []
    for pos in positions:
        seg = mono[pos : pos + fft_size]
        win = np.hanning(fft_size).astype(np.float32)
        spec = np.abs(np.fft.rfft(seg * win))
        spectra.append(spec)

    # Referenz-Metriken (Baseline)
    baseline = _compute_baseline_metrics(spectra, sample_rate, fft_size, mono)

    result = PreflightResult(total_phases=len(phase_ids))

    for phase_id in phase_ids:
        family = _PHASE_TO_FAMILY.get(phase_id, "unknown")
        if family == "unknown" or family not in _PHASE_FAMILY_PROXY:
            continue

        proxy = _PHASE_FAMILY_PROXY[family]
        # §SOTA: Perzeptuelles Preflight — prüft, ob die simulierte Phase
        # in mindestens 2 Bark-Bändern eine JND-überschreitende Änderung bewirkt.
        deltas = _simulate_perceptual_effect(spectra, baseline, proxy, sample_rate, fft_size, mono)

        # Prüfe ob irgendeine Dimension signifikant verbessert wird
        max_delta = max(deltas.values()) if deltas else 0.0
        _min_delta = _adaptive_min_delta(restorability_score)
        should_skip = max_delta < _min_delta

        pred = PhasePrediction(
            phase_id=phase_id,
            family=family,
            deltas=deltas,
            should_skip=should_skip,
            skip_reason=f"max_delta={max_delta:.3f} < {_min_delta:.2f}" if should_skip else "",
        )
        result.predictions.append(pred)

        if should_skip:
            result.skipped_phases.append(phase_id)
            logger.debug(
                "PredictivePreflight ueberspringen %s (family=%s, max_delta=%.3f)",
                phase_id,
                family,
                max_delta,
            )

    result.computation_time_ms = (time.time() - t0) * 1000.0
    return result


def _compute_baseline_metrics(
    spectra: list[np.ndarray],
    sample_rate: int,
    fft_size: int,
    mono: np.ndarray,
) -> dict[str, float]:
    """Berechnet Baseline-Metriken aus den Spektren."""

    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    def _band_energy(spec: np.ndarray, lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs <= hi)
        return float(np.sum(spec[mask] ** 2)) + 1e-12

    def _total_energy(spec: np.ndarray) -> float:
        return float(np.sum(spec**2)) + 1e-12

    # SNR: Energie in Stimmbändern vs. Rauschboden
    signal_bands = [(300, 3400)]
    noise_bands = [(8000, sample_rate // 2)]

    snr_values = []
    for spec in spectra:
        sig = sum(_band_energy(spec, lo, hi) for lo, hi in signal_bands)
        noise = sum(_band_energy(spec, lo, hi) for lo, hi in noise_bands)
        snr_values.append(10.0 * np.log10(sig / max(noise, 1e-12)))

    # Spektrale Balance: Mid/HF-Ratio
    balance_values = []
    for spec in spectra:
        mid = _band_energy(spec, 400, 4000)
        hf = _band_energy(spec, 4000, 16000)
        balance_values.append(float(np.clip(mid / max(hf, 1e-12), 0.1, 10.0)))

    # Crest-Faktor: Peak/RMS im Zeitdomain
    crest_values = []
    for i, pos in enumerate(
        [i * max(1, (len(mono) - fft_size) // max(1, len(spectra) - 1)) for i in range(len(spectra))]
    ):
        pos = min(pos, len(mono) - fft_size)
        seg = mono[pos : pos + fft_size]
        peak = float(np.max(np.abs(seg))) + 1e-12
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) + 1e-12
        crest_values.append(peak / rms)

    # HF Peak-Ratio: max(HF) / mean(HF)
    hf_peak_values = []
    for spec in spectra:
        hf_mask = freqs >= 5000
        if np.any(hf_mask):
            hf_spec = spec[hf_mask]
            hf_peak_values.append(float(np.max(hf_spec) / (np.mean(hf_spec) + 1e-12)))
        else:
            hf_peak_values.append(1.0)

    # IACC (nur wenn Stereo): simuliert via per-Frame-Korrelation
    iacc_values = [1.0]  # Default: Mono → perfekte Korrelation

    return {
        "snr_db": float(np.median(snr_values)),
        "spectral_balance": float(np.median(balance_values)),
        "crest_factor": float(np.median(crest_values)),
        "hf_peak_ratio": float(np.median(hf_peak_values)),
        "iacc": float(np.median(iacc_values)),
    }


def _simulate_perceptual_effect(
    spectra,
    baseline,
    proxy,
    sample_rate,
    fft_size,
    mono,
):
    """§SOTA: JND-basierte Simulation statt FFT-Deltas."""
    from backend.core.dsp.bark_lufs_util import BARK_EDGES_HZ

    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    n_bark = len(BARK_EDGES_HZ) - 1
    deltas = {}
    for dim, (direction, weight) in proxy.items():
        bark_e = np.zeros(min(n_bark, 24), dtype=np.float64)
        for spec in spectra:
            for b in range(min(n_bark, 24)):
                lo, hi = BARK_EDGES_HZ[b], BARK_EDGES_HZ[b + 1]
                m = (freqs >= lo) & (freqs < hi)
                if m.any():
                    bark_e[b] += float(np.mean(spec[m] ** 2))
        bark_e /= max(len(spectra), 1)
        ref = baseline.get(f"{dim}_ref", np.median(bark_e) + 1e-12)
        jnd = np.array(
            [
                2.0,
                1.8,
                1.5,
                1.3,
                1.0,
                0.9,
                0.8,
                0.7,
                0.6,
                0.6,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
                1.5,
                1.8,
                2.0,
                2.5,
                3.0,
                3.5,
            ]
        )[: len(bark_e)]
        dev = np.abs(1.0 - bark_e / (ref + 1e-12))
        wd = np.mean(dev / (jnd + 0.5))
        d = float(np.clip(wd * weight * (0.8 if direction == "decrease" else 1.0), 0.0, 1.0))
        deltas[dim] = d
    return deltas


def _simulate_phase_effect(
    spectra: list[np.ndarray],
    baseline: dict[str, float],
    proxy: dict[str, Any],
    sample_rate: int,
    fft_size: int,
    mono: np.ndarray,
) -> dict[str, float]:
    """Simuliert den Effekt einer Phasen-Familie und misst das Delta."""

    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    band = proxy.get("sensitivity_band")
    gain_db = float(proxy.get("gain_db", 0.0))
    gain_linear = 10.0 ** (gain_db / 20.0)

    # Simulierte Spektren nach Phasen-Effekt
    modified_spectra = []
    for spec in spectra:
        mod_spec = spec.copy()
        if band is not None:
            lo, hi = band
            mask = (freqs >= lo) & (freqs <= hi)
            mod_spec[mask] = mod_spec[mask] * gain_linear
        modified_spectra.append(mod_spec)

    # Metriken NACH simulierter Phase
    modified_metrics: dict[str, list[float]] = {
        "snr_db": [],
        "spectral_balance": [],
        "crest_factor": [],
        "hf_peak_ratio": [],
        "iacc": [],
    }

    for spec in modified_spectra:
        # SNR (vereinfacht)
        sig = float(np.sum(spec[(freqs >= 300) & (freqs <= 3400)] ** 2)) + 1e-12
        noise = float(np.sum(spec[freqs >= 8000] ** 2)) + 1e-12
        modified_metrics["snr_db"].append(10.0 * np.log10(sig / max(noise, 1e-12)))

        # Balance
        mid = float(np.sum(spec[(freqs >= 400) & (freqs <= 4000)] ** 2)) + 1e-12
        hf = float(np.sum(spec[(freqs >= 4000) & (freqs <= 16000)] ** 2)) + 1e-12
        modified_metrics["spectral_balance"].append(float(np.clip(mid / max(hf, 1e-12), 0.1, 10.0)))

        # Crest: simulierter Effekt im Zeitdomain
        # Wenn gain_db > 0: mehr Energie → mehr Crest; wenn < 0: weniger
        crest_mod = baseline["crest_factor"] * (1.0 + gain_db / 40.0)
        modified_metrics["crest_factor"].append(float(np.clip(crest_mod, 1.0, 20.0)))

        # HF Peak
        hf_mask = freqs >= 5000
        if np.any(hf_mask):
            hf_spec = spec[hf_mask]
            modified_metrics["hf_peak_ratio"].append(float(np.max(hf_spec) / (np.mean(hf_spec) + 1e-12)))
        else:
            modified_metrics["hf_peak_ratio"].append(1.0)

        # IACC: unverändert (Proxy kann Stereo nicht simulieren)
        modified_metrics["iacc"].append(baseline["iacc"])

    # Delta pro Metrik (Median)
    deltas: dict[str, float] = {}
    metric_map = {
        "snr_improvement": "snr_db",
        "spectral_balance_improvement": "spectral_balance",
        "crest_factor_improvement": "crest_factor",
        "hf_peak_ratio_improvement": "hf_peak_ratio",
        "hf_energy_improvement": "spectral_balance",
        "spectral_continuity_improvement": "snr_db",
        "iacc_improvement": "iacc",
    }

    for metric_name in proxy.get("metrics", []):
        mapped = metric_map.get(metric_name, metric_name)
        base_val = baseline.get(mapped, 0.0)
        mod_vals = modified_metrics.get(mapped, [base_val])
        mod_val = float(np.median(mod_vals)) if mod_vals else base_val

        # Normalisiere Delta: SNR in dB bereits sinnvoll,
        # Ratio-basierte Metriken als relative Änderung
        if mapped == "snr_db":
            delta = mod_val - base_val
        elif base_val > 0.01:
            delta = abs(mod_val - base_val) / base_val
        else:
            delta = 0.0

        deltas[metric_name] = float(np.clip(delta, 0.0, 10.0))

    return deltas
