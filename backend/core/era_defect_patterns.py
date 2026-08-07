"""backend/core/era_defect_patterns.py — Era × Material → typische Defektmuster

Aurik's autonomous decision-making depends on knowing *before* signal analysis
which defects are statistically probable for a given material and era.  This
knowledge base encodes decades of archival and mastering practice into a
lookup table so the :class:`~backend.core.defect_scanner.DefectScanner` can
weight its findings against realistic priors, and the
:class:`~backend.core.causal_defect_reasoner.CausalDefectReasoner` can predict
defects that may be present but below the detection threshold.

Scientific references:
  Copeland (2008) «Manual of Analogue Sound Restoration» — BBC degradation taxonomy
  Schüller (2004) «Preservation of Audio Materials» (IASA-TC 03/04)
  Hockman & Davies (2011) «Analysis of vinyl surface noise»
  Rumsey & McCormick (2009) «Sound and Recording» — codec artefact appendix
  Katz (2007) «Mastering Audio» — transfer chain degradation
  Zeller (2012) «DDR Rundfunktechnik»
  Eargle (2004) «The Microphone Book»
  IEC 60386:1987 — wow/flutter limits per medium
  IASA-TC 04 (2009) — access copy guidelines

Usage::

    from backend.core.era_defect_patterns import get_era_material_defect_priors
    priors = get_era_material_defect_priors(era_decade=1950, material_type="shellac")
    # priors["CRACKLE"] -> (severity_prior: float, probability: float)
    for defect_name, (sev, prob) in priors.items():
        if prob > 0.6:
            logger.info("High prior for %s (prob=%.2f)", defect_name, prob)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefectPrior:
    """Statistical expectation for one defect type in a given era × material context.

    Attributes:
        severity_prior:  Expected typical severity [0..1] — feeds DefectScanner
                         baseline calibration.
        probability:     Probability that this defect is present at all [0..1].
                         < 0.10 → essentially absent; ≥ 0.60 → very common.
        notes:           Free-text explanation for logging / UI.
    """

    severity_prior: float  # [0..1]
    probability: float  # [0..1]
    notes: str = ""


# ---------------------------------------------------------------------------
# Era × Material defect tables
# ---------------------------------------------------------------------------
# Structure:  {(era_decade, material_key): {defect_name: DefectPrior}}
#
# era_decade  – decade start year (1890, 1900, 1910, … 2020)
# material_key – canonical material string (matches backend/core/medium_detector.py)
# defect_name  – matches DefectType.name in defect_scanner.py (UPPERCASE)
#
# Convention for probability:
#   0.90+ = almost always present in archival specimens
#   0.70–0.89 = very common
#   0.50–0.69 = common
#   0.30–0.49 = occasional
#   0.10–0.29 = rare
#   < 0.10   = essentially absent (not listed)
# ---------------------------------------------------------------------------

_ERA_MATERIAL_PRIORS: dict[tuple[int, str], dict[str, DefectPrior]] = {
    # ===================================================================
    # WAX CYLINDERS (1880s–1912)  ★ Extremely degraded format
    # ===================================================================
    (1890, "wax_cylinder"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.75, 0.95, "Surface crumbling + white noise floor"),
        "CRACKLE": DefectPrior(0.70, 0.93, "Wax micro-fractures cause continuous crackle"),
        "BANDWIDTH_LOSS": DefectPrior(0.80, 0.98, "Physical BW ceiling ≤ 4 kHz"),
        "WOW": DefectPrior(0.55, 0.85, "Early mandrel irregularities → ±1–2 % wow"),
        "FLUTTER": DefectPrior(0.45, 0.70, "Mechanical irregularity in early machines"),
        "LOW_FREQ_RUMBLE": DefectPrior(0.40, 0.65, "Motor rumble from spring-wound mechanisms"),
        "DROPOUTS": DefectPrior(0.60, 0.80, "Wax delamination causes signal loss"),
        "DC_OFFSET": DefectPrior(0.30, 0.55, "Early ADC or offset in playback chain"),
        "PITCH_DRIFT": DefectPrior(0.65, 0.88, "Spring-wound motors → variable speed"),
        "MODULATION_NOISE": DefectPrior(0.35, 0.50, "Wax grain modulated by signal"),
    },
    (1900, "wax_cylinder"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.72, 0.93, "Surface degradation"),
        "CRACKLE": DefectPrior(0.68, 0.91, "Wax micro-fractures"),
        "BANDWIDTH_LOSS": DefectPrior(0.82, 0.98, "Physical BW ≤ 5 kHz"),
        "WOW": DefectPrior(0.50, 0.82, "Spring-wound motor variations"),
        "FLUTTER": DefectPrior(0.40, 0.65, "Mechanical irregularity"),
        "PITCH_DRIFT": DefectPrior(0.60, 0.85, "Variable spring tension over play"),
        "DROPOUTS": DefectPrior(0.55, 0.75, "Wax delamination"),
        "LOW_FREQ_RUMBLE": DefectPrior(0.38, 0.62, "Motor rumble"),
    },
    # ===================================================================
    # SHELLAC 78 RPM (1910–1958)
    # Bandwidth ~4–7 kHz depending on era; very high surface noise (SNR ≈ 10–20 dB)
    # ===================================================================
    (1910, "shellac"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.72, 0.95, "Shellac compound formula: high grain noise"),
        "CRACKLE": DefectPrior(0.65, 0.92, "Surface wear from multiple plays"),
        "BANDWIDTH_LOSS": DefectPrior(0.70, 0.98, "Physical BW ≤ 5 kHz"),
        "WOW": DefectPrior(0.45, 0.78, "Shellac pressing eccentricity"),
        "LOW_FREQ_RUMBLE": DefectPrior(0.35, 0.60, "Acoustic gramophone horn resonance"),
        "PITCH_DRIFT": DefectPrior(0.40, 0.70, "78 RPM motor variation ±3 %"),
        "CLICKS": DefectPrior(0.50, 0.80, "Groove damage from sharp needles"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.55, 0.72, "Tracing distortion at inner groove"),
    },
    (1920, "shellac"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.70, 0.94, "Filler material (slate) causes surface hiss"),
        "CRACKLE": DefectPrior(0.65, 0.91, "Multi-play steel needle damage"),
        "CLICKS": DefectPrior(0.55, 0.85, "Steel-needle impacts on groove walls"),
        "BANDWIDTH_LOSS": DefectPrior(0.72, 0.97, "Acoustic BW ≤ 6 kHz"),
        "WOW": DefectPrior(0.42, 0.75, "78 RPM eccentricity ±2 %"),
        "PITCH_DRIFT": DefectPrior(0.38, 0.65, "Spring-motor variable speed"),
        "LOW_FREQ_RUMBLE": DefectPrior(0.30, 0.55, "Heavy tone-arm resonance"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.50, 0.70, "Late-sided tracing distortion"),
        "PRINT_THROUGH": DefectPrior(0.18, 0.30, "Pre-echo from wax master stacking"),
    },
    (1930, "shellac"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.68, 0.92, "Improved compound but still high floor"),
        "CRACKLE": DefectPrior(0.62, 0.88, "Carbon/slate filler + needle wear"),
        "CLICKS": DefectPrior(0.52, 0.82, "Groove impacts; heavy stylus force"),
        "BANDWIDTH_LOSS": DefectPrior(0.70, 0.96, "Electrical recording BW ≤ 7 kHz"),
        "WOW": DefectPrior(0.38, 0.70, "Electrical turntable motors better"),
        "PITCH_DRIFT": DefectPrior(0.28, 0.52, "AC synchronous motors improve consistency"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.48, 0.68, "High lateral displacement at inner groove"),
        "SOFT_SATURATION": DefectPrior(0.35, 0.60, "Cutter head harmonic distortion ≈ 2–3 %"),
        "HUM": DefectPrior(0.30, 0.58, "AC interference in early electrical chain"),
        "PRINT_THROUGH": DefectPrior(0.15, 0.28, "Shellac master pre-echo (faint)"),
    },
    (1940, "shellac"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.65, 0.90, "War-era compound substitutions"),
        "CRACKLE": DefectPrior(0.60, 0.86, "Wartime shellac shortages → inferior filler"),
        "CLICKS": DefectPrior(0.50, 0.80, "Needle wear; increased plays"),
        "BANDWIDTH_LOSS": DefectPrior(0.65, 0.95, "BW ≤ 8 kHz at best"),
        "DROPOUTS": DefectPrior(0.38, 0.62, "Wartime pressing defects"),
        "HUM": DefectPrior(0.32, 0.60, "Power supply interference"),
        "SOFT_SATURATION": DefectPrior(0.40, 0.65, "Lacquer master saturation in recording"),
        "WOW": DefectPrior(0.30, 0.55, "Turntable consistency improved"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.45, 0.65, "Persistent cutter geometry limit"),
    },
    (1950, "shellac"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.60, 0.88, "Late shellac era; compound slightly better"),
        "CRACKLE": DefectPrior(0.55, 0.83, "Still high due to formula"),
        "CLICKS": DefectPrior(0.45, 0.75, "Nylon/sapphire needles reduce groove damage"),
        "BANDWIDTH_LOSS": DefectPrior(0.60, 0.93, "Late 78 RPM BW ≤ 8 kHz"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.42, 0.62, "Standard cutter geometry"),
        "SOFT_SATURATION": DefectPrior(0.35, 0.58, "Lacquer master saturation"),
        "WOW": DefectPrior(0.25, 0.45, "Electric turntable motors standard"),
    },
    # ===================================================================
    # VINYL (1948–present)
    # SNR 40–70 dB; BW to 16 kHz (later 20 kHz); wow/flutter < 0.2% WRMS
    # ===================================================================
    (1950, "vinyl"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.30, 0.65, "Early vinyl: some compound noise"),
        "CRACKLE": DefectPrior(0.35, 0.70, "Steel-needle legacy on early vinyl"),
        "RIAA_CURVE_ERROR": DefectPrior(0.40, 0.72, "RIAA standardized 1954; pre-RIAA pressings common"),
        "CLICKS": DefectPrior(0.28, 0.58, "Sharp needle impact on microgroove"),
        "WOW": DefectPrior(0.22, 0.45, "33 RPM turntable WOW ≤ 0.2% WRMS"),
        "LOW_FREQ_RUMBLE": DefectPrior(0.35, 0.62, "Belt-drive resonance; bearing noise"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.40, 0.68, "Cutter geometry at inner groove"),
        "SOFT_SATURATION": DefectPrior(0.25, 0.50, "Lacquer master saturation"),
        "BANDWIDTH_LOSS": DefectPrior(0.20, 0.40, "Early pressings BW ≤ 14 kHz"),
        "STEREO_IMBALANCE": DefectPrior(0.15, 0.30, "Early stereo alignment issues"),
    },
    (1960, "vinyl"): {
        "CRACKLE": DefectPrior(0.28, 0.62, "Normal vinyl surface noise"),
        "CLICKS": DefectPrior(0.22, 0.52, "Stylus impacts; dust particles"),
        "LOW_FREQ_RUMBLE": DefectPrior(0.30, 0.58, "Turntable bearing noise"),
        "RIAA_CURVE_ERROR": DefectPrior(0.20, 0.40, "Some pressings still pre-RIAA"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.35, 0.62, "Classic vinyl geometry limit"),
        "WOW": DefectPrior(0.15, 0.35, "Improved motors; IEC standard"),
        "FLUTTER": DefectPrior(0.10, 0.25, "Belt-drive flutter"),
        "GROOVE_ECHO": DefectPrior(0.18, 0.38, "Adjacent groove pre-echo"),
        "STEREO_IMBALANCE": DefectPrior(0.12, 0.28, "Cartridge alignment"),
        "HIGH_FREQ_NOISE": DefectPrior(0.22, 0.48, "Vinyl surface and pressing"),
    },
    (1970, "vinyl"): {
        "CRACKLE": DefectPrior(0.25, 0.58, "Standard vinyl aging"),
        "CLICKS": DefectPrior(0.20, 0.50, "Stylus and groove wear"),
        "LOW_FREQ_RUMBLE": DefectPrior(0.28, 0.55, "Bearing noise; resonance at 20–50 Hz"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.30, 0.58, "Tracing distortion"),
        "GROOVE_ECHO": DefectPrior(0.20, 0.42, "Groove-to-groove echo ≈ 1 revolution"),
        "HIGH_FREQ_NOISE": DefectPrior(0.18, 0.42, "Vinyl surface noise"),
        "SOFT_SATURATION": DefectPrior(0.15, 0.32, "Compression on master"),
        "DYNAMIC_COMPRESSION_EXCESS": DefectPrior(0.30, 0.60, "1970s radio-ready compression"),
        "WOW": DefectPrior(0.12, 0.28, "Mature turntable technology"),
    },
    (1980, "vinyl"): {
        "CRACKLE": DefectPrior(0.22, 0.55, "Aging vinyl from 1980s pressings"),
        "CLICKS": DefectPrior(0.18, 0.48, "Normal stylus wear pattern"),
        "LOW_FREQ_RUMBLE": DefectPrior(0.25, 0.50, "Turntable isolation improving"),
        "DYNAMIC_COMPRESSION_EXCESS": DefectPrior(0.38, 0.68, "Heavy 1980s loudness war"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.25, 0.52, "Tracing distortion"),
        "HIGH_FREQ_NOISE": DefectPrior(0.15, 0.38, "Better compound formulas"),
        "GROOVE_ECHO": DefectPrior(0.15, 0.35, "Pre-echo from groove geometry"),
        "SOFT_SATURATION": DefectPrior(0.20, 0.42, "Mastering saturation"),
    },
    # ===================================================================
    # REEL-TO-REEL TAPE (1948–1990)
    # 15/30 ips professional; BW 20–20 kHz at 30 ips; SNR 65–72 dB
    # ===================================================================
    (1950, "reel_tape"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.28, 0.60, "Early oxide formulations; bias noise"),
        "PRINT_THROUGH": DefectPrior(0.35, 0.70, "Tape pack echo ≈ ±270 ms at 7.5 ips"),
        "WOW": DefectPrior(0.20, 0.45, "Capstan/pinch-roller irregularities"),
        "FLUTTER": DefectPrior(0.25, 0.52, "Reel-servo flutter ≈ 6–12 Hz"),
        "DROPOUTS": DefectPrior(0.22, 0.48, "Oxide shedding on early formulations"),
        "HUM": DefectPrior(0.25, 0.52, "AC hum from early electronics"),
        "BANDWIDTH_LOSS": DefectPrior(0.15, 0.38, "7.5 ips BW ≤ 15 kHz"),
        "BIAS_ERROR": DefectPrior(0.28, 0.55, "Early bias calibration imprecision"),
        "DC_OFFSET": DefectPrior(0.18, 0.40, "Early preamp offset"),
    },
    (1960, "reel_tape"): {
        "PRINT_THROUGH": DefectPrior(0.30, 0.65, "Oxide pack magnetization leakage"),
        "HIGH_FREQ_NOISE": DefectPrior(0.22, 0.50, "Tape hiss at 7.5 ips; bias noise"),
        "WOW": DefectPrior(0.15, 0.38, "Capstan servo better; but still present"),
        "FLUTTER": DefectPrior(0.18, 0.42, "Flutter from 60 Hz capstan"),
        "DROPOUTS": DefectPrior(0.18, 0.40, "Oxide shedding; dust on head"),
        "BIAS_ERROR": DefectPrior(0.22, 0.48, "Alignment drift over time"),
        "HEAD_WEAR": DefectPrior(0.20, 0.45, "HF loss from worn playback head"),
    },
    (1970, "reel_tape"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.18, 0.45, "Ampex 456 / BASF LH900 hiss"),
        "PRINT_THROUGH": DefectPrior(0.25, 0.58, "Tape pack pre/post echo"),
        "MODULATION_NOISE": DefectPrior(0.20, 0.48, "Oxide grain modulation"),
        "DROPOUTS": DefectPrior(0.15, 0.38, "Oxide shedding from Ampex 456"),
        "HEAD_WEAR": DefectPrior(0.18, 0.42, "HF loss from ferrite head wear"),
        "FLUTTER": DefectPrior(0.12, 0.30, "Servo flutter at reel change"),
        "STICKY_SHED_RESIDUE": DefectPrior(0.30, 0.62, "Ampex 456 binder hydrolysis → sticky shed"),
        "AZIMUTH_ERROR": DefectPrior(0.20, 0.45, "Head alignment drift → HF comb-filter"),
        "BIAS_ERROR": DefectPrior(0.15, 0.35, "Frequency response coloration"),
    },
    (1980, "reel_tape"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.15, 0.40, "Dolby NR-A reduces hiss; still present"),
        "DOLBY_NR_MISMATCH": DefectPrior(0.25, 0.55, "Dolby-A decoding mismatch → pumping"),
        "STICKY_SHED_RESIDUE": DefectPrior(0.28, 0.60, "Ampex 456/BASF binder problem peak era"),
        "MODULATION_NOISE": DefectPrior(0.18, 0.42, "Oxide modulation at high signal levels"),
        "AZIMUTH_ERROR": DefectPrior(0.18, 0.40, "Multi-track head block warping"),
        "HEAD_WEAR": DefectPrior(0.15, 0.38, "HF roll from head gap wear"),
        "PRINT_THROUGH": DefectPrior(0.20, 0.48, "Standard tape pack pre-echo"),
        "GENERATION_LOSS": DefectPrior(0.25, 0.55, "Bounce-recording generation loss"),
        "DROPOUTS": DefectPrior(0.12, 0.30, "Mature formulations better"),
    },
    # ===================================================================
    # CASSETTE TAPE (1970–2010)
    # IEC 60386: wow ≤ 0.2% WRMS; BW 40–14 kHz (Type I) to 40–18 kHz (Type IV)
    # ===================================================================
    (1970, "tape"): {  # "tape" = cassette in canonical keys
        "HIGH_FREQ_NOISE": DefectPrior(0.42, 0.78, "Type I oxide hiss; Dolby B partial"),
        "WOW": DefectPrior(0.35, 0.70, "Capstan/pinch-roller variations in cassette"),
        "FLUTTER": DefectPrior(0.30, 0.65, "Cassette mechanism flutter 2–4 kHz"),
        "BANDWIDTH_LOSS": DefectPrior(0.35, 0.68, "Type I BW ≤ 12 kHz without Dolby"),
        "DOLBY_NR_MISMATCH": DefectPrior(0.22, 0.48, "Dolby B playback without encode"),
        "DROPOUTS": DefectPrior(0.25, 0.52, "Tape oxide shedding"),
        "MODULATION_NOISE": DefectPrior(0.28, 0.58, "Oxide grain noise"),
        "BIAS_ERROR": DefectPrior(0.25, 0.55, "Type I vs Type II bias mismatch"),
    },
    (1980, "tape"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.35, 0.70, "Even with Dolby B: residual hiss"),
        "WOW": DefectPrior(0.28, 0.62, "Cassette mechanism; ±0.15% WRMS"),
        "FLUTTER": DefectPrior(0.25, 0.58, "Flutter from pad pressure variations"),
        "DOLBY_NR_MISMATCH": DefectPrior(0.28, 0.60, "Dolby B/C decoding errors common"),
        "BANDWIDTH_LOSS": DefectPrior(0.22, 0.48, "Type II / Cr02 ≤ 15 kHz; Type IV better"),
        "MODULATION_NOISE": DefectPrior(0.22, 0.50, "Chromium dioxide oxide noise"),
        "BIAS_ERROR": DefectPrior(0.20, 0.45, "Type mismatch on dual-deck recorders"),
        "DROPOUTS": DefectPrior(0.18, 0.40, "Oxide drop-out from head contact"),
    },
    (1990, "tape"): {
        "HIGH_FREQ_NOISE": DefectPrior(0.28, 0.62, "Dolby C or S standard but not universal"),
        "WOW": DefectPrior(0.20, 0.50, "Consumer mechanism WOW ±0.1% WRMS"),
        "DOLBY_NR_MISMATCH": DefectPrior(0.30, 0.65, "Dolby NR mismatch peak era (S vs C)"),
        "FLUTTER": DefectPrior(0.18, 0.42, "Flutter from aging mechanisms"),
        "MODULATION_NOISE": DefectPrior(0.18, 0.40, "Noise-modulated by signal"),
        "DROPOUTS": DefectPrior(0.15, 0.35, "Aging oxide shedding"),
        "GENERATION_LOSS": DefectPrior(0.25, 0.55, "Home dubbing → generation loss"),
    },
    # ===================================================================
    # CD / DIGITAL (1982–present)
    # SNR ≈ 96 dB; BW 20–20 kHz; THD < 0.01% (IEC 60908)
    # ===================================================================
    (1980, "cd"): {
        "DIGITAL_ARTIFACTS": DefectPrior(0.25, 0.55, "Early CD pressing errors; 16-bit noise"),
        "QUANTIZATION_NOISE": DefectPrior(0.18, 0.42, "16-bit word length dithering issues"),
        "JITTER_ARTIFACTS": DefectPrior(0.20, 0.48, "Early DAC jitter; clock inaccuracies"),
        "BANDWIDTH_LOSS": DefectPrior(0.10, 0.25, "Some early CDs: anti-aliasing filter ringing"),
        "ALIASING": DefectPrior(0.15, 0.35, "Brick-wall filter artefacts at 20 kHz"),
    },
    (1990, "cd"): {
        "DIGITAL_ARTIFACTS": DefectPrior(0.15, 0.38, "CD error correction; read errors"),
        "QUANTIZATION_NOISE": DefectPrior(0.10, 0.28, "16-bit noise floor at −96 dBFS"),
        "DYNAMIC_COMPRESSION_EXCESS": DefectPrior(0.40, 0.72, "1990s loudness war begins"),
        "JITTER_ARTIFACTS": DefectPrior(0.12, 0.30, "Transport jitter audible in some systems"),
        "ALIASING": DefectPrior(0.08, 0.20, "Better anti-aliasing filters"),
        "CLIPPING": DefectPrior(0.18, 0.42, "Digital clipping from hot masters"),
    },
    (2000, "cd"): {
        "DYNAMIC_COMPRESSION_EXCESS": DefectPrior(0.55, 0.82, "Loudness war peak era"),
        "CLIPPING": DefectPrior(0.35, 0.68, "Digital clipping on hyper-compressed masters"),
        "DIGITAL_ARTIFACTS": DefectPrior(0.10, 0.25, "Mature format; pressing errors rare"),
        "QUANTIZATION_NOISE": DefectPrior(0.08, 0.20, "16-bit; visible only on quiet passages"),
    },
    (2010, "cd"): {
        "DYNAMIC_COMPRESSION_EXCESS": DefectPrior(0.45, 0.75, "Still common; streaming masters diverge"),
        "CLIPPING": DefectPrior(0.25, 0.55, "Digital clipping from brick-wall limiter"),
        "DIGITAL_ARTIFACTS": DefectPrior(0.08, 0.18, "CD format mature"),
    },
    # ===================================================================
    # MP3 (1995–present)
    # ===================================================================
    (2000, "mp3_low"): {
        "COMPRESSION_ARTIFACTS": DefectPrior(0.70, 0.92, "Pre-echo, ringing, musical noise (< 128 kbps)"),
        "BANDWIDTH_LOSS": DefectPrior(0.60, 0.88, "Lowpass cutoff 16–18 kHz at 128 kbps"),
        "PRE_ECHO": DefectPrior(0.55, 0.82, "MDCT pre-echo on transients"),
        "TRANSIENT_SMEARING": DefectPrior(0.50, 0.78, "Temporal masking failure"),
        "HIGH_FREQ_NOISE": DefectPrior(0.40, 0.70, "Quantization noise at HF"),
        "ALIASING": DefectPrior(0.25, 0.52, "Subband aliasing at low bitrate"),
        "STEREO_IMBALANCE": DefectPrior(0.15, 0.35, "MS stereo artefacts"),
    },
    (2000, "mp3_high"): {
        "COMPRESSION_ARTIFACTS": DefectPrior(0.25, 0.52, "Subtle artefacts at 256–320 kbps"),
        "BANDWIDTH_LOSS": DefectPrior(0.10, 0.28, "Minimal HF rolloff"),
        "PRE_ECHO": DefectPrior(0.15, 0.38, "Rare but audible on extreme transients"),
        "HIGH_FREQ_NOISE": DefectPrior(0.12, 0.30, "Near-Nyquist quantization noise"),
    },
    (2010, "mp3_low"): {
        "COMPRESSION_ARTIFACTS": DefectPrior(0.65, 0.88, "Low-bitrate streaming (96–128 kbps)"),
        "BANDWIDTH_LOSS": DefectPrior(0.55, 0.85, "HF cutoff at 16 kHz"),
        "PRE_ECHO": DefectPrior(0.50, 0.78, "Attack smearing"),
        "TRANSIENT_SMEARING": DefectPrior(0.45, 0.72, "Temporal masking failures"),
    },
    # ===================================================================
    # DAT (Digital Audio Tape) (1987–2005)
    # ===================================================================
    (1990, "dat"): {
        "DIGITAL_ARTIFACTS": DefectPrior(0.20, 0.48, "Read errors → dropouts or clicks"),
        "DROPOUTS": DefectPrior(0.25, 0.55, "Helical scan head clogs"),
        "JITTER_ARTIFACTS": DefectPrior(0.15, 0.38, "Clock jitter from consumer transport"),
        "QUANTIZATION_NOISE": DefectPrior(0.10, 0.25, "16-bit word length"),
        "TRANSPORT_BUMP": DefectPrior(0.18, 0.42, "Transport mechanism vibration"),
    },
    (2000, "dat"): {
        "DIGITAL_ARTIFACTS": DefectPrior(0.30, 0.65, "Aging DAT tape: dropout rate increases"),
        "DROPOUTS": DefectPrior(0.35, 0.70, "DAT head clog most common failure"),
        "JITTER_ARTIFACTS": DefectPrior(0.18, 0.42, "Clock jitter"),
        "QUANTIZATION_NOISE": DefectPrior(0.10, 0.25, "16-bit noise"),
    },
}

# ---------------------------------------------------------------------------
# Fallback table: material-only (era-agnostic)
# Used when era_decade is None or no specific era×material entry exists.
# ---------------------------------------------------------------------------

_MATERIAL_ONLY_PRIORS: dict[str, dict[str, DefectPrior]] = {
    "vinyl": {
        "CRACKLE": DefectPrior(0.28, 0.65, "Standard vinyl surface noise"),
        "CLICKS": DefectPrior(0.22, 0.55, "Stylus impact; dust"),
        "LOW_FREQ_RUMBLE": DefectPrior(0.28, 0.58, "Bearing noise 20–60 Hz"),
        "INNER_GROOVE_DISTORTION": DefectPrior(0.30, 0.58, "Cutter geometry"),
        "HIGH_FREQ_NOISE": DefectPrior(0.20, 0.48, "Vinyl surface noise"),
        "WOW": DefectPrior(0.15, 0.35, "Turntable WOW"),
        "GROOVE_ECHO": DefectPrior(0.18, 0.40, "Adjacent groove pre-echo"),
    },
    "shellac": {
        "HIGH_FREQ_NOISE": DefectPrior(0.68, 0.92, "Shellac grain noise"),
        "CRACKLE": DefectPrior(0.62, 0.88, "Surface degradation"),
        "CLICKS": DefectPrior(0.50, 0.80, "Needle damage"),
        "BANDWIDTH_LOSS": DefectPrior(0.72, 0.97, "Physical BW ≤ 8 kHz"),
        "WOW": DefectPrior(0.38, 0.68, "78 RPM eccentricity"),
    },
    "wax_cylinder": {
        "HIGH_FREQ_NOISE": DefectPrior(0.75, 0.95, "Surface crumbling"),
        "CRACKLE": DefectPrior(0.72, 0.93, "Wax micro-fractures"),
        "BANDWIDTH_LOSS": DefectPrior(0.82, 0.98, "Physical BW ≤ 4 kHz"),
        "PITCH_DRIFT": DefectPrior(0.62, 0.87, "Spring-wound motor"),
    },
    "reel_tape": {
        "HIGH_FREQ_NOISE": DefectPrior(0.20, 0.50, "Tape hiss (speed-dependent)"),
        "PRINT_THROUGH": DefectPrior(0.28, 0.60, "Tape pack echo"),
        "STICKY_SHED_RESIDUE": DefectPrior(0.25, 0.55, "Binder hydrolysis"),
        "AZIMUTH_ERROR": DefectPrior(0.20, 0.45, "Head alignment drift"),
        "FLUTTER": DefectPrior(0.15, 0.38, "Reel-servo flutter"),
    },
    "tape": {  # cassette
        "HIGH_FREQ_NOISE": DefectPrior(0.38, 0.72, "Type I/II oxide hiss"),
        "WOW": DefectPrior(0.30, 0.65, "Capstan/pinch-roller"),
        "DOLBY_NR_MISMATCH": DefectPrior(0.25, 0.55, "Dolby decode mismatch"),
        "FLUTTER": DefectPrior(0.22, 0.50, "Mechanism flutter"),
    },
    "cd": {
        "DYNAMIC_COMPRESSION_EXCESS": DefectPrior(0.35, 0.65, "Loudness war"),
        "CLIPPING": DefectPrior(0.20, 0.48, "Digital clipping from hot masters"),
        "QUANTIZATION_NOISE": DefectPrior(0.10, 0.28, "16-bit word length floor"),
    },
    "mp3_low": {
        "COMPRESSION_ARTIFACTS": DefectPrior(0.65, 0.90, "Psychoacoustic codec artefacts"),
        "BANDWIDTH_LOSS": DefectPrior(0.55, 0.85, "HF cutoff"),
        "PRE_ECHO": DefectPrior(0.48, 0.78, "MDCT pre-echo"),
        "TRANSIENT_SMEARING": DefectPrior(0.42, 0.70, "Temporal smearing"),
    },
    "mp3_high": {
        "COMPRESSION_ARTIFACTS": DefectPrior(0.20, 0.45, "Subtle codec artefacts"),
        "PRE_ECHO": DefectPrior(0.12, 0.32, "Rare on high bitrate"),
        "BANDWIDTH_LOSS": DefectPrior(0.08, 0.22, "Minimal HF rolloff"),
    },
    "dat": {
        "DROPOUTS": DefectPrior(0.30, 0.65, "Helical scan head clogs"),
        "DIGITAL_ARTIFACTS": DefectPrior(0.25, 0.55, "Read errors"),
        "JITTER_ARTIFACTS": DefectPrior(0.15, 0.38, "Transport jitter"),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _nearest_era(era_decade: int, available: list[int]) -> int:
    """Gibt closest available era key to era_decade zurück."""
    return min(available, key=lambda d: abs(d - era_decade))


def get_era_material_defect_priors(
    era_decade: int | None,
    material_type: str,
) -> dict[str, tuple[float, float]]:
    """Gibt defect severity priors for a given era × material combination zurück.

    The returned dict maps each known defect name to a
    ``(severity_prior, probability)`` tuple suitable for direct use as a
    calibration signal to the DefectScanner.

    Args:
        era_decade:    Recording decade (e.g. 1950, 1970). If ``None``, uses
                       material-only priors.
        material_type: Canonical material key (vinyl, shellac, tape, reel_tape,
                       wax_cylinder, cd, mp3_low, mp3_high, dat, …).

    Returns:
        Dict mapping defect name strings → (severity_prior, probability).
    """
    mat_key = str(material_type or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    # Normalise cassette
    if mat_key == "cassette":
        mat_key = "tape"

    result: dict[str, tuple[float, float]] = {}

    # 1. Try exact era × material
    if era_decade is not None:
        exact = _ERA_MATERIAL_PRIORS.get((int(era_decade), mat_key))
        if exact is not None:
            result = {k: (v.severity_prior, v.probability) for k, v in exact.items()}
            logger.debug(
                "era_defect_patterns: exact match era=%d mat=%s → %d priors",
                era_decade,
                mat_key,
                len(result),
            )
            return result

        # 2. Nearest available era for this material
        available_eras = [e for (e, m) in _ERA_MATERIAL_PRIORS if m == mat_key]
        if available_eras:
            nearest = _nearest_era(int(era_decade), available_eras)
            era_priors = _ERA_MATERIAL_PRIORS[(nearest, mat_key)]
            gap = abs(nearest - int(era_decade))
            # Scale probability by distance (priors from 30 years away half weight)
            weight = max(0.3, 1.0 - gap / 60.0)
            result = {k: (v.severity_prior, v.probability * weight) for k, v in era_priors.items()}
            logger.debug(
                "era_defect_patterns: nearest era=%d (gap=%d, weight=%.2f) mat=%s → %d priors",
                nearest,
                gap,
                weight,
                mat_key,
                len(result),
            )
            return result

    # 3. Material-only fallback
    mat_fb = _MATERIAL_ONLY_PRIORS.get(mat_key, {})
    result = {k: (v.severity_prior, v.probability) for k, v in mat_fb.items()}
    logger.debug(
        "era_defect_patterns: material-only Ersatzpfad mat=%s → %d priors",
        mat_key,
        len(result),
    )
    return result


def get_high_probability_defects(
    era_decade: int | None,
    material_type: str,
    probability_threshold: float = 0.60,
) -> list[str]:
    """Gibt defect names with probability ≥ threshold for quick prioritisation zurück.

    Args:
        era_decade:            Recording decade.
        material_type:         Canonical material key.
        probability_threshold: Minimum probability to include [0..1].

    Returns:
        Sorted list of defect name strings.
    """
    priors = get_era_material_defect_priors(era_decade, material_type)
    return sorted(name for name, (_, prob) in priors.items() if prob >= probability_threshold)


def get_severity_adjustment_factors(
    era_decade: int | None,
    material_type: str,
) -> dict[str, float]:
    """Gibt per-defect severity multipliers for DefectScanner calibration zurück.

    A severity multiplier > 1.0 means the DefectScanner should lower its
    detection threshold (more sensitive) — because this defect is historically
    common and may be under-reported on this material.

    Returns:
        Dict mapping defect name → severity_multiplier [0.5..2.0].
    """
    priors = get_era_material_defect_priors(era_decade, material_type)
    factors: dict[str, float] = {}
    for name, (severity, prob) in priors.items():
        # High-probability, high-severity defects → lower threshold (multiplier > 1)
        # Rare defects → no boosting (multiplier ≤ 1)
        if prob >= 0.70:
            # Multiply scanner sensitivity for this defect type
            factors[name] = 1.0 + 0.5 * prob + 0.3 * severity
        elif prob >= 0.40:
            factors[name] = 1.0 + 0.2 * prob
    return factors


__all__ = [
    "DefectPrior",
    "get_era_material_defect_priors",
    "get_high_probability_defects",
    "get_severity_adjustment_factors",
]
