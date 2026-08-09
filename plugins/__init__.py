"""Plugins-Paket — ML- und DSP-Plugin-Registrierung für Aurik 10.0.0."""

# Ermöglicht Plugin-Importe als aurik6.plugins.*

# ---------------------------------------------------------------------------
# SOTA-Plugins (v10.0.0.x) — kein Docker, kein CUDA, CPU-only
# ---------------------------------------------------------------------------

from .apollo_plugin import (
    ApolloPlugin,
    CodecRepairResult,
    get_apollo,
    repair_codec_artifacts,
)
from .miipher_dit_plugin import (
    MiipherDiTPlugin,
    MiipherDiTResult,
    get_miipher_dit,
)
from .bigvgan_v2_plugin import (
    BigVGANv2Plugin,
    VocoderResult,
    get_bigvgan_v2,
    synthesize_audio,
)
from .bs_roformer_plugin import (
    BSRoFormerPlugin,
    StemSeparationResult,
    get_bs_roformer,
    separate_stems,
)
from .cqtdiff_plus_plugin import (
    CQTdiffPlusPlugin,
    InpaintingResult,
    get_cqtdiff_plus,
    inpaint_gap,
)
from .laion_clap_plugin import (
    AudioTaggingResult,
    LAIONCLAPPlugin,
    get_laion_clap,
    tag_audio,
)
from .utmos_plugin import (
    MOSResult,
    UTMOSPlugin,
    estimate_mos,
    get_utmos,
)
from .vocos_plugin import (
    VocosPlugin,
    VocosResult,
    get_vocos_plugin,
    vocode_mel,
)

__all__ = [
    # Apollo — Codec-Artefakt-Entfernung (MP3/AAC/ATRAC)
    "ApolloPlugin",
    "AudioTaggingResult",
    # BS-RoFormer — Stem Separation
    "BSRoFormerPlugin",
    # BigVGAN-v2 — Sekundärer Vocoder (optional, Apache 2.0)
    "BigVGANv2Plugin",
    # CQTdiff+ — Diffusions-Inpainting (Lücken ≥ 50 ms)
    "CQTdiffPlusPlugin",
    "CodecRepairResult",
    "InpaintingResult",
    # LAION-CLAP — Audio-Tagging Instrumente/Genre/Material
    "LAIONCLAPPlugin",
    "MOSResult",
    "StemSeparationResult",
    # UTMOS — MOS-Schätzung ohne Referenz (Musik-orientiert)
    "UTMOSPlugin",
    "VocoderResult",
    # Vocos — Primärer Vocoder (MIT, 8× schneller als BigVGAN-v2 auf CPU)
    "VocosPlugin",
    "VocosResult",
    "estimate_mos",
    "get_apollo",
    "get_bigvgan_v2",
    "get_bs_roformer",
    "get_cqtdiff_plus",
    "get_laion_clap",
    "get_utmos",
    "get_vocos_plugin",
    "inpaint_gap",
    "repair_codec_artifacts",
    "separate_stems",
    "synthesize_audio",
    "tag_audio",
    "vocode_mel",
]
