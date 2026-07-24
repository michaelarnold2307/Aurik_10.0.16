"""§v10.15 Phase Display Formatter
================================
Maps internal phase IDs to rich, human-readable display names with emojis.
Matching the log-level quality for user-facing UI.

Used by: UV3 progress_callback, Watchdog dialog, status bar.
"""

from __future__ import annotations

# ── Phase ID → Rich Display Name ──────────────────────────────────
PHASE_DISPLAY: dict[str, str] = {
    # Core restoration
    "phase_01_click_removal": "🔍🩹 Klick-Erkennung",
    "phase_02_hum_removal": "〰️ Brumm-Entfernung",
    "phase_03_denoise": "🧹🌊 Entrauschung",
    "phase_04_eq_correction": "🎛️📐 EQ-Korrektur",
    "phase_05_rumble_filter": "📉 Rumpelfilter",
    "phase_06_frequency_restoration": "🎛️🔧 Frequenz-Wiederherstellung",
    "phase_07_harmonic_restoration": "🎵✨ Harmonische Restauration",
    "phase_08_transient_preservation": "⚡ Transienten-Erhalt",
    "phase_09_crackle_removal": "🧹⚡ Knistern-Entfernung",
    # Transport / time
    "phase_12_wow_flutter_fix": "🎢 Gleichlauf-Korrektur",
    "phase_25_azimuth_correction": "📐 Azimut-Korrektur",
    "phase_31_speed_pitch_correction": "🎯 Geschwindigkeits-Korrektur",
    # Stereo / spatial
    "phase_13_stereo_enhancement": "🔊✨ Stereo-Anreicherung",
    "phase_14_phase_correction": "🔄🔊 Phasenlage-Korrektur",
    "phase_15_stereo_balance": "⚖️ Stereo-Balance",
    "phase_34_mid_side_processing": "🎯 M/S-Prozessor",
    "phase_48_stereo_width_enhancer": "📐 Stereo-Breite",
    "phase_62_crosstalk_cancellation": "🔇 Übersprech-Unterdrückung",
    # Tonal / EQ
    "phase_16_final_eq": "🎛️✨ Abschluss-EQ",
    "phase_17_mastering_polish": "💿✨ Mastering-Politur",
    "phase_18_noise_gate": "🔇 Noise Gate",
    "phase_19_de_esser": "🎤✨ De-Esser",
    "phase_37_bass_enhancement": "🔊 Bass-Anhebung",
    "phase_38_presence_boost": "🎤 Präsenz-Anhebung",
    "phase_39_air_band_enhancement": "💨 Luftband-Anhebung",
    "phase_40_loudness_normalization": "📏 Lautheits-Normalisierung",
    "phase_47_truepeak_limiter": "📏 True-Peak-Limiter",
    # Spectral repair
    "phase_23_spectral_repair": "🔧 Spektrale Reparatur",
    "phase_50_spectral_repair": "🔧 Spektrale Nachreparatur",
    "phase_56_spectral_band_gap_repair": "🔧 Bandlücken-Reparatur",
    # Dropout / gaps
    "phase_24_dropout_repair": "🩹 Aussetzer-Reparatur",
    "phase_27_click_pop_removal": "🎯 Knackser-Entfernung",
    # Noise / surface
    "phase_28_surface_noise_profiling": "📊 Oberflächen-Rauschen",
    "phase_29_tape_hiss_reduction": "📼 Band-Rausch-Unterdrückung",
    "phase_59_modulation_noise_reduction": "📼 Modulations-Rauschen",
    # Dynamics
    "phase_26_dynamic_range_expansion": "📈 Dynamik-Erweiterung",
    "phase_36_transient_shaper": "🔨 Transienten-Shaper",
    "phase_54_transparent_dynamics": "🎚️ Transparente Dynamik",
    # Vocal / production
    "phase_43_ml_deesser": "🎤 ML De-Esser",
    "phase_46_studio_reverb_removal": "🏠 Studio-Hall-Entfernung",
    "phase_49_advanced_dereverb": "🔇 Enthallung",
    "phase_65_vocal_naturalness_restoration": "🌿 Gesangs-Natürlichkeit",
    # Reverb / room
    "phase_20_reverb_reduction": "🏠 Hall-Reduktion",
    # Carrier-specific
    "phase_60_inner_groove_distortion_repair": "💿 Innenrillen-Verzerrung",
    "phase_61_groove_echo_cancellation": "💿 Rillen-Echo",
    "phase_64_tape_splice_repair": "📼 Band-Klebestelle",
    # Advanced / AI
    "phase_41_output_format_optimization": "📦 Ausgabe-Optimierung",
    "phase_53_semantic_audio": "🧠 Semantische Analyse",
    "phase_55_diffusion_inpainting": "🎨 Diffusion-Inpainting",
    # Post-processing
    "phase_42_stem_remix": "🎚️ Stem-Remix",
    "phase_44_loudness_range": "📏 Loudness-Range",
    "phase_45_dc_offset": "🔌 DC-Offset",
    "phase_51_format_conversion": "🔄 Format-Konvertierung",
    "phase_52_metadata_embedding": "📋 Metadaten",
    # §v10.202: Ergänzte fehlende Phasen
    "phase_10_compression": "📉 Kompression",
    "phase_11_limiting": "📏 Limiting",
    "phase_21_exciter": "⚡ Exciter",
    "phase_22_tape_saturation": "📼 Band-Sättigung",
    "phase_30_dc_offset_removal": "🔌 DC-Offset-Entfernung",
    "phase_32_mono_to_stereo": "🔊 Mono→Stereo",
    "phase_33_stereo_width_limiter": "📐 Stereo-Weiten-Limiter",
    "phase_35_multiband_compression": "🎚️ Multiband-Kompression",
    "phase_42_vocal_enhancement": "🎤 Gesangs-Optimierung",
    "phase_46_spatial_enhancement": "🌌 Räumlichkeit",
    "phase_51_drums_enhancement": "🥁 Schlagzeug-Verbesserung",
    "phase_52_piano_restoration": "🎹 Klavier-Restauration",
    "phase_57_print_through_reduction": "📼 Kopiereffekt-Entfernung",
    "phase_58_lyrics_guided_enhancement": "🎤 Textgeführte Optimierung",
    "phase_63_intermodulation_reduction": "📉 Intermodulations-Reduktion",
    "phase_glue_stage": "🔗 Zusammenführung",
    "phase_07_declipper": "🔧 De-Clipping",
}

# ── Carrier chain display names ───────────────────────────────────
CARRIER_NAMES: dict[str, str] = {
    "reel_tape": "📀 Profi-Spulenband",
    "vinyl": "💿 Schallplatte (Vinyl)",
    "shellac": "💿 Schellack-Platte",
    "cassette": "📼 Kassette (Band)",
    "lacquer_disc": "💿 Lackfolie",
    "wire_recording": "🧲 Drahtaufnahme",
    "wax_cylinder": "🧴 Wachszylinder",
    "minidisc": "💽 Minidisc",
    "mp3_low": "💾 MP3 (niedrige Bitrate)",
    "mp3_high": "💾 MP3 (hohe Bitrate)",
    "cd_digital": "💿 CD (Digital)",
    "streaming": "🌐 Streaming",
    "unknown": "❓ Unbekannt",
}

# ── Era display names ─────────────────────────────────────────────
ERA_NAMES: dict[int, str] = {
    1900: "📯 1900er",
    1910: "📯 1910er",
    1920: "🎺 1920er",
    1930: "🎷 1930er",
    1940: "🎙️ 1940er",
    1950: "🎸 1950er",
    1960: "🎵 1960er",
    1970: "🪩 1970er",
    1980: "💿 1980er",
    1990: "💽 1990er",
    2000: "💻 2000er",
    2010: "📱 2010er",
    2020: "🎧 2020er",
}


def get_phase_display(phase_id: str) -> str:
    """Return rich display name for a phase, with emoji prefix."""
    # Try exact match
    if phase_id in PHASE_DISPLAY:
        return PHASE_DISPLAY[phase_id]
    # Try partial match (strip leading/trailing numbers)
    for key, name in PHASE_DISPLAY.items():
        if key.endswith(phase_id) or phase_id.endswith(key.replace("phase_", "")):
            return name
    # Fallback: human-readable from ID
    return phase_id.replace("phase_", "").replace("_", " ").title().lstrip("0123456789 ")


def get_carrier_display(carrier_key: str) -> str:
    """Return rich display name for a carrier material."""
    key = str(carrier_key).lower().replace("-", "_").replace(" ", "_")
    return CARRIER_NAMES.get(key, f"📀 {carrier_key}")


def get_era_display(decade: int) -> str:
    """Return rich display name for an era/decade."""
    decade_norm = (decade // 10) * 10
    return ERA_NAMES.get(decade_norm, f"📅 {decade_norm}er")


def get_phase_icon(phase_id: str) -> str:
    """Return just the emoji icon for a phase."""
    display = get_phase_display(phase_id)
    # Extract the emoji prefix (first 1-2 chars before text)
    for i, ch in enumerate(display):
        if ch.isalpha() or ch == " ":
            return display[:i].strip()
    return ""
