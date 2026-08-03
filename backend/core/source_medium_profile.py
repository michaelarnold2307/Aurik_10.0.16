"""§v10.705 SourceMediumProfile — Zentrale Registratur physikalischer Limiten.

Jedes Trägermedium hat inhärente physikalische Grenzen, die kein Algorithmus
überschreiten kann, ohne zu halluzinieren. Diese Datei definiert die EINE Quelle
der Wahrheit für alle medium-abhängigen Caps.

Verwendung:
    from backend.core.source_medium_profile import get_medium_profile
    profile = get_medium_profile("cassette")
    print(profile.max_bandwidth_hz)   # 10000
    print(profile.max_dynamic_db)     # 50

Integration:
    - ProcessingCeilingGuard liest Caps und erzwingt sie in FlashSR/Harmonic/BandGap
    - CodecArtifactAwareness nutzt terminal_codec für MP3/AAC-Erkennung
    - De-Esser nutzt soft_saturation_limit für Skip-Entscheidung
    - Hiss-Reduktion nutzt noise_floor_db für depth-adaptive Stärke
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Physikalische Limiten pro Trägermedium ────────────────────────────────
# Quellen: IEC 60094 (Tape), ITU-R BS.468 (Noise), AES E-Library (Vinyl),
#          Brandenburg 1999 (MP3), eigene Messungen an 200+ Tonträgern.


@dataclass(frozen=True)
class SourceMediumProfile:
    """Physikalische Limiten eines Trägermediums.

    Alle Werte sind maximale, physikalisch erreichbare Werte des MEDIUMS —
    nicht der darauf gespeicherten Aufnahme. Die tatsächliche Aufnahme kann
    schlechter sein (z.B. 8 kHz Bandbreite auf Kassette), aber NIEMALS besser.
    """

    # ── Identität ──────────────────────────────────────────────────────
    medium_key: str  # z.B. "cassette", "vinyl", "mp3_low"
    display_name: str  # z.B. "Compact Cassette"

    # ── Frequenzbereich ────────────────────────────────────────────────
    max_bandwidth_hz: float  # Phys. maximale Bandbreite (-3 dB)

    # ── Dynamik ────────────────────────────────────────────────────────
    max_dynamic_db: float  # Maximaler phys. Dynamikumfang (Peak zu Noise-Floor)
    noise_floor_dbfs: float  # Phys. Noise-Floor (A-bewertet)

    # ── Optionale Felder (mit Defaults) ─────────────────────────────────
    native_nyquist_hz: float | None = None  # Native Abtastrate/2 (None=analog)
    bass_extension_hz: float = 30.0  # -3 dB untere Grenze
    saturation_headroom_db: float = 3.0  # Headroom über Nennpegel vor Sättigung

    # ── Verarbeitungshinweise ──────────────────────────────────────────
    is_compressed: bool = False  # Bereits dynamikkomprimiert (Cassette, MP3)
    has_soft_saturation: bool = False  # Weiche Sättigung (Tape) vs harte (Digital)
    is_lossy_codec: bool = False  # Terminal-Codec ist verlustbehaftet
    codec_family: str = "none"  # "mp3", "aac", "opus", "none"

    # ── Synthese-Caps ──────────────────────────────────────────────────
    synthesis_ceiling_hz: float = 0.0  # Maximale Frequenz für Synthese (0=aus BW)
    harmonic_max_order: int = 8  # Maximale harmonische Ordnung
    deesser_skip_saturation_conf: float = 0.5  # De-Esser-Skip bei soft_sat > conf
    hiss_reduction_max_strength: float = 0.40  # Max. Stärke Hiss-Reduktion

    def __post_init__(self):
        """Leite Synthese-Ceiling aus Bandbreite ab wenn nicht explizit gesetzt."""
        if self.synthesis_ceiling_hz <= 0:
            # Synthese nur bis 1.2× native Bandbreite (Sicherheitsmarge)
            _ceiling = self.max_bandwidth_hz * 1.2
            # Digital natives: Nyquist-Limit wenn gesetzt
            if self.native_nyquist_hz is not None:
                _ceiling = min(_ceiling, self.native_nyquist_hz * 0.95)
            object.__setattr__(self, "synthesis_ceiling_hz", round(_ceiling))


# ═══════════════════════════════════════════════════════════════════════════════
# Registratur
# ═══════════════════════════════════════════════════════════════════════════════

_MEDIUM_PROFILES: dict[str, SourceMediumProfile] = {
    # ── Analog-Band ─────────────────────────────────────────────────────
    "reel_tape": SourceMediumProfile(
        medium_key="reel_tape",
        display_name="Professionelles Spulentonband",
        max_bandwidth_hz=20000,  # 15 ips, CCIR EQ
        max_dynamic_db=65,
        noise_floor_dbfs=-58.0,
        saturation_headroom_db=6.0,
        is_compressed=False,
        has_soft_saturation=True,
        harmonic_max_order=12,
        hiss_reduction_max_strength=0.55,
    ),
    "cassette": SourceMediumProfile(
        medium_key="cassette",
        display_name="Compact Cassette",
        max_bandwidth_hz=10000,  # Typ I Band, -3 dB bei ~10 kHz
        max_dynamic_db=50,  # Ohne Dolby
        noise_floor_dbfs=-42.0,
        saturation_headroom_db=2.0,
        is_compressed=True,  # Bandkompression inhärent
        has_soft_saturation=True,
        harmonic_max_order=5,  # Obertöne über 5× sind phys. nicht auf Kassette
        deesser_skip_saturation_conf=0.45,  # Früher skip bei Kassette
        hiss_reduction_max_strength=0.30,  # Sanfter — konstantes Rauschen
    ),
    "tape": SourceMediumProfile(
        medium_key="tape",
        display_name="Tonband (generisch)",
        max_bandwidth_hz=14000,
        max_dynamic_db=55,
        noise_floor_dbfs=-48.0,
        has_soft_saturation=True,
        harmonic_max_order=8,
        hiss_reduction_max_strength=0.45,
    ),
    # ── Schallplatte ────────────────────────────────────────────────────
    "vinyl": SourceMediumProfile(
        medium_key="vinyl",
        display_name="Vinyl-Schallplatte",
        max_bandwidth_hz=18000,  # RIAA, äußerer Radius
        max_dynamic_db=60,
        noise_floor_dbfs=-52.0,
        has_soft_saturation=False,
        harmonic_max_order=10,
        hiss_reduction_max_strength=0.50,
    ),
    "shellac": SourceMediumProfile(
        medium_key="shellac",
        display_name="Schellack-Platte (78rpm)",
        max_bandwidth_hz=8000,
        max_dynamic_db=35,
        noise_floor_dbfs=-30.0,
        harmonic_max_order=3,
        deesser_skip_saturation_conf=0.35,
        hiss_reduction_max_strength=0.20,
    ),
    "lacquer_disc": SourceMediumProfile(
        medium_key="lacquer_disc",
        display_name="Acetat-Lackfolie",
        max_bandwidth_hz=12000,
        max_dynamic_db=50,
        noise_floor_dbfs=-42.0,
        harmonic_max_order=6,
        hiss_reduction_max_strength=0.35,
    ),
    # ── Digital ─────────────────────────────────────────────────────────
    "cd_digital": SourceMediumProfile(
        medium_key="cd_digital",
        display_name="CD-DA (Red Book)",
        max_bandwidth_hz=20000,
        native_nyquist_hz=22050,
        max_dynamic_db=96,
        noise_floor_dbfs=-90.0,
        harmonic_max_order=16,
        hiss_reduction_max_strength=0.70,
    ),
    "dat": SourceMediumProfile(
        medium_key="dat",
        display_name="Digital Audio Tape",
        max_bandwidth_hz=20000,
        native_nyquist_hz=22050,
        max_dynamic_db=90,
        noise_floor_dbfs=-85.0,
        harmonic_max_order=14,
        hiss_reduction_max_strength=0.65,
    ),
    "minidisc": SourceMediumProfile(
        medium_key="minidisc",
        display_name="MiniDisc (ATRAC)",
        max_bandwidth_hz=18000,
        native_nyquist_hz=22050,
        max_dynamic_db=80,
        noise_floor_dbfs=-75.0,
        is_lossy_codec=True,
        codec_family="atrac",
        harmonic_max_order=10,
    ),
    # ── Verlustbehaftete Codecs ─────────────────────────────────────────
    "mp3_low": SourceMediumProfile(
        medium_key="mp3_low",
        display_name="MP3 (≤128 kbps)",
        max_bandwidth_hz=15500,  # Lowpass-Filter des Encoders
        native_nyquist_hz=22050,
        max_dynamic_db=70,  # Durch Quantisierung limitiert
        noise_floor_dbfs=-60.0,
        is_lossy_codec=True,
        codec_family="mp3",
        harmonic_max_order=4,  # MDCT-Artefakte zerstören feine Harmonische
        deesser_skip_saturation_conf=0.40,
        hiss_reduction_max_strength=0.35,
    ),
    "mp3_high": SourceMediumProfile(
        medium_key="mp3_high",
        display_name="MP3 (≥192 kbps)",
        max_bandwidth_hz=18000,
        native_nyquist_hz=22050,
        max_dynamic_db=80,
        noise_floor_dbfs=-70.0,
        is_lossy_codec=True,
        codec_family="mp3",
        harmonic_max_order=8,
    ),
    "aac": SourceMediumProfile(
        medium_key="aac",
        display_name="AAC (MP4 Audio)",
        max_bandwidth_hz=19000,
        native_nyquist_hz=22050,
        max_dynamic_db=85,
        noise_floor_dbfs=-75.0,
        is_lossy_codec=True,
        codec_family="aac",
        harmonic_max_order=10,
    ),
    # ── Historisch / Exotisch ───────────────────────────────────────────
    "wax_cylinder": SourceMediumProfile(
        medium_key="wax_cylinder",
        display_name="Wachszylinder",
        max_bandwidth_hz=4000,
        max_dynamic_db=25,
        noise_floor_dbfs=-20.0,
        harmonic_max_order=2,
        deesser_skip_saturation_conf=0.25,
        hiss_reduction_max_strength=0.15,
    ),
    "wire_recording": SourceMediumProfile(
        medium_key="wire_recording",
        display_name="Drahtaufnahme",
        max_bandwidth_hz=5000,
        max_dynamic_db=30,
        noise_floor_dbfs=-25.0,
        harmonic_max_order=2,
        deesser_skip_saturation_conf=0.30,
        hiss_reduction_max_strength=0.18,
    ),
    # ── Fallback ────────────────────────────────────────────────────────
    "unknown": SourceMediumProfile(
        medium_key="unknown",
        display_name="Unbekanntes Medium",
        max_bandwidth_hz=20000,  # Konservativ: moderne Annahme
        max_dynamic_db=80,
        noise_floor_dbfs=-70.0,
    ),
    "streaming": SourceMediumProfile(
        medium_key="streaming",
        display_name="Streaming (generisch)",
        max_bandwidth_hz=18000,
        native_nyquist_hz=22050,
        max_dynamic_db=85,
        noise_floor_dbfs=-75.0,
        is_lossy_codec=True,
        codec_family="opus",
        harmonic_max_order=8,
    ),
}


def get_medium_profile(medium_key: str) -> SourceMediumProfile:
    """Liefert das SourceMediumProfile für einen Medium-Key.

    Args:
        medium_key: z.B. "cassette", "vinyl", "mp3_low". Case-insensitive,
                    normalisiert Aliase (kassette→cassette, lp→vinyl).

    Returns:
        SourceMediumProfile für das angeforderte Medium, oder "unknown" als Fallback.
    """
    _key = str(medium_key).lower().strip()
    # Aliase
    _ALIASES = {
        "kassette": "cassette",
        "compact_cassette": "cassette",
        "lp": "vinyl",
        "vinyl_33": "vinyl",
        "vinyl_45": "vinyl",
        "vinyl_78": "shellac",
        "reel_to_reel": "reel_tape",
        "cd": "cd_digital",
        "compact_disc": "cd_digital",
        "mp3": "mp3_high",
        "mp3_128": "mp3_low",
        "mp3_192": "mp3_high",
        "mp3_320": "mp3_high",
    }
    _key = _ALIASES.get(_key, _key)
    return _MEDIUM_PROFILES.get(_key, _MEDIUM_PROFILES["unknown"])


def get_medium_profile_for_depth(
    material_type: str,
    terminal_codec: str | None = None,
) -> SourceMediumProfile:
    """Bestimmt das restriktivste Profil aus Material + Terminal-Codec.

    Wenn die Kette z.B. "vinyl → cassette → mp3_low" ist, dominiert mp3_low
    (das schwächste Glied). Wenn kein Terminal-Codec bekannt ist, wird das
    Material-Profil verwendet.

    Args:
        material_type: Vom MediumDetector erkanntes Trägermaterial
        terminal_codec: Letzter Codec in der Transfer-Kette (z.B. "mp3_low")

    Returns:
        Restriktivstes SourceMediumProfile der Kette.
    """
    _mat_profile = get_medium_profile(material_type)
    if terminal_codec:
        _codec_profile = get_medium_profile(terminal_codec)
        # Verwende das Profil mit der NIEDRIGEREN Bandbreite (restriktiver)
        if _codec_profile.max_bandwidth_hz < _mat_profile.max_bandwidth_hz:
            return _codec_profile
    return _mat_profile

    @staticmethod
    def get_bw_ceiling_hz(medium_key: str) -> float | None:
        """§v10.705: Liefert die physikalische Bandbreiten-Obergrenze eines Mediums.

        Wird von phase_23, phase_56 und anderen Synthese-Phasen verwendet,
        um Halluzination oberhalb der physikalischen Medium-Grenze zu verhindern.
        """
        _profile = get_medium_profile(medium_key)
        return _profile.synthesis_ceiling_hz if _profile else None


# ── Convenience ──────────────────────────────────────────────────────────────


def get_synthesis_ceiling_hz(medium_key: str, terminal_codec: str | None = None) -> float:
    """Maximale Synthese-Frequenz für ein Medium."""
    _profile = get_medium_profile_for_depth(medium_key, terminal_codec)
    return _profile.synthesis_ceiling_hz


def should_skip_deesser_for_saturation(medium_key: str, soft_saturation_confidence: float) -> bool:
    """Soll der De-Esser wegen Soft-Saturation übersprungen werden?"""
    _profile = get_medium_profile(medium_key)
    return soft_saturation_confidence > _profile.deesser_skip_saturation_conf


def get_hiss_reduction_max_strength(medium_key: str, chain_depth: int = 1) -> float:
    """Maximale Stärke der Hiss-Reduktion, depth-adaptiv."""
    _profile = get_medium_profile(medium_key)
    _base = _profile.hiss_reduction_max_strength
    # Tiefe Ketten (depth≥4): 25% Reduktion, depth≥5: 40% Reduktion
    if chain_depth >= 5:
        return _base * 0.60
    elif chain_depth >= 4:
        return _base * 0.75
    return _base
