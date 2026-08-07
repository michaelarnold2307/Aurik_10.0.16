"""§EraTarget EraCarrierTargetModel — Era × Carrier spectral target (v10.0.0).

Provides a physically-grounded spectral target frame per (era_decade, carrier_type).
The key innovation: *noise_texture_preserve_ratio* — how much of the measured
noise is authentic carrier character that MUST NOT be removed.

This directly lifts the Wiener G_floor in phase_03 and phase_29 for historical
material, preventing over-suppression of vinyl crackle texture, tape hiss sheen,
and shellac ambience that are part of the era's sonic identity.

noise_texture_preserve_ratio → nr_g_floor():
  ratio = 0.70 (1935 shellac) → G_floor = max(0.10, 0.35) = 0.35
  ratio = 0.38 (1965 vinyl)  → G_floor = max(0.10, 0.19) = 0.19
  ratio = 0.05 (1990 CD)     → G_floor = 0.10 (no change from base)

Usage in phase_03 G3 OMLSA block:
    _era_target = kwargs.get("_restoration_context", {}).get("era_carrier_target", {})
    _G_floor_g3 = float(_era_target.get("nr_g_floor", 0.10))

Usage in UV3 (after EraClassifier / room-acoustics fingerprint):
    from backend.core.dsp.era_carrier_target import get_era_carrier_target_model
    _era_target = get_era_carrier_target_model().get_target(decade, transfer_chain)
    _restoration_context["era_carrier_target"] = _era_target.to_dict()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EraTarget:
    """Physical spectral target for one era × carrier combination."""

    noise_texture_preserve_ratio: float  # [0.0, 1.0] — authentic noise to keep
    authentic_harmonic_ratio: float  # [0.0, 1.0] — H2/H4 warmth to preserve
    bw_ceiling_hz: float  # authentic BW ceiling (Hz)
    dr_ceiling_db: float  # authentic DR ceiling (dB)

    def nr_g_floor(self, base_g_floor: float = 0.10) -> float:
        """Wiener G_floor adapted for this era/carrier.

        Scales noise_texture_preserve_ratio to a Wiener gain floor:
          G_floor_era = noise_texture_preserve_ratio × 0.5
        Combined with the existing base_g_floor (§2.62 minimum 0.10).
        """
        era_floor = float(self.noise_texture_preserve_ratio * 0.5)
        return float(max(base_g_floor, era_floor))

    def to_dict(self) -> dict:
        """Serialisiert this EraTarget as a dictionary for metadata and phase injection."""
        return {
            "noise_texture_preserve_ratio": self.noise_texture_preserve_ratio,
            "authentic_harmonic_ratio": self.authentic_harmonic_ratio,
            "bw_ceiling_hz": self.bw_ceiling_hz,
            "dr_ceiling_db": self.dr_ceiling_db,
            "nr_g_floor": self.nr_g_floor(),
        }


# ---------------------------------------------------------------------------
# Lookup table  (era_decade, carrier_key) → EraTarget
#
# Sources: physical acoustics, ITU-R BS.1116, NAB/RIAA standards,
# ARSC Technical Committee reports, Mallinson (1987), Copeland (2008).
# ---------------------------------------------------------------------------

_ERA_CARRIER_TABLE: dict[tuple[int, str], EraTarget] = {
    # --- Acoustic recording era 1900-1925 ---
    (1900, "acoustic"): EraTarget(0.92, 0.90, 2500.0, 26.0),
    (1910, "acoustic"): EraTarget(0.90, 0.90, 3000.0, 27.0),
    (1920, "acoustic"): EraTarget(0.88, 0.88, 3800.0, 29.0),
    # --- Early electrical + shellac 1925-1945 ---
    (1925, "shellac"): EraTarget(0.75, 0.85, 7000.0, 34.0),
    (1930, "shellac"): EraTarget(0.72, 0.85, 7500.0, 35.0),
    (1935, "shellac"): EraTarget(0.70, 0.83, 8000.0, 36.0),
    (1940, "shellac"): EraTarget(0.68, 0.80, 8500.0, 37.0),
    # --- Vinyl LP / early analogue tape 1945-1960 ---
    (1945, "vinyl"): EraTarget(0.55, 0.75, 11000.0, 48.0),
    (1950, "vinyl"): EraTarget(0.50, 0.72, 12000.0, 50.0),
    (1955, "vinyl"): EraTarget(0.45, 0.70, 13000.0, 52.0),
    (1945, "tape"): EraTarget(0.45, 0.70, 12000.0, 48.0),
    (1950, "tape"): EraTarget(0.42, 0.68, 13000.0, 50.0),
    # --- Vinyl LP stereo / analogue tape 1960-1975 ---
    (1960, "vinyl"): EraTarget(0.38, 0.65, 15000.0, 60.0),
    (1965, "vinyl"): EraTarget(0.35, 0.62, 16000.0, 62.0),
    (1970, "vinyl"): EraTarget(0.32, 0.60, 16000.0, 64.0),
    (1960, "tape"): EraTarget(0.35, 0.60, 15000.0, 58.0),
    (1965, "tape"): EraTarget(0.32, 0.58, 16000.0, 60.0),
    (1970, "tape"): EraTarget(0.30, 0.55, 16000.0, 62.0),
    # --- Vinyl / tape / Cassette 1975-1985 ---
    (1975, "vinyl"): EraTarget(0.28, 0.55, 16000.0, 65.0),
    (1980, "vinyl"): EraTarget(0.25, 0.50, 17000.0, 66.0),
    (1975, "cassette"): EraTarget(0.40, 0.60, 13000.0, 50.0),
    (1980, "cassette"): EraTarget(0.35, 0.55, 14000.0, 54.0),
    (1975, "tape"): EraTarget(0.28, 0.52, 16000.0, 63.0),
    (1980, "tape"): EraTarget(0.25, 0.48, 17000.0, 65.0),
    # --- CD / DAT / digital 1985-2000 ---
    (1985, "cd"): EraTarget(0.08, 0.30, 20000.0, 90.0),
    (1990, "cd"): EraTarget(0.06, 0.25, 20000.0, 92.0),
    (1995, "cd"): EraTarget(0.05, 0.20, 20000.0, 94.0),
    (1985, "dat"): EraTarget(0.06, 0.25, 20000.0, 90.0),
    (1990, "dat"): EraTarget(0.05, 0.20, 20000.0, 92.0),
    # --- Lossy / streaming 2000+ ---
    (2000, "mp3"): EraTarget(0.03, 0.10, 18000.0, 60.0),
    (2005, "mp3"): EraTarget(0.03, 0.08, 19000.0, 62.0),
    (2010, "digital"): EraTarget(0.02, 0.05, 22000.0, 96.0),
    (2015, "digital"): EraTarget(0.02, 0.04, 22000.0, 96.0),
    (2020, "digital"): EraTarget(0.01, 0.03, 22000.0, 96.0),
}

_FALLBACK_TARGET = EraTarget(0.20, 0.40, 16000.0, 60.0)


# ---------------------------------------------------------------------------
# Carrier normalisation
# ---------------------------------------------------------------------------


def _normalize_carrier(carrier: str) -> str:
    """Normalisiert carrier string to one of the canonical keys in the table."""
    c = str(carrier).lower().replace("-", "_").replace(" ", "_")
    if "acoustic" in c:
        return "acoustic"
    if "shellac" in c or "wax" in c or "cylinder" in c or "78" in c:
        return "shellac"
    if "cassette" in c:
        return "cassette"
    if "vinyl" in c or "lp" in c:
        return "vinyl"
    if "dat" in c:
        return "dat"
    if "cd" in c:
        return "cd"
    if "tape" in c or "reel" in c or ("analog" in c and "digital" not in c):
        return "tape"
    if "mp3" in c or "aac" in c or "lossy" in c or "ogg" in c or "vorbis" in c:
        return "mp3"
    # FLAC, WAV, PCM, digital → modern digital
    return "digital"


def _nearest_decade(era_decade: int, carrier_key: str) -> int:
    """Snap era_decade to the nearest available decade for carrier_key."""
    available = [d for (d, c) in _ERA_CARRIER_TABLE if c == carrier_key]
    if not available:
        return era_decade
    return min(available, key=lambda d: abs(d - era_decade))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EraCarrierTargetModel:
    """Leichtgewichtiges lookup model — no ML, deterministic, stateless."""

    def get_target(
        self,
        era_decade: int | None,
        carrier: str | list | None,
    ) -> EraTarget:
        """Gibt EraTarget for the given era and carrier zurück.

        Args:
            era_decade: Recording decade (e.g. 1935, 1960). None → fallback.
            carrier:    Carrier type string or list of strings (transfer_chain).
                        The first element is used when a list is given.

        Returns:
            EraTarget; on error returns the fallback (vinyl-era defaults).
        """
        try:
            if era_decade is None:
                return _FALLBACK_TARGET
            decade = int(era_decade)

            # Resolve carrier string
            if isinstance(carrier, (list, tuple)):
                carrier_str = str(carrier[0]) if carrier else "digital"
            else:
                carrier_str = str(carrier or "digital")

            carrier_key = _normalize_carrier(carrier_str)
            snapped = _nearest_decade(decade, carrier_key)
            target = _ERA_CARRIER_TABLE.get((snapped, carrier_key))

            if target is None:
                # Round to nearest 5 and retry
                snapped5 = (decade // 5) * 5
                target = _ERA_CARRIER_TABLE.get((snapped5, carrier_key))

            if target is None:
                target = _FALLBACK_TARGET

            logger.debug(
                "EraCarrierTarget: era=%d→%d carrier=%s→%s noise_preserve=%.2f g_floor=%.2f",
                decade,
                snapped,
                carrier_str,
                carrier_key,
                target.noise_texture_preserve_ratio,
                target.nr_g_floor(),
            )
            return target
        except Exception as exc:
            logger.debug("EraCarrierTargetModel.get_target nicht blockierend: %s", exc)
            return _FALLBACK_TARGET


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_model_holder: list[EraCarrierTargetModel | None] = [None]
_model_lock = threading.Lock()


def get_era_carrier_target_model() -> EraCarrierTargetModel:
    """Thread-safe singleton factory."""
    if _model_holder[0] is None:
        with _model_lock:
            if _model_holder[0] is None:
                _model_holder[0] = EraCarrierTargetModel()
    instance = _model_holder[0]
    assert instance is not None
    return instance
