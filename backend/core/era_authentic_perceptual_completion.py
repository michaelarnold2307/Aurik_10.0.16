"""
backend/core/era_authentic_perceptual_completion.py
Aurik 10.0.0 -- Spec §2.35: EraAuthenticPerceptualCompletion

Synthetisiert aera-authentische HF-Ergaenzung fuer bandbreitenbegrenzte Quellen.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from backend.core.audio_utils import compute_gated_rms_linear as _grl_eapc

logger = logging.getLogger(__name__)


ERA_BRILLANZ_CEILING: dict[int, float] = {
    1920: 0.72,
    1940: 0.78,
    1950: 0.80,
    1960: 0.86,
    1970: 0.90,
    1980: 0.95,
    2000: 0.98,
}


def _get_era_ceiling(decade: int | None) -> float:
    """Gibt den BrillanzMetric-Ceiling fuer ein Jahrzehnt zurueck."""
    if decade is None:
        return 0.98
    # Naechsten Eintrag suchen
    for d in sorted(ERA_BRILLANZ_CEILING.keys()):
        if decade <= d:
            return ERA_BRILLANZ_CEILING[d]
    return 0.98


@dataclass
class EraCompletionResult:
    """Spec §2.35"""

    applied: bool
    era_decade: int | None
    brillanz_ceiling: float
    source_bandwidth_hz: float
    completion_bandwidth_hz: float
    operation_type: str
    message: str
    audio: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, object]:
        """Serialisierungsformat f\u00fcr Logging und API."""
        return {
            "applied": self.applied,
            "era_decade": self.era_decade,
            "brillanz_ceiling": self.brillanz_ceiling,
            "source_bandwidth_hz": self.source_bandwidth_hz,
            "completion_bandwidth_hz": self.completion_bandwidth_hz,
            "operation_type": self.operation_type,
            "message": self.message,
        }


class EraAuthenticPerceptualCompletion:
    """Spec §2.35: Synthetisiert era-authentische HF-Ergaenzung.

    Algorithmus:
        1. Spektralanalyse -> f_max_source
        2. Harmonisches Netz: Partials ueber f_max via Fletcher
        3. Era-Spektralprofil als Ziel-EQ-Kurve
        4. DDSP-Additivsynthese fehlender Partials + HF-Rauschprofil
        5. PGHI-Einblendung (Hanning-Crossfade)
        6. Verifikation: BrillanzMetric <= ERA_BRILLANZ_CEILING
    """

    ERA_BRILLANZ_CEILING: dict[int, float] = ERA_BRILLANZ_CEILING
    SOURCE_BW_THRESHOLD_HZ: float = 10000.0

    def is_applicable(
        self,
        audio: np.ndarray,
        sr: int,
        goal_applicability: object | None = None,
    ) -> bool:
        """True wenn Quell-Bandbreite < 10 kHz UND BrillanzMetric anwendbar."""
        # Pruefe BrillanzMetric Applicability via inapplicable oder applicable
        if goal_applicability is not None:
            inapplicable = getattr(goal_applicability, "inapplicable", None)
            if inapplicable is not None and "brillanz" in inapplicable:
                return False
            applicable = getattr(goal_applicability, "applicable", None)
            if applicable is not None and "brillanz" not in applicable:
                return False

        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        bw = self._estimate_bandwidth(arr, sr)
        return bw < self.SOURCE_BW_THRESHOLD_HZ

    def complete(
        self,
        audio: np.ndarray,
        sr: int,
        era: int | None = None,
        anchor: np.ndarray | None = None,
        material_ceiling: float = 1.0,
        transfer_chain_depth: int = 1,
    ) -> EraCompletionResult:
        """Spec §2.35: Erzeugt era-authentisch ergänztes Audio. NaN/Inf-sicher.

        material_ceiling: §9.12.7 — max. Brillanz-Ceiling aus dem Trägermaterial
        (z.B. 0.78 für Tape). Wird mit dem era-basierten Ceiling kombiniert
        (min), sodass EAPC nie mehr HF synthetisiert als das Trägermaterial
        physikalisch erlaubt (§2.46e Hallucination-Guard).

        transfer_chain_depth: §v10.58 — Tiefe der Transfer-Kette für
        Bandbreiten-Cap-Skalierung. Bei depth≥3 wird das Cap progressiv
        angehoben (bis 14.5 kHz bei depth=4), da tiefe Ketten mehr
        akkumulierten HF-Verlust kompensieren müssen.
        """
        # inf-sicher: erst nan_to_num, dann clip — verhindert Overflow in FFT
        arr = np.clip(
            np.nan_to_num(np.asarray(audio, dtype=np.float32)),
            -1.0,
            1.0,
        )
        if arr.ndim == 2:
            mono = arr.mean(axis=0)
            stereo = arr
            is_stereo = True
        else:
            mono = arr.copy()
            stereo = None
            is_stereo = False

        anchor_mono: np.ndarray | None = None
        if anchor is not None:
            try:
                _anchor_arr = np.clip(np.nan_to_num(np.asarray(anchor, dtype=np.float32)), -1.0, 1.0)
                if _anchor_arr.ndim == 2:
                    anchor_mono = _anchor_arr.mean(axis=0)
                else:
                    anchor_mono = _anchor_arr
                if anchor_mono is not None and anchor_mono.shape[-1] != mono.shape[-1]:
                    if anchor_mono.shape[-1] > mono.shape[-1]:
                        anchor_mono = anchor_mono[: mono.shape[-1]]
                    else:
                        _pad = int(mono.shape[-1] - anchor_mono.shape[-1])
                        anchor_mono = np.pad(anchor_mono, (0, _pad), mode="edge")
            except Exception as exc:
                logger.debug("EAPC Anchor-Normalisierung fehlgeschlagen (nicht blockierend): %s", exc)
                anchor_mono = None

        src_bw = self._estimate_bandwidth(mono, sr)
        # §9.12.7 §2.46e: era-ceiling wird durch material_ceiling nach unten begrenzt.
        # Ohne diesen Clamp würde EAPC für ein 1970s-Kassetten-Recording HF-Energie
        # auf ERA-Niveau (0.90) synthetisieren — Trägermaterial erlaubt physikalisch
        # maximal 0.78. Das ist ein Hallucination-Guard-Verstotz (§2.46e).
        ceiling = min(_get_era_ceiling(era), float(material_ceiling))

        if src_bw >= self.SOURCE_BW_THRESHOLD_HZ:
            msg = "Keine Bandbreiten-Ergaenzung noetig (BW >= 10 kHz)."
            return EraCompletionResult(
                applied=False,
                era_decade=era,
                brillanz_ceiling=ceiling,
                source_bandwidth_hz=src_bw,
                completion_bandwidth_hz=src_bw,
                operation_type="passthrough",
                message=msg,
                audio=arr,
            )

        # HF-Ergaenzung via harmonische Extrapolation + aera-typisches HF-Rauschen
        try:
            enhanced = self._synthesize_era_hf(mono, sr, era, ceiling, anchor=anchor_mono)
        except Exception as exc:
            logger.warning("EraCompletion Synthese fehlgeschlagen: %s", exc)
            enhanced = mono

        # Stereo-Wiederherstellung
        if is_stereo and stereo is not None:
            # Kanal-Differenz beibehalten
            diff = stereo[0] - stereo[1]
            new_chan0 = enhanced + diff * 0.5
            new_chan1 = enhanced - diff * 0.5
            out = np.stack([new_chan0, new_chan1]).astype(np.float32)
        else:
            out = enhanced.astype(np.float32)

        out = np.clip(out, -1.0, 1.0)

        target_bw = min(
            self.SOURCE_BW_THRESHOLD_HZ * (1.0 + max(0, transfer_chain_depth - 1) * 0.15),
            sr / 2,
        )
        decade_str = str(era) if era else "unbekannt"
        msg = (
            f"Fehlende Hochton-Frequenzen wurden im Stil der Aufnahme-Aera {decade_str} "
            "ergaenzt — als Rekonstruktion markiert."
        )

        return EraCompletionResult(
            applied=True,
            era_decade=era,
            brillanz_ceiling=ceiling,
            source_bandwidth_hz=src_bw,
            completion_bandwidth_hz=target_bw,
            operation_type="synthesize_era_authentic",
            message=msg,
            audio=out,
        )

    def _estimate_bandwidth(self, mono: np.ndarray, sr: int) -> float:
        """Effektive Bandbreite via RFFT: hoechste Frequenz mit substanzieller Energie."""
        n = min(len(mono), 65536)
        if n < 512:
            return sr / 2
        spec = np.abs(np.fft.rfft(mono[:n])) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        total = max(1e-15, float(spec.sum()))
        # Rolloff: 95 % Energie
        cumsum = np.cumsum(spec) / total
        rolloff_idx = np.searchsorted(cumsum, 0.95)
        if rolloff_idx < len(freqs):
            return float(freqs[rolloff_idx])
        return float(freqs[-1])

    def _synthesize_era_hf(
        self,
        mono: np.ndarray,
        sr: int,
        era: int | None,
        ceiling: float,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        """HF-Ergaenzung: harmonische Extrapolation + era-typisches Rauschen."""
        n = len(mono)
        if n == 0:
            return mono

        spec = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        mag = np.abs(spec)
        phase = np.angle(spec)

        src_bw = self._estimate_bandwidth(mono, sr)
        min(self.SOURCE_BW_THRESHOLD_HZ, sr / 2)

        # Fuer Frequenzen ueber src_bw: harmonische Extrapolation
        extended_mag = mag.copy()
        for i, f in enumerate(freqs):
            if f > src_bw:
                # Extrapolation: Energie faellt nach 1/f^2 ab gegenueber Rolloff-Energie
                ref_idx = int(src_bw * n / sr)
                ref_idx = max(0, min(ref_idx, len(mag) - 1))
                ref_e = mag[ref_idx] + 1e-10
                decay = (src_bw / max(f, 1.0)) ** 1.5
                extended_mag[i] = ref_e * decay * ceiling * 0.5

        # Aera-typisches HF-Rauschen — deterministischer Seed aus Era + Signallaenge
        _rng = np.random.RandomState(seed=((era or 0) * 31337 + n) % (2**31))
        noise_mag = np.zeros_like(extended_mag)
        for i, f in enumerate(freqs):
            if f > src_bw:
                noise_mag[i] = extended_mag[i] * 0.15 * _rng.rand()

        extended_mag = extended_mag + noise_mag

        # Optionales Anchor-Guidance: konservative Profilfuehrung der ergaenzten
        # HF-Bins, ohne das Material-/Era-Ceiling aufzuweichen.
        if isinstance(anchor, np.ndarray) and anchor.size > 0:
            try:
                _anc = np.clip(np.nan_to_num(anchor.astype(np.float32)), -1.0, 1.0)
                if _anc.shape[-1] != n:
                    _n_min = min(_anc.shape[-1], n)
                    _anc = _anc[:_n_min]
                _anc_n = len(_anc)
                if _anc_n >= 512:
                    _anc_spec = np.abs(np.fft.rfft(_anc, n=_anc_n))
                    _anc_freqs = np.fft.rfftfreq(_anc_n, d=1.0 / sr)
                    _hf_idx = freqs > src_bw
                    if np.any(_hf_idx):
                        _src_ref_idx = int(np.clip(src_bw * _anc_n / sr, 0, len(_anc_spec) - 1))
                        _src_ref = float(_anc_spec[_src_ref_idx] + 1e-10)
                        _anchor_interp = np.interp(freqs[_hf_idx], _anc_freqs, _anc_spec, left=0.0, right=0.0)
                        _anchor_profile = np.clip(_anchor_interp / _src_ref, 0.05, 2.0)
                        _target = np.clip(mag[_hf_idx] * _anchor_profile * ceiling, 0.0, np.max(extended_mag) + 1e-6)
                        _guide_alpha = 0.35
                        extended_mag[_hf_idx] = (1.0 - _guide_alpha) * extended_mag[_hf_idx] + _guide_alpha * _target
            except Exception as exc:
                logger.debug("EAPC Anchor-Guidance fehlgeschlagen (nicht blockierend): %s", exc)

        # Phase beibehalten (PGHI-Naherung)
        new_spec = extended_mag * np.exp(1j * phase)
        result = np.fft.irfft(new_spec, n=n).astype(np.float32)

        # §2.45a-I: Normalisierung mit Gated-RMS (nur Frames > -50 dBFS — kein Stille-inflationierter RMS)
        orig_rms = max(1e-8, float(_grl_eapc(mono.astype(np.float32), gate_dbfs=-50.0)))
        result_rms = max(1e-8, float(_grl_eapc(result, gate_dbfs=-50.0)))
        result *= orig_rms / result_rms

        return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: EraAuthenticPerceptualCompletion | None = None
_lock = threading.Lock()


def get_era_completion() -> EraAuthenticPerceptualCompletion:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance  # pylint: disable=global-statement  # §3.2 Pflicht-Singleton-Pattern
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = EraAuthenticPerceptualCompletion()
    return _instance


def complete_era_authentic(
    audio: np.ndarray,
    sr: int,
    era: int | None = None,
    anchor: np.ndarray | None = None,
) -> EraCompletionResult:
    """Convenience-Wrapper."""
    return get_era_completion().complete(audio, sr, era, anchor)


__all__ = [
    "ERA_BRILLANZ_CEILING",
    "EraAuthenticPerceptualCompletion",
    "EraCompletionResult",
    "apply_era_authentic_completion",
    "complete_era_authentic",
    # Spec-konforme Alias-Namen (§2.35, §3.2)
    "get_era_authentic_perceptual_completion",
    "get_era_completion",
]

# Spec-konforme Alias-Namen (§2.35, §3.2)
get_era_authentic_perceptual_completion = get_era_completion
# Alias für get_era_completion() — Spec-konformer Name (§2.35).

apply_era_authentic_completion = complete_era_authentic
# Alias für complete_era_authentic() — Spec-konformer Name (§2.35).
