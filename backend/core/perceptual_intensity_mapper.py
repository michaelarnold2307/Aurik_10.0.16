"""
§v10 Perceptual Intensity Mapper (PIM) — Zentrale Entscheidungsintelligenz.

Der PIM ist die fehlende Orchestrierungsschicht zwischen Auriks
Entscheidungsmodulen und der DSP-Phasen-Ausführung. Er übersetzt
ALLE verfügbaren Kontext-Informationen in eine per-Frequenzband,
per-Zeitsegment kalibrierte Intensitäts-Map.

Architektur:
    ArtisticIntent ─┐
    PerceptualSalience ─┐
    SourceProfile ──┐   │
    SpectralAnalysis ─┐ │   │
    PMGG-Delta ───┐  │ │   │
                  ▼  ▼ ▼   ▼
            ┌─────────────────┐
            │  PIM ORCHESTRATOR │  ← NEU (diese Datei)
            └────────┬────────┘
                     │
                     ▼
        PerBandIntensityMap {freq_band: strength}
        PerSegmentIntensityMap {time_segment: strength}

Ein menschlicher Toningenieur entscheidet pro Frequenzband, pro
Song-Sektion und pro Defekttyp — der PIM automatisiert genau das.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 10 kritische Frequenzbänder (angepasst an Bark-Skala + musikalische Relevanz)
# Jedes Band bekommt eine eigene Intensität — kein "one size fits all"
CRITICAL_BANDS: dict[str, tuple[float, float]] = {
    "sub_bass": (20, 60),  # Sub-Bass — fast nie NR nötig
    "bass": (60, 250),  # Bass/Fundamental — sanft
    "low_mid": (250, 500),  # Untere Mitten — Vorsicht: Wärme
    "mid": (500, 2000),  # Mitten — Vorsicht: Stimme, Gitarren
    "presence": (2000, 4000),  # Präsenz — Vorsicht: Sprachverständlichkeit
    "low_treble": (4000, 6000),  # Untere Höhen — Sibilanz-Zone
    "mid_treble": (6000, 8000),  # Mittlere Höhen — De-Essing-Zone
    "high_treble": (8000, 12000),  # Hohe Höhen — Luft, Brillanz
    "air": (12000, 16000),  # Air-Band — fast nie NR, eher Boost
    "ultra": (16000, 20000),  # Ultra-HF — Rauschen ja, Signal nein
}

# Song-Sektions-Typen mit Default-Intensitäts-Modifikatoren

# §v10 #8: Wet/Dry-Mix pro Song-Sektion
SECTION_WET_DRY: dict[str, float] = {
    "intro": 0.40,  # Intro: mehr Dry (natürlich)
    "verse": 0.55,  # Strophe: moderate Bearbeitung
    "chorus": 0.75,  # Refrain: aggressivere Reinigung
    "bridge": 0.60,  # Bridge: ausgewogen
    "outro": 0.35,  # Outro: sehr natürlich
    "solo": 0.30,  # Solo: maximal natürlich
}

SECTION_MODIFIERS: dict[str, dict[str, float]] = {
    "intro": {"nr_mod": 0.4, "comp_mod": 0.3, "eq_mod": 0.5},
    "verse": {"nr_mod": 0.7, "comp_mod": 0.6, "eq_mod": 0.8},
    "chorus": {"nr_mod": 0.5, "comp_mod": 1.0, "eq_mod": 0.9},
    "bridge": {"nr_mod": 0.6, "comp_mod": 0.5, "eq_mod": 0.7},
    "outro": {"nr_mod": 0.3, "comp_mod": 0.3, "eq_mod": 0.5},
    "solo": {"nr_mod": 0.2, "comp_mod": 0.4, "eq_mod": 0.3},  # Solo: maximal schonen!
}


@dataclass
class PerBandIntensity:
    """Kalibrierte Intensität für EIN Frequenzband."""

    band_name: str
    freq_lo: float
    freq_hi: float
    # DSP-Parameter für dieses Band
    noise_reduction_strength: float = 0.5  # 0.0 = keine NR, 1.0 = maximale NR
    de_ess_strength: float = 0.5  # 0.0 = kein De-Essing
    eq_gain_db: float = 0.0  # EQ-Korrektur
    compression_ratio: float = 1.0  # 1.0 = keine Kompression
    transient_preserve: float = 1.0  # 1.0 = voll erhalten, 0.0 = glatt
    # Entscheidungs-Metadaten
    salience_weight: float = 1.0  # Wie salient ist dieses Band?
    risk_factor: float = 0.5  # Risiko der Überbearbeitung


@dataclass
class PerSegmentIntensity:
    """Kalibrierte Intensität für EINEN Zeit-Abschnitt."""

    section_type: str
    start_sample: int
    end_sample: int
    # Sektions-spezifische Modifikatoren
    overall_strength: float = 1.0  # Multiplikator für alle Intensitäten in dieser Sektion
    nr_mod: float = 1.0
    comp_mod: float = 1.0
    eq_mod: float = 1.0


@dataclass
class IntensityMap:
    """Vollständige Intensitäts-Map — die zentrale Entscheidung des PIM."""

    per_band: dict[str, PerBandIntensity] = field(default_factory=dict)
    per_segment: list[PerSegmentIntensity] = field(default_factory=list)
    global_modifiers: dict[str, float] = field(default_factory=dict)
    decision_log: list[str] = field(default_factory=list)

    def get_nr_strength(self, band: str, section: str = "verse") -> float:
        """Berechnet die finale NR-Stärke für ein Band in einer Sektion."""
        band_intensity = self.per_band.get(band, PerBandIntensity(band, 0, 0))
        base = band_intensity.noise_reduction_strength

        # Sektions-Modifikator finden
        for seg in self.per_segment:
            if seg.section_type == section:
                base *= seg.nr_mod
                break

        # Global-Modifikator
        base *= self.global_modifiers.get("nr_global", 1.0)

        return float(np.clip(base, 0.0, 1.0))


class PerceptualIntensityMapper:
    """§v10 Zentrale Entscheidungsintelligenz — übersetzt Kontext in Intensität."""

    def __init__(self) -> None:
        self._last_map: IntensityMap | None = None

    def compute_intensity_map(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        artistic_intent: Any | None = None,
        salience_result: Any | None = None,
        source_profile: Any | None = None,
        pmgg_delta: float = 0.0,
        song_structure: list[dict] | None = None,
        material: str = "unknown",
    ) -> IntensityMap:
        """Berechnet die optimale Intensitäts-Map aus ALLEN verfügbaren Kontexten.

        Dies ist der EINE Aufruf, der alle Entscheidungsmodule konsultiert und
        eine kohärente, per-Band, per-Segment kalibrierte Map zurückgibt.

        Args:
            audio:             Mono-Audio für Analyse
            sr:                Sample-Rate
            artistic_intent:   Von get_artistic_intent()
            salience_result:   Von PerceptualSalienceEstimator
            source_profile:    Von get_source_profile()
            pmgg_delta:        Aktuelles PMGG-Δ (0.0 = keine Änderung)
            song_structure:    Von SongStructureAnalyzer
            material:          Material-Typ

        Returns:
            IntensityMap mit per-Band und per-Segment kalibrierten Werten
        """
        imap = IntensityMap()
        mono = np.asarray(audio, dtype=np.float64)
        if mono.ndim > 1:
            mono = mono.mean(axis=-1) if mono.shape[-1] <= 2 else mono.mean(axis=0)

        # ── Schritt 1: Per-Band Basis-Intensität aus spektraler Analyse ──
        band_energies = self._analyze_band_energies(mono, sr)
        noise_floor_per_band = self._estimate_noise_floor_per_band(mono, sr)

        for band_name, (lo, hi) in CRITICAL_BANDS.items():
            energy = band_energies.get(band_name, -60.0)
            noise = noise_floor_per_band.get(band_name, -80.0)
            snr = energy - noise  # Höherer SNR = weniger NR nötig

            # Basis-NR: invers zum SNR
            nr_strength = float(np.clip(1.0 - (snr + 20.0) / 60.0, 0.0, 1.0))

            # Schütze musikalisch kritische Bänder
            if band_name in ("low_mid", "mid", "presence"):
                nr_strength *= 0.6  # Stimme schonen
            if band_name in ("air",):
                nr_strength *= 0.2  # Luft nie aggressiv entrauschen
            if band_name in ("ultra",):
                nr_strength = 0.8  # Ultra-HF: Rauschen ja, Signal nein → NR ok

            imap.per_band[band_name] = PerBandIntensity(
                band_name=band_name,
                freq_lo=lo,
                freq_hi=hi,
                noise_reduction_strength=nr_strength,
                de_ess_strength=0.7 if band_name in ("low_treble", "mid_treble") else 0.1,
                compression_ratio=1.0,
                transient_preserve=1.0 if band_name in ("mid", "presence") else 0.8,
            )

        # ── Schritt 2: Artistic Intent Override ──
        if artistic_intent is not None:
            intent = artistic_intent
            warmth = getattr(intent, "warmth_target", 0.5)
            brilliance = getattr(intent, "brilliance_target", 0.5)
            preserve_dyn = getattr(intent, "preserve_dynamics", True)
            risk = getattr(intent, "risk_tolerance", 0.3)

            # Wärme-Präferenz: schone tiefe Mitten
            for band in ("bass", "low_mid"):
                if band in imap.per_band:
                    imap.per_band[band].noise_reduction_strength *= 1.0 - warmth * 0.4

            # Brillanz-Präferenz: schone Höhen
            for band in ("high_treble", "air"):
                if band in imap.per_band:
                    imap.per_band[band].noise_reduction_strength *= 1.0 - brilliance * 0.5

            # Dynamik-Präferenz: reduziere Kompression
            if preserve_dyn:
                imap.global_modifiers["comp_global"] = 0.5

            # Risiko-Toleranz: skaliere Gesamt-Intensität
            imap.global_modifiers["nr_global"] = 0.6 + 0.4 * risk

            imap.decision_log.append(
                f"ArtisticIntent: warmth={warmth:.2f} brill={brilliance:.2f} dyn={preserve_dyn} risk={risk:.2f}"
            )

        # ── Schritt 3: Perceptual Salience → Intensität ──
        if salience_result is not None:
            mean_sal = getattr(salience_result, "mean_salience", 0.5)
            # Höhere Salience = mehr hörbare Defekte = mehr NR gerechtfertigt
            sal_mod = 0.7 + 0.3 * mean_sal
            imap.global_modifiers["nr_global"] = imap.global_modifiers.get("nr_global", 1.0) * sal_mod
            imap.decision_log.append(f"Salience: mean={mean_sal:.2f} → mod={sal_mod:.2f}")

        # ── Schritt 4: Source-Profile → Intensität ──
        if source_profile is not None:
            nf = getattr(source_profile, "expected_noise_floor_db", -60.0)
            if nf > -40:  # Sehr lautes Rauschen (Vinyl, Kassette)
                imap.global_modifiers["nr_global"] = imap.global_modifiers.get("nr_global", 1.0) * 1.3
                imap.decision_log.append(f"Source: high noise floor ({nf:.0f} dB) → NR +30%")
            elif nf < -70:  # Sehr leises Rauschen (CD, Streaming)
                imap.global_modifiers["nr_global"] = imap.global_modifiers.get("nr_global", 1.0) * 0.5
                imap.decision_log.append(f"Source: low noise floor ({nf:.0f} dB) → NR -50%")

        # ── Schritt 5: PMGG-Δ → Intensität ──
        if pmgg_delta < -0.02:  # Verschlechterung — Intensität reduzieren
            imap.global_modifiers["nr_global"] = imap.global_modifiers.get("nr_global", 1.0) * 0.7
            imap.decision_log.append(f"PMGG: Δ={pmgg_delta:.3f} (Verschlechterung) → NR -30%")
        elif pmgg_delta > 0.02:  # Verbesserung — Intensität beibehalten
            imap.decision_log.append(f"PMGG: Δ={pmgg_delta:.3f} (Verbesserung) → Intensität ok")

        # ── Schritt 6: Song-Struktur → Per-Segment-Intensität ──
        if song_structure:
            for seg in song_structure:
                stype = seg.get("label", "verse")
                mods = SECTION_MODIFIERS.get(stype, SECTION_MODIFIERS["verse"])
                imap.per_segment.append(
                    PerSegmentIntensity(
                        section_type=stype,
                        start_sample=int(seg.get("start_sample", 0)),
                        end_sample=int(seg.get("end_sample", 0)),
                        nr_mod=mods["nr_mod"],
                        comp_mod=mods["comp_mod"],
                        eq_mod=mods["eq_mod"],
                    )
                )
                imap.decision_log.append(f"Section {stype}: nr={mods['nr_mod']:.1f}x comp={mods['comp_mod']:.1f}x")
        else:
            # Fallback: Einheits-Sektion
            imap.per_segment.append(PerSegmentIntensity(section_type="verse", start_sample=0, end_sample=len(mono)))

        # ── Finale Clamp- und Safety-Guards ──
        for band_intensity in imap.per_band.values():
            band_intensity.noise_reduction_strength = float(np.clip(band_intensity.noise_reduction_strength, 0.0, 0.95))
            band_intensity.de_ess_strength = float(np.clip(band_intensity.de_ess_strength, 0.0, 0.90))
            band_intensity.compression_ratio = float(np.clip(band_intensity.compression_ratio, 1.0, 4.0))

        imap.global_modifiers["nr_global"] = float(np.clip(imap.global_modifiers.get("nr_global", 1.0), 0.1, 2.0))

        self._last_map = imap
        logger.info(
            "PIM: %d Bänder × %d Segmente — %d Entscheidungen",
            len(imap.per_band),
            len(imap.per_segment),
            len(imap.decision_log),
        )
        return imap

    def _analyze_band_energies(self, mono: np.ndarray, sr: int) -> dict[str, float]:
        """Misst die Energie pro kritischem Frequenzband."""
        from scipy import signal as scipy_signal

        energies = {}
        for band_name, (lo, hi) in CRITICAL_BANDS.items():
            if hi >= sr / 2:
                energies[band_name] = -90.0
                continue
            sos = scipy_signal.butter(4, [lo, hi], "bandpass", fs=sr, output="sos")
            filtered = scipy_signal.sosfilt(sos, mono)
            rms = float(np.sqrt(np.mean(filtered**2)) + 1e-12)
            energies[band_name] = 20.0 * np.log10(rms)
        return energies

    def _estimate_noise_floor_per_band(self, mono: np.ndarray, sr: int) -> dict[str, float]:
        """Schätzt den Rauschboden pro Frequenzband (P10-Methode)."""
        from scipy import signal as scipy_signal

        nf = {}
        frame_len = int(0.1 * sr)
        for band_name, (lo, hi) in CRITICAL_BANDS.items():
            if hi >= sr / 2:
                nf[band_name] = -80.0
                continue
            sos = scipy_signal.butter(4, [lo, hi], "bandpass", fs=sr, output="sos")
            filtered = scipy_signal.sosfilt(sos, mono)
            # P10 der Frame-RMS als Rauschboden
            rms_vals = []
            for i in range(0, len(filtered) - frame_len, frame_len):
                chunk = filtered[i : i + frame_len]
                rms_vals.append(20.0 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-12))
            if rms_vals:
                nf[band_name] = float(np.percentile(rms_vals, 10))
            else:
                nf[band_name] = -80.0
        return nf


# Singleton
_instance: PerceptualIntensityMapper | None = None


def get_perceptual_intensity_mapper() -> PerceptualIntensityMapper:
    """Gibt den PIM-Singleton zurück."""
    global _instance
    if _instance is None:
        _instance = PerceptualIntensityMapper()
    return _instance
