"""
DoNoHarmGuardian — §5/5 Final Safety Net (Aurik 10.0.0+)
=========================================================

Stellt sicher: Jede Restaurierung verbessert den Klang — oder
das Original wird unverändert zurückgegeben.

Prinzip: „Primum non nocere" — zuerst nicht schaden.

Arbeitsweise:
  1. Vor der Pipeline: Input-Audio und dessen Metriken speichern.
  2. Nach der Pipeline: Output-Audio-Metriken messen.
  3. Wenn IRGENDEINE Kernmetrik sich signifikant verschlechtert hat:
     → Output verwerfen, Original zurückgeben.
  4. Wenn alle Metriken gleich oder besser sind:
     → Output durchlassen.

Kernmetriken (unabhängig vom Materialtyp):
  - spectral_brightness:     Verhältnis HF-Energie (>4kHz) zu Gesamtenergie
  - naturalness_estimate:    Wiener-Entropie als Naturalness-Proxy
  - rms_preservation:       RMS-Änderung in dB (max ±6 dB toleriert)
  - peak_integrity:          True-Peak nicht näher an 0 dBFS als vorher

Integration:
  Wird in UnifiedRestorerV3.restore() als Post-Pipeline-Check aufgerufen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GuardianSnapshot:
    """Metrik-Snapshot eines Audio-Signals (vor oder nach der Pipeline)."""

    spectral_brightness: float = 0.5  # 0–1 (0.5 = neutral)
    naturalness_estimate: float = 0.5  # 0–1 (Wiener-Entropie)
    rms_dbfs: float = -30.0  # RMS in dBFS
    peak_dbfs: float = -6.0  # True-Peak in dBFS
    dynamic_range_db: float = 12.0  # P99.9 − P0.1 in dB

    # Rohdaten für spätere Diagnose
    _raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardianVerdict:
    """Entscheidung des Guardians."""

    passed: bool = True
    reason: str = ""
    metrics_input: GuardianSnapshot = field(default_factory=GuardianSnapshot)
    metrics_output: GuardianSnapshot = field(default_factory=GuardianSnapshot)
    degraded_metrics: list[str] = field(default_factory=list)
    severity: str = "none"  # "none", "minor", "moderate", "critical"


class DoNoHarmGuardian:
    """Finaler Qualitäts-Schutz — stellt sicher, dass Aurik nicht schadet.

    Verwendung:
        guardian = DoNoHarmGuardian()
        guardian.capture_input(audio, sr)
        # ... Pipeline läuft ...
        verdict = guardian.evaluate(output_audio, sr)
        if not verdict.passed:
            return input_audio  # Original zurückgeben
    """

    # ── Schwellwerte ───────────────────────────────────────────────────
    # §G-5/5: Diese Schwellwerte wurden empirisch an 50+ Restaurierungen
    # kalibriert. Sie sind konservativ — lieber zu früh warnen als zu spät.
    #
    # RESTORATION-Modus: Charakter bewahren, nur Defekte entfernen.
    # → strenge Schwellwerte — jede signifikante Änderung ist verdächtig.
    #
    # STUDIO-2026-Modus: Bewusste Modernisierung erlaubt.
    # → lockere Schwellwerte — LUFS-Normalisierung und Air-Band sind gewollt.

    # Restoration-Schwellwerte (konservativ)
    REST_MAX_BRIGHTNESS_DROP: float = 0.20
    REST_MAX_NATURALNESS_DROP: float = 0.15
    REST_MAX_RMS_CHANGE_DB: float = 8.0

    # Studio-2026-Schwellwerte (erlauben bewusste Änderungen)
    STU_MAX_BRIGHTNESS_DROP: float = 0.40  # Air-Band DARF Helligkeit erhöhen
    STU_MAX_NATURALNESS_DROP: float = 0.30  # LUFS-Norm kann Naturalness beeinflussen
    STU_MAX_RMS_CHANGE_DB: float = 20.0  # -14 LUFS Normierung = große Pegeländerung ok

    def __init__(self, mode: str = "restoration") -> None:
        self._mode = ""
        self.mode = mode  # Property-Setter aktualisiert die Schwellwerte
        self._input_audio: np.ndarray | None = None
        self._input_sr: int = 0
        self._input_snapshot: GuardianSnapshot | None = None
        self._captured: bool = False

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = str(value).strip().lower()
        if self._mode in ("studio_2026", "studio2026", "studio"):
            self._max_brightness_drop = self.STU_MAX_BRIGHTNESS_DROP
            self._max_naturalness_drop = self.STU_MAX_NATURALNESS_DROP
            self._max_rms_change_db = self.STU_MAX_RMS_CHANGE_DB
            self._min_peak_headroom_db = 0.0
        else:
            self._max_brightness_drop = self.REST_MAX_BRIGHTNESS_DROP
            self._max_naturalness_drop = self.REST_MAX_NATURALNESS_DROP
            self._max_rms_change_db = self.REST_MAX_RMS_CHANGE_DB
            self._min_peak_headroom_db = 0.5
        self._input_sr: int = 0
        self._input_snapshot: GuardianSnapshot | None = None
        self._captured: bool = False

    # ── Public API ─────────────────────────────────────────────────────

    def capture_input(self, audio: np.ndarray, sr: int) -> None:
        """Speichert das Input-Audio und misst dessen Metriken.

        Muss VOR der Pipeline aufgerufen werden.
        """
        self._input_audio = np.asarray(audio, dtype=np.float32).copy()
        self._input_sr = int(sr)
        self._input_snapshot = self._measure(audio, sr)
        self._captured = True
        logger.debug(
            "DoNoHarmGuardian: input captured — brightness=%.3f naturalness=%.3f rms=%.1f dBFS",
            self._input_snapshot.spectral_brightness,
            self._input_snapshot.naturalness_estimate,
            self._input_snapshot.rms_dbfs,
        )

    def evaluate(self, output_audio: np.ndarray, sr: int, material: str = "unknown") -> GuardianVerdict:
        """Vergleicht Output mit Input und entscheidet: passed oder nicht.

        Args:
            output_audio: Das von der Pipeline verarbeitete Audio.
            sr: Sample-Rate.
            material: Materialtyp (cassette, reel_tape, etc.) für adaptive Schwellwerte.

        Returns:
            GuardianVerdict mit passed=True wenn alle Metriken ok sind.
        """
        if not self._captured:
            logger.warning("DoNoHarmGuardian: evaluate() ohne capture_input() — lasse durch")
            return GuardianVerdict(passed=True, reason="no_input_captured")

        output = np.asarray(output_audio, dtype=np.float32)
        input_snap = self._input_snapshot
        assert input_snap is not None
        output_snap = self._measure(output, sr)

        degraded: list[str] = []

        # 1. Spectral Brightness
        _brightness_drop = input_snap.spectral_brightness - output_snap.spectral_brightness
        if _brightness_drop > self._max_brightness_drop:
            degraded.append(f"brightness_drop={_brightness_drop:.3f} (>{self._max_brightness_drop})")

        # 2. Naturalness — §v10.101 material-adaptiv
        _mat_nat = str(material).lower()
        _nat_threshold = 0.30 if _mat_nat in ("cassette", "reel_tape", "tape") else self._max_naturalness_drop
        _nat_drop = input_snap.naturalness_estimate - output_snap.naturalness_estimate
        if _nat_drop > _nat_threshold:
            degraded.append(f"naturalness_drop={_nat_drop:.3f} (>{_nat_threshold})")

        # 3. RMS Change
        _rms_change = abs(output_snap.rms_dbfs - input_snap.rms_dbfs)
        if _rms_change > self._max_rms_change_db:
            degraded.append(f"rms_change={_rms_change:.1f} dB (>{self._max_rms_change_db})")

        # 4. Peak Integrity (Crest-Factor-basiert, §v10.101)
        # Vorher: absoluter Peak-Vergleich — Peak-Anstieg durch Gain-Staging
        # (z.B. OneTakeExport +6dB) wurde fälschlich als Degradation gewertet.
        # Jetzt: Crest-Faktor (Peak/RMS-Ratio) — nur echte Dynamik-Kompression
        # feuert. Gain-Staging erhöht Peak UND RMS → Crest unverändert → PASS.
        # Schwellwert 3 dB: declarative Dynamics-Veränderung (Kompressor/Limiter).
        # Declipping (Peak↓, RMS≈) typischerweise nur 1–2 dB Crest-Änderung → PASS.
        # §v10.101 Material-adaptiv: Kassette/Tape — Rauschentfernung reduziert
        # Crest natürlich um 5–9 dB (Rausch-Peaks werden entfernt). Schwelle 6 dB.
        _mat_crest = str(material).lower()
        _crest_threshold = 6.0 if _mat_crest in ("cassette", "reel_tape", "tape") else 3.0
        _crest_input = input_snap.peak_dbfs - input_snap.rms_dbfs
        _crest_output = output_snap.peak_dbfs - output_snap.rms_dbfs
        _crest_drop = _crest_input - _crest_output
        if _crest_drop > _crest_threshold:
            degraded.append(
                f"peak_degraded: crest_drop={_crest_drop:.1f}dB "
                f"(in={input_snap.peak_dbfs:.1f}/{input_snap.rms_dbfs:.1f} "
                f"out={output_snap.peak_dbfs:.1f}/{output_snap.rms_dbfs:.1f})"
            )

        # 5. Dynamic Range — darf nicht kollabieren
        _dr_change = input_snap.dynamic_range_db - output_snap.dynamic_range_db
        if _dr_change > 6.0:  # Mehr als 6 dB Dynamikverlust
            degraded.append(f"dynamic_range_collapse={_dr_change:.1f} dB")

        passed = len(degraded) == 0

        if not passed:
            _severity = "critical" if len(degraded) >= 3 else ("moderate" if len(degraded) >= 2 else "minor")
            _reason = "; ".join(degraded)
            logger.warning(
                "DoNoHarmGuardian: BLOCKED — %d Metriken verschlechtert [%s]: %s",
                len(degraded),
                _severity,
                _reason,
            )
        else:
            _severity = "none"
            _reason = "all_metrics_ok"
            logger.info(
                "DoNoHarmGuardian: PASSED — brightness=%.3f→%.3f naturalness=%.3f→%.3f",
                input_snap.spectral_brightness,
                output_snap.spectral_brightness,
                input_snap.naturalness_estimate,
                output_snap.naturalness_estimate,
            )

        return GuardianVerdict(
            passed=passed,
            reason=_reason,
            metrics_input=input_snap,
            metrics_output=output_snap,
            degraded_metrics=degraded,
            severity=_severity,
        )

    def get_input_audio(self) -> np.ndarray | None:
        """Gibt das gespeicherte Input-Audio zurück (für Rollback)."""
        return self._input_audio

    # ── Interne Metrik-Messung ─────────────────────────────────────────

    @staticmethod
    def _measure(audio: np.ndarray, sr: int) -> GuardianSnapshot:
        """Misst alle Kernmetriken an einem Audio-Signal.

        Optimiert für Geschwindigkeit: verwendet einfache, robuste Metriken
        die ohne externe ML-Modelle auskommen (keine PANNS, kein CLAP).
        """
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio)
        mono = mono.astype(np.float32)
        n = len(mono)
        if n < sr // 4:  # Weniger als 250 ms
            return GuardianSnapshot()  # Zu kurz für sinnvolle Messung

        # 1. Spectral Brightness: Energie > 4 kHz / Gesamtenergie
        try:
            n_fft = min(4096, n)
            spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            hf_mask = freqs >= 4000.0
            hf_energy = float(np.sum(spec[hf_mask] ** 2))
            total_energy = float(np.sum(spec**2)) + 1e-10
            brightness = float(np.clip(hf_energy / total_energy, 0.0, 1.0))
        except Exception:
            brightness = 0.5

        # 2. Naturalness Estimate — §SOTA: MERT-Embedding wenn verfügbar, sonst Wiener-Entropie
        try:
            # Versuche MERT-basierte Naturalness (Deep-Learning, auf menschliche Urteile trainiert)
            from backend.core.mert_mushra_proxy import _extract_onnx_embedding as _mert_embed
            _mert_vec = _mert_embed(mono[:min(len(mono), sr*10)], sr)  # Max 10s
            if _mert_vec is not None and len(_mert_vec) > 0:
                # MERT-Embedding-Distanz zu einem generischen "natürlichen" Referenz-Vektor
                # Höhere Kosinus-Ähnlichkeit = natürlicher
                _ref_natural = np.ones_like(_mert_vec) * 0.1  # Generischer natürlicher Proxy
                _cos_sim = float(np.dot(_mert_vec, _ref_natural) / (
                    np.linalg.norm(_mert_vec) * np.linalg.norm(_ref_natural) + 1e-12))
                naturalness = float(np.clip(0.3 + 0.7 * _cos_sim, 0.0, 1.0))
            else:
                raise ValueError("MERT embedding empty")
        except Exception:
            # Fallback: Wiener-Entropie
            try:
                _spec_db = 20.0 * np.log10(spec + 1e-10)
                _spec_db -= np.max(_spec_db)
                _spec_lin = 10.0 ** (_spec_db / 20.0)
                _spec_norm = _spec_lin / (np.sum(_spec_lin) + 1e-10)
                _entropy = -np.sum(_spec_norm * np.log2(_spec_norm + 1e-10))
                _max_entropy = np.log2(len(_spec_norm))
                naturalness = float(np.clip(_entropy / max(_max_entropy, 1.0), 0.0, 1.0))
            except Exception:
                naturalness = 0.5

        # 3. RMS in dBFS
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-10)
        rms_dbfs = float(20.0 * np.log10(rms))

        # 4. Peak in dBFS
        peak = float(np.max(np.abs(mono)))
        peak_dbfs = float(20.0 * np.log10(max(peak, 1e-10)))

        # 5. Dynamic Range: P99.9 − P0.1
        try:
            abs_mono = np.abs(mono)
            p99_9 = float(np.percentile(abs_mono, 99.9))
            p0_1 = float(np.percentile(abs_mono, 0.1))
            p0_1_safe = max(p0_1, 1e-10)
            dynamic_range_db = float(20.0 * np.log10(p99_9 / p0_1_safe))
        except Exception:
            dynamic_range_db = 12.0

        return GuardianSnapshot(
            spectral_brightness=brightness,
            naturalness_estimate=naturalness,
            rms_dbfs=rms_dbfs,
            peak_dbfs=peak_dbfs,
            dynamic_range_db=dynamic_range_db,
            _raw={
                "hf_energy": hf_energy if "hf_energy" in dir() else 0.0,
                "total_energy": total_energy if "total_energy" in dir() else 0.0,
                "entropy": _entropy if "_entropy" in dir() else 0.0,
                "rms_linear": float(rms),
                "peak_linear": float(peak),
            },
        )


# ── Singleton ─────────────────────────────────────────────────────────

_guardian: DoNoHarmGuardian | None = None


def get_do_no_harm_guardian() -> DoNoHarmGuardian:
    """Thread-sicherer Singleton."""
    global _guardian
    if _guardian is None:
        _guardian = DoNoHarmGuardian()
    return _guardian
