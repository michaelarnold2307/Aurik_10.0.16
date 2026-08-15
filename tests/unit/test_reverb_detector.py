"""§v10.998: Reverb-Tail-Detektion — die SOTA-Dereverb-Lücke, gemessen.

Messung vorher: Ein Hall-Fall (RT60 1.2s) wurde von der Consensus NIE
erkannt → 0 Reparatur-Schritte. Dieser Detektor schließt die Lücke —
mit dem Abkling-Check, der gehaltene Töne (flache Hülle) von Hall
(abfallende Hülle) unterscheidet.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from backend.core.defect_consensus_pipeline import detect_reverb_tail

SR = 48000


def _make_reverb(dry: np.ndarray, sr: int, rt60: float = 1.2) -> np.ndarray:
    rng = np.random.default_rng(1)
    n_ir = sr * 2
    s0 = int(0.030 * sr)
    ir = np.zeros(n_ir, dtype=np.float64)
    ir[s0:] = rng.standard_normal(n_ir - s0) * np.exp(-6.9078 * np.arange(n_ir - s0) / sr / rt60)
    ir[s0 : s0 + int(0.005 * sr)] += 0.8
    rev = fftconvolve(np.asarray(dry, dtype=np.float64), ir)[: len(dry)]
    rev = rev / (np.max(np.abs(rev)) + 1e-9) * (np.max(np.abs(dry)) + 1e-9)
    return rev.astype(np.float32)


def _sine(seconds: float = 2.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)


def _burst_train(seconds: float = 2.0, freq: float = 440.0) -> np.ndarray:
    """Impulsfolge (100 ms Ton / 400 ms Pause) — physikalisch korrekt für
    Hall-Tests: Nur in den Pausen zeigt sich der Nachhall-Abfall."""
    n = int(SR * seconds)
    sig = np.zeros(n, dtype=np.float32)
    on, off = int(0.100 * SR), int(0.400 * SR)
    period = on + off
    t = np.arange(on) / SR
    tone = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    for start in range(0, n - on, period):
        sig[start : start + on] = tone
    return sig


def test_dry_sustained_tone_is_not_reverb():
    """Gehaltener Ton (flache Hülle) darf NICHT als Hall gelten."""
    result = detect_reverb_tail(_sine(), SR)
    assert result["defects"] == []


def test_dry_burst_train_is_not_reverb():
    """Impulsfolge OHNE Hall: Pausen sind still — kein Abfall in den Pausen."""
    result = detect_reverb_tail(_burst_train(), SR)
    assert result["defects"] == []


def test_synthetic_hall_is_detected():
    """RT60 1.2s auf Impulsfolge → reverb_tail mit Severity > 0.1."""
    rev = _make_reverb(_burst_train(), SR, rt60=1.2)
    result = detect_reverb_tail(rev, SR)
    defects = result["defects"]
    assert len(defects) == 1
    d = defects[0]
    assert d["type"] == "reverb_tail"
    assert d["severity"] > 0.1, f"Severity zu niedrig: {d['severity']}"
    assert d["confidence"] > 0.4


def test_shorter_reverb_gets_lower_severity():
    """RT60 0.4s (dezent) → geringere Severity als RT60 1.2s (stark)."""
    mild = detect_reverb_tail(_make_reverb(_burst_train(), SR, rt60=0.4), SR)["defects"]
    strong = detect_reverb_tail(_make_reverb(_burst_train(), SR, rt60=1.2), SR)["defects"]
    assert mild and strong
    assert mild[0]["severity"] < strong[0]["severity"]


def test_consensus_includes_reverb_on_hall_case():
    """End-to-End: Die Consensus meldet REVERB_TAIL auf dem Hall-Fall."""
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline

    rev = _make_reverb(_burst_train(seconds=2.0), SR, rt60=1.2)
    manifest = DefectConsensusPipeline().analyze(rev, SR)
    reverb_defects = [d for d in manifest.defects if str(d.category).endswith("REVERB_TAIL")]
    assert reverb_defects, f"Kein REVERB_TAIL im Manifest: {[d.category for d in manifest.defects]}"
    assert reverb_defects[0].severity > 0.05  # Planner-Gate (sev < 0.05) überwunden


def test_severity_fallback_for_zero_severity_detections():
    """Selbst-widersprüchliche Befunde (detektiert, sev=0) → aus Confidence abgeleitet."""
    from backend.core.defect_consensus_pipeline import (
        DefectCategory,
        DefectHypothesis,
        ParallelDefectScanner,
    )

    class _FakeScanner(ParallelDefectScanner):
        def _register_detectors(self):
            self._detectors = [
                (
                    "fake",
                    lambda audio, sr: {
                        "defects": [{"type": "crackle", "start": 0.0, "end": 0.1, "severity": 0.0, "confidence": 0.8}]
                    },
                )
            ]
            self._registration_report = []

    hypotheses = _FakeScanner().scan_all(np.zeros(SR, dtype=np.float32), SR)
    assert len(hypotheses) == 1
    assert hypotheses[0].category == DefectCategory.CRACKLE
    assert hypotheses[0].severity >= 0.4  # max(0.05, 0.8*0.5)
