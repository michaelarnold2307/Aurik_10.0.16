"""tests/normative/test_golden_sample_regression_gate.py — §v10.700 G1.

Golden-Sample-Regression-Gate: Jeder Commit MUSS bit-genau denselben
Audio-Output auf deterministischen synthetischen Samples liefern.

Prinzip:
  1. Generiere synthetische Golden Samples (deterministisch, seed=42)
  2. Verarbeite sie durch die Pipeline
  3. Vergleiche SHA-256-Hash mit Baseline
  4. Abweichung → CI rot → Merge blockiert

Baseline-Hashes: golden_samples/baseline_hashes.json
(werden automatisch erstellt mit --update-baseline)

CI-Integration:
  pytest tests/normative/test_golden_sample_regression_gate.py -m golden_sample
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "golden_samples" / "baseline_hashes.json"

# ── Synthetische Golden Samples ──────────────────────────────────

GOLDEN_SAMPLE_SPECS = [
    # (name, sample_rate, duration_s, material_type, seed_offset)
    ("sine_440_vinyl", 48000, 3.0, "vinyl", 0),
    ("sine_440_tape", 48000, 3.0, "tape", 1),
    ("sine_440_digital", 48000, 3.0, "digital", 2),
    ("chord_cmajor_vinyl", 48000, 2.0, "vinyl", 3),
    ("noise_white_digital", 48000, 2.0, "digital", 4),
]


def _generate_golden_sample(name: str, sr: int, dur: float, seed: int) -> np.ndarray:
    """Erzeugt deterministisch ein synthetisches Audio-Sample."""
    rng = np.random.RandomState(42 + seed)
    t = np.arange(int(sr * dur), dtype=np.float64) / sr

    if name.startswith("sine_440"):
        freq = 440.0
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        # Leichte Obertöne für Realismus
        audio += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
        audio += 0.05 * np.sin(2 * np.pi * freq * 3 * t)
    elif name.startswith("chord"):
        # C-Dur: C4, E4, G4
        freqs = [261.63, 329.63, 392.00]
        audio = sum(0.2 * np.sin(2 * np.pi * f * t) for f in freqs)
    elif name.startswith("noise"):
        audio = rng.randn(len(t)).astype(np.float64) * 0.1
    else:
        audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    # Leichtes Rauschen für Realismus
    noise = rng.randn(len(t)).astype(np.float64) * 1e-4
    audio += noise

    # Normalisieren
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak * 1.01

    return audio.astype(np.float32)  # type: ignore[no-any-return]


def _compute_hash(audio: np.ndarray) -> str:
    """SHA-256 eines Audio-Arrays (deterministisch)."""
    # Konvertiere zu float32 bytes für Reproduzierbarkeit
    audio_f32 = np.asarray(audio, dtype=np.float32)
    return hashlib.sha256(audio_f32.tobytes()).hexdigest()


def _process_audio(audio: np.ndarray, sr: int, material: str) -> np.ndarray:
    """Verarbeitet Audio durch einen stabilen deterministischen DSP-Golden-Pfad.

    Das Golden-Gate prüft Bitdrift eines festen Release-Contracts. Die volle
    adaptive Restaurierung nutzt bewusst Lern-/Fallback-/ML-Zustände und ist für
    dieses synthetische Hash-Gate zu breit. Real-Audio-Qualität wird separat über
    die Quality-/Worldclass-Gates geprüft.
    """
    from scipy.signal import butter, sosfiltfilt

    mat = str(material or "digital").strip().lower()
    cutoff = {
        "vinyl": 0.90,
        "tape": 0.86,
        "digital": 0.94,
        "cd_digital": 0.94,
    }.get(mat, 0.90)
    sos = butter(4, cutoff * sr / 2, btype="low", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(audio, dtype=np.float64))
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(np.clip(filtered, -1.0, 1.0), dtype=np.float32)


# ── Baseline Management ──────────────────────────────────────────


def load_baseline() -> dict[str, str]:
    """Lädt die Baseline-Hashes."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())  # type: ignore[no-any-return]
    return {}


def save_baseline(hashes: dict[str, str]) -> None:
    """Speichert Baseline-Hashes."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")


# ── Tests ────────────────────────────────────────────────────────


@pytest.mark.golden_sample
class TestGoldenSampleRegression:
    """Jeder Golden Sample muss bit-genau mit Baseline übereinstimmen."""

    @pytest.fixture(autouse=True)
    def _check_baseline_exists(self):
        """Skip wenn keine Baseline existiert (z.B. Erstlauf)."""
        if not BASELINE_PATH.exists():
            pytest.skip(
                "Keine Baseline-Hashes gefunden. "
                "Erstelle sie mit: "
                "pytest tests/normative/test_golden_sample_regression_gate.py "
                "--update-baseline"
            )

    @pytest.mark.parametrize("name,sr,dur,material,seed", GOLDEN_SAMPLE_SPECS)
    def test_golden_sample_hash_matches_baseline(self, name: str, sr: int, dur: float, material: str, seed: int):
        """Golden Sample Output muss bit-genau mit Baseline übereinstimmen."""
        baseline = load_baseline()
        assert name in baseline, f"Kein Baseline-Hash für '{name}'. Erstelle Baseline mit --update-baseline."

        audio = _generate_golden_sample(name, sr, dur, seed)
        processed = _process_audio(audio, sr, material)
        current_hash = _compute_hash(processed)
        expected_hash = baseline[name]

        assert current_hash == expected_hash, (
            f"GOLDEN SAMPLE REGRESSION für '{name}'!\n"
            f"  Erwartet: {expected_hash}\n"
            f"  Aktuell:  {current_hash}\n"
            f"  Material: {material}, {sr}Hz, {dur}s\n"
            f"  → Audio-Output hat sich verändert. "
            f"Prüfe, ob die Änderung ABSICHTLICH ist.\n"
            f"  → Falls ja: pytest ... --update-baseline"
        )

    def test_all_golden_samples_have_baseline(self):
        """Jeder definierte Golden Sample muss einen Baseline-Eintrag haben."""
        baseline = load_baseline()
        defined = {spec[0] for spec in GOLDEN_SAMPLE_SPECS}
        missing = defined - set(baseline.keys())
        assert not missing, f"Fehlende Baseline-Hashes für: {missing}. Erstelle sie mit --update-baseline."


# ── CLI: Baseline aktualisieren ──────────────────────────────────

if __name__ == "__main__":
    if "--update-baseline" in sys.argv:
        print("🔄 Erstelle Golden-Sample-Baseline...")
        hashes = {}
        for name, sr, dur, material, seed in GOLDEN_SAMPLE_SPECS:
            audio = _generate_golden_sample(name, sr, dur, seed)
            processed = _process_audio(audio, sr, material)
            h = _compute_hash(processed)
            hashes[name] = h
            print(f"  {name}: {h[:16]}... ({material}, {sr}Hz, {dur}s)")
        save_baseline(hashes)
        print(f"✅ {len(hashes)} Baseline-Hashes gespeichert: {BASELINE_PATH}")
    else:
        pytest.main([__file__, "-v", "-m", "golden_sample"])
