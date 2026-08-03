#!/usr/bin/env python3
"""Generate synthetic but realistic corpus audio files for Aurik benchmarking.

Creates 20+ unique recordings across 6 material categories with clean + damaged
variants and valid manifest entries. All files are synthetic (CC0 / Public Domain).

Usage: python scripts/generate_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "corpus"
SAMPLE_RATE = 48000
DURATION_S = 15  # 15 seconds per recording


# ── Musical primitives ──────────────────────────────────────────────────
def _note_freq(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _make_tone(freq: float, duration_s: float) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = np.sin(np.pi * t / duration_s) ** 0.5
    return np.sin(2 * np.pi * freq * t) * envelope * 0.7


def _make_chord(notes: list[int], duration_s: float) -> np.ndarray:
    signal = np.zeros(int(duration_s * SAMPLE_RATE))
    for midi in notes:
        signal += _make_tone(_note_freq(midi), duration_s)
    return signal / len(notes)


def _make_bassline(key_midi: int, duration_s: float) -> np.ndarray:
    pattern = [key_midi, key_midi + 4, key_midi + 7, key_midi + 5]
    note_len = duration_s / len(pattern)
    signal = np.zeros(int(duration_s * SAMPLE_RATE))
    for i, midi in enumerate(pattern):
        start = int(i * note_len * SAMPLE_RATE)
        note = _make_tone(_note_freq(midi - 12), note_len * 1.3)
        end = start + len(note)
        if end > len(signal):
            note = note[: len(signal) - start]
        signal[start : start + len(note)] += note
    return signal * 0.6


def _make_melody(key_midi: int, duration_s: float) -> np.ndarray:
    pentatonic = [0, 2, 4, 7, 9]
    notes = []
    note_len = 0.4
    t = 0.0
    while t < duration_s:
        degree = random.choice(pentatonic[:4])
        octave = random.choice([0, 0, 0, 1])
        notes.append((key_midi + degree + 12 * octave, min(note_len, duration_s - t)))
        t += note_len
    signal = np.zeros(int(duration_s * SAMPLE_RATE))
    t = 0.0
    for midi, length in notes:
        start = int(t * SAMPLE_RATE)
        note = _make_tone(_note_freq(midi), length * 1.1)
        end = start + len(note)
        if end > len(signal):
            note = note[: len(signal) - start]
        signal[start : start + len(note)] += note
        t += length
    return signal * 0.5


def _make_rhythm(duration_s: float) -> np.ndarray:
    signal = np.zeros(int(duration_s * SAMPLE_RATE))
    beat_len = 0.25
    for i in range(int(duration_s / beat_len)):
        if i % 4 in (0, 2):
            start = int(i * beat_len * SAMPLE_RATE)
            length = int(0.05 * SAMPLE_RATE)
            noise = np.random.randn(length) * 0.15
            envelope = np.exp(-np.arange(length) / (0.03 * SAMPLE_RATE))
            signal[start : start + length] += noise * envelope
    return signal * 0.3


def _make_stereo(mono: np.ndarray, width: float = 0.6) -> np.ndarray:
    """Create stereo from mono with slight channel differences."""
    stereo = np.zeros((len(mono), 2))
    stereo[:, 0] = mono * (1.0 - width * 0.3)
    stereo[:, 1] = mono * (1.0 + width * 0.3)
    delay = int(0.008 * SAMPLE_RATE)
    stereo[delay:, 1] += stereo[:-delay, 1] * 0.1
    return stereo / np.abs(stereo).max() * 0.9


# ── Defect generators (all handle mono and stereo) ──────────────────────
def _add_hiss(audio: np.ndarray, level: float = 0.02) -> np.ndarray:
    return audio + np.random.randn(*audio.shape) * level


def _add_hum(audio: np.ndarray, freq: float = 50.0, level: float = 0.03) -> np.ndarray:
    n = len(audio)
    t = np.arange(n) / SAMPLE_RATE
    hum = np.sin(2 * np.pi * freq * t) * level
    hum += np.sin(2 * np.pi * freq * 3 * t) * level * 0.3
    if audio.ndim > 1:
        hum = hum[:, np.newaxis]
    return audio + hum


def _add_wow_flutter(audio: np.ndarray, depth: float = 0.003, rate: float = 0.5) -> np.ndarray:
    n = len(audio)
    t = np.arange(n) / SAMPLE_RATE
    mod = 1.0 + depth * np.sin(2 * np.pi * rate * t + 0.3)
    indices = np.cumsum(mod) - mod[0]
    indices = np.clip(indices, 0, n - 1)
    if audio.ndim > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            result[:, ch] = np.interp(indices, np.arange(n), audio[:, ch])
        return result
    return np.interp(indices, np.arange(n), audio)


def _add_dropouts(audio: np.ndarray, count: int = 8) -> np.ndarray:
    result = audio.copy()
    n = len(audio)
    for _ in range(count):
        pos = random.randint(0, n - int(0.08 * SAMPLE_RATE))
        length = random.randint(int(0.01 * SAMPLE_RATE), int(0.08 * SAMPLE_RATE))
        fade = np.hanning(length * 2)[:length]
        if audio.ndim > 1:
            result[pos : pos + length, :] *= (1 - fade[:, np.newaxis])
        else:
            result[pos : pos + length] *= (1 - fade)
    return result


def _add_crackle(audio: np.ndarray, density: float = 0.05) -> np.ndarray:
    result = audio.copy()
    n_total = len(audio)
    n_clicks = int(n_total * density)
    for _ in range(n_clicks):
        pos = random.randint(0, n_total - 50)
        length = random.randint(2, 15)
        impulse = np.random.randn(length) * random.uniform(0.1, 0.4)
        decay = np.exp(-np.arange(length) / 5)
        shaped = impulse * decay
        end = min(pos + length, n_total)
        seg_len = end - pos
        if audio.ndim > 1:
            result[pos:end] += shaped[:seg_len, np.newaxis]
        else:
            result[pos:end] += shaped[:seg_len]
    return result


def _add_surface_noise(audio: np.ndarray, level: float = 0.015) -> np.ndarray:
    from scipy.signal import butter, lfilter

    n = len(audio)
    noise = np.random.randn(*audio.shape)
    b, a_coeff = butter(2, [300 / (SAMPLE_RATE / 2), 4000 / (SAMPLE_RATE / 2)], "band")
    noise = lfilter(b, a_coeff, noise, axis=0)
    noise *= level / (noise.std() + 1e-8)
    result = audio + noise
    for _ in range(random.randint(3, 8)):
        pos = random.randint(0, n - 100)
        pop = np.random.randn(50) * 0.3
        if audio.ndim > 1:
            result[pos : pos + 50, :] += pop[:, np.newaxis]
        else:
            result[pos : pos + 50] += pop
    return result


# ── Recording definitions ───────────────────────────────────────────────
# Each: (filename_base, material, era, genre, key_midi, defect_fns, is_vocal)
# defect_fns: list of (suffix_label, fn(audio)->damaged_audio)
RECORDINGS = [
    # ── Vinyl (5) ──
    ("vinyl_blues_1950s", "vinyl", 1954, "blues", 40,
     [("hiss_hum", lambda a: _add_hum(_add_hiss(a, 0.015), 50, 0.025)),
      ("crackle", lambda a: _add_surface_noise(a, 0.02))], True),
    ("vinyl_jazz_1960s", "vinyl", 1963, "jazz", 48,
     [("hiss_crackle", lambda a: _add_crackle(_add_hiss(a, 0.018), 0.04)),
      ("surface_noise", lambda a: _add_surface_noise(a, 0.025))], False),
    ("vinyl_rock_1970s", "vinyl", 1972, "rock", 45,
     [("clicks", lambda a: _add_crackle(a, 0.06)),
      ("hiss", lambda a: _add_hiss(a, 0.02))], True),
    ("vinyl_classical_1960s", "vinyl", 1961, "classical", 55,
     [("hiss_hum_wow", lambda a: _add_hum(_add_hiss(_add_wow_flutter(a, 0.002, 0.4), 0.012), 60, 0.02)),
      ("surface_noise", lambda a: _add_surface_noise(a, 0.018))], False),
    ("vinyl_soul_1970s", "vinyl", 1975, "soul", 42,
     [("crackle_hiss", lambda a: _add_hiss(_add_crackle(a, 0.05), 0.016)),
      ("wow_flutter", lambda a: _add_wow_flutter(a, 0.004, 0.6))], True),

    # ── Tape (5) ──
    ("tape_country_1960s", "tape", 1965, "country", 43,
     [("hiss", lambda a: _add_hiss(a, 0.025)),
      ("dropouts", lambda a: _add_dropouts(a, 10))], True),
    ("tape_folk_1970s", "tape", 1973, "folk", 50,
     [("hiss_hum", lambda a: _add_hum(_add_hiss(a, 0.02), 50, 0.03)),
      ("wow_flutter", lambda a: _add_wow_flutter(a, 0.003, 0.5))], True),
    ("tape_rock_1980s", "tape", 1982, "rock", 47,
     [("hiss", lambda a: _add_hiss(a, 0.022)),
      ("dropouts_hiss", lambda a: _add_dropouts(_add_hiss(a, 0.018), 6))], False),
    ("tape_jazz_1950s", "tape", 1958, "jazz", 52,
     [("hiss_hum", lambda a: _add_hum(_add_hiss(a, 0.028), 60, 0.035)),
      ("wow_flutter", lambda a: _add_wow_flutter(a, 0.005, 0.7))], False),
    ("tape_classical_1970s", "tape", 1976, "classical", 56,
     [("hiss", lambda a: _add_hiss(a, 0.019)),
      ("dropouts_hum", lambda a: _add_hum(_add_dropouts(a, 5), 50, 0.025))], False),

    # ── Reel tape (2 more, 1 exists) ──
    ("reel_jazz_1950s", "reel_tape", 1955, "jazz", 49,
     [("hiss_hum", lambda a: _add_hum(_add_hiss(a, 0.02), 60, 0.028)),
      ("dropouts", lambda a: _add_dropouts(a, 7))], False),
    ("reel_classical_1960s", "reel_tape", 1964, "classical", 54,
     [("hiss", lambda a: _add_hiss(a, 0.015)),
      ("wow_flutter", lambda a: _add_wow_flutter(a, 0.002, 0.35))], False),

    # ── Shellac (2 more, 1 exists) ──
    ("shellac_blues_1930s", "shellac", 1932, "blues", 38,
     [("hiss_crackle_hum", lambda a: _add_hum(_add_crackle(_add_hiss(a, 0.03), 0.08), 50, 0.04)),
      ("surface_noise", lambda a: _add_surface_noise(a, 0.035))], True),
    ("shellac_classical_1940s", "shellac", 1945, "classical", 53,
     [("crackle", lambda a: _add_crackle(a, 0.09)),
      ("hiss_hum", lambda a: _add_hum(_add_hiss(a, 0.025), 50, 0.03))], False),

    # ── Cassette (2 more, 1 exists) ──
    ("cassette_rock_1990s", "cassette", 1992, "rock", 46,
     [("hiss_wow", lambda a: _add_wow_flutter(_add_hiss(a, 0.025), 0.004, 0.55)),
      ("dropouts", lambda a: _add_dropouts(a, 6))], True),
    ("cassette_hiphop_1980s", "cassette", 1987, "hiphop", 36,
     [("hiss", lambda a: _add_hiss(a, 0.022)),
      ("wow_flutter", lambda a: _add_wow_flutter(a, 0.003, 0.45))], True),

    # ── Digital (2 more, 1 exists) ──
    ("digital_pop_2000s", "digital", 2003, "pop", 42,
     [("clicks", lambda a: _add_crackle(a, 0.03)),
      ("mp3_artifacts", lambda a: _add_hiss(a, 0.005))], True),
    ("digital_jazz_2010s", "digital", 2014, "jazz", 51,
     [("clicks", lambda a: _add_crackle(a, 0.02)),
      ("mp3_artifacts", lambda a: _add_hiss(a, 0.004))], False),
]


def _build_audio(key_midi: int) -> np.ndarray:
    """Build a 15-second stereo musical piece."""
    dur = DURATION_S
    mono = (
        _make_bassline(key_midi, dur)
        + _make_chord([key_midi, key_midi + 4, key_midi + 7], dur) * 0.4
        + _make_melody(key_midi, dur)
        + _make_rhythm(dur)
    )
    mono /= np.abs(mono).max() + 1e-8
    return _make_stereo(mono)


def _sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> int:
    generated = 0
    manifests: dict[str, list[dict]] = {}

    for base, material, era, genre, key, defect_fns, vocal in RECORDINGS:
        print(f"Generating: {material}/{base}")
        audio = _build_audio(key)

        # Clean version
        clean_path = CORPUS_ROOT / material / "clean" / f"{base}_clean.wav"
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(clean_path), audio, SAMPLE_RATE, subtype="PCM_16")
        generated += 1

        # Damaged variants
        for suffix, defect_fn in defect_fns:
            damaged_path = CORPUS_ROOT / material / "damaged" / f"{base}_{suffix}.wav"
            damaged_path.parent.mkdir(parents=True, exist_ok=True)
            damaged = defect_fn(audio)
            damaged /= np.abs(damaged).max() + 1e-8
            sf.write(str(damaged_path), damaged, SAMPLE_RATE, subtype="PCM_16")
            generated += 1

        # Build manifest entry
        duration = len(audio) / SAMPLE_RATE
        entry = {
            "file": f"clean/{base}_clean.wav",
            "duration_s": round(duration, 3),
            "sample_rate": SAMPLE_RATE,
            "material": material,
            "era_year": era,
            "genre": genre,
            "condition": "clean",
            "channels": 2,
            "bit_depth": 16,
            "vocal": vocal,
            "license": "CC0 (synthetisch generiert)",
            "source_attribution": "Aurik Corpus Generator (synthetisch, Public Domain)",
            "checksum_sha256": _sha256(clean_path),
            "defect_types": [],
        }
        manifests.setdefault(material, []).append(entry)

        for suffix, _ in defect_fns:
            dpath = CORPUS_ROOT / material / "damaged" / f"{base}_{suffix}.wav"
            dtypes = suffix.split("_")
            dentry = {
                "file": f"damaged/{base}_{suffix}.wav",
                "duration_s": round(duration, 3),
                "sample_rate": SAMPLE_RATE,
                "material": material,
                "era_year": era,
                "genre": genre,
                "condition": "damaged",
                "channels": 2,
                "bit_depth": 16,
                "vocal": vocal,
                "license": "CC0 (synthetisch generiert)",
                "source_attribution": "Aurik Corpus Generator (synthetisch, Public Domain)",
                "checksum_sha256": _sha256(dpath),
                "defect_types": dtypes,
            }
            manifests.setdefault(material, []).append(dentry)

    # Write manifests
    for material, entries in manifests.items():
        mpath = CORPUS_ROOT / material / "manifest.yaml"
        existing = []
        if mpath.exists():
            try:
                existing = yaml.safe_load(mpath.read_text()).get("entries", [])
            except Exception:
                pass
        all_entries = existing + entries
        manifest = {
            "corpus_version": "1.0.0",
            "material": material,
            "description": f"Synthetic {material} recordings for Aurik real-audio validation (CC0)",
            "entries": all_entries,
        }
        mpath.write_text(
            yaml.dump(manifest, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"  Wrote {mpath} ({len(all_entries)} entries)")

    print(f"\n✅ Generated {generated} audio files across {len(manifests)} materials")
    print(f"   Total unique recordings: {len(RECORDINGS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
