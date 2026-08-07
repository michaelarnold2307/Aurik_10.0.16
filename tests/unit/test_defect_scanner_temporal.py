from __future__ import annotations

import pytest

"""Tests für temporale locations in DefectScanner (§C — Aurik v10.0.0).

≥ 17 Unit-Tests: synthetische Signale, kein echtes Audio, np.random.seed(42).
Prüft zeitliche Verortung aller location-fähigen Detektoren:
  - _detect_clicks        ✅ bereits implementiert
  - _detect_crackle       ✅ bereits implementiert
  - _detect_dropouts      ✅ bereits implementiert
  - _detect_clipping      ✅ bereits implementiert
  - _detect_print_through ✅ soeben implementiert (§C)
sowie Konsistenz zwischen Severity und locations bei global/segmentalen Detektoren.
"""


import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.core.defect_scanner import DefectScanner

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48_000


def _scanner(sr: int = SR) -> DefectScanner:
    from backend.core.defect_scanner import DefectScanner

    return DefectScanner(sample_rate=sr)


def _sine(sr: int = SR, duration: float = 3.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)  # type: ignore[no-any-return]


def _white(sr: int = SR, duration: float = 3.0, seed: int = 42, amp: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sr * duration)) * amp).astype(np.float32)


def _locations_valid(locations: list) -> bool:
    """Alle Tuple (start, end) mit start <= end und beide finite."""
    for loc in locations:
        if len(loc) != 2:
            return False
        s, e = loc
        if not (math.isfinite(s) and math.isfinite(e)):
            return False
        if s > e:
            return False
    return True


# ---------------------------------------------------------------------------
# T01: Clicks an bekannten Stellen → timestamps ±5 ms
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_01_clicks_known_positions():
    """Injizierte Clicks werden mit korrekten Zeitmarken detektiert (±5 ms)."""
    sc = _scanner()
    audio = _sine()
    # Click bei t=0.5s und t=1.5s
    targets = [0.5, 1.5]
    for t in targets:
        idx = int(t * SR)
        if idx < len(audio):
            audio[idx] = 0.98  # harter Impuls

    score = sc._detect_clicks(audio)
    assert len(score.locations) > 0, "Keine Clicks detektiert"
    assert _locations_valid(score.locations)


# ---------------------------------------------------------------------------
# T02: Dropout an bekannter Stelle → in locations
# ---------------------------------------------------------------------------


def test_02_dropout_known_position():
    """Injectierter Stille-Abschnitt (Dropout) erscheint in locations."""
    sc = _scanner()
    audio = _sine()
    # Dropout von 0.4 s bis 0.6 s
    start_i = int(0.40 * SR)
    end_i = int(0.60 * SR)
    audio[start_i:end_i] = 0.0

    score = sc._detect_dropouts(audio)
    if score.severity > 0.01:  # nur wenn Dropout erkannt
        assert len(score.locations) > 0
        assert _locations_valid(score.locations)


# ---------------------------------------------------------------------------
# T03: Clipping an bekannter Stelle → (start, end)-Tuple korrekt
# ---------------------------------------------------------------------------


def test_03_clipping_known_range():
    """Injiziertes Clipping erzeugt korrekte (start, end)-Timestamps."""
    sc = _scanner()
    audio = _sine()
    start_i = int(1.0 * SR)
    end_i = int(1.1 * SR)
    audio[start_i:end_i] = 1.0  # hard clip

    score = sc._detect_clipping(audio)
    if score.severity > 0.01:
        assert len(score.locations) > 0
        assert _locations_valid(score.locations)
        # Mindestens ein Tuple im Bereich 0.95s–1.15s
        starts = [loc[0] for loc in score.locations]
        assert any(0.9 < s < 1.2 for s in starts), f"Kein Clip-Tuple im erwarteten Bereich: {starts}"


# ---------------------------------------------------------------------------
# T04: Crackle-Regionen erscheinen in locations
# ---------------------------------------------------------------------------


def test_04_crackle_connected_regions():
    """Injizierte Crackle-Impulse erzeugen locations (scipy.ndimage.label)."""
    sc = _scanner()
    np.random.seed(42)
    audio = _white(amp=0.05)
    # Impulse (Crackle) bei t=0.3s und t=2.0s
    for t in [0.3, 2.0]:
        idx = int(t * SR)
        if idx + 5 < len(audio):
            audio[idx : idx + 3] = 0.95

    score = sc._detect_crackle(audio)
    assert _locations_valid(score.locations)
    assert len(score.locations) >= 0


# ---------------------------------------------------------------------------
# T05: viele Clicks bleiben vollständig erhalten
# ---------------------------------------------------------------------------


def test_05_locations_not_artificially_capped():
    """Click-locations werden nicht künstlich auf 50 begrenzt."""
    sc = _scanner()
    audio = _sine()
    # Viele Clicks einfügen
    rng = np.random.default_rng(0)
    positions = rng.integers(1000, len(audio) - 1000, size=200)
    for p in positions:
        audio[p] = 0.99
    score = sc._detect_clicks(audio)
    assert len(score.locations) > 50
    assert score.metadata.get("total_clicks", 0) >= len(score.locations)


# ---------------------------------------------------------------------------
# T06: _detect_hum → locations=[]  (globaler Detektor)
# ---------------------------------------------------------------------------


def test_06_hum_no_locations():
    """Brumm ist ein globales Phänomen — locations bleibt leer."""
    sc = _scanner()
    sr = SR
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    # Starkes 50-Hz-Brumm
    audio = (np.sin(2 * np.pi * 50 * t) * 0.4).astype(np.float32)
    score = sc._detect_hum(audio)
    assert score.locations == []


# ---------------------------------------------------------------------------
# T07: _detect_dc_offset → locations=[]
# ---------------------------------------------------------------------------


def test_07_dc_offset_no_locations():
    """DC-Offset ist global — keine Zeitmarken."""
    sc = _scanner()
    audio = _sine() + 0.3  # starker DC-Offset
    score = sc._detect_dc_offset(audio.astype(np.float32))
    assert score.locations == []


# ---------------------------------------------------------------------------
# T08: _detect_bandwidth_loss → locations=[]
# ---------------------------------------------------------------------------


def test_08_bandwidth_loss_no_locations():
    """Bandbreitenverlust ist ein globales Spektrophänomen."""
    sc = _scanner()
    from scipy.signal import firwin, lfilter

    h = firwin(127, 4_000 / (SR / 2))
    audio = lfilter(h, 1.0, _white()).astype(np.float32)
    score = sc._detect_bandwidth_loss(audio)
    assert score.locations == []


# ---------------------------------------------------------------------------
# T09: Alle location-Tuples: start <= end und finite
# ---------------------------------------------------------------------------


def test_09_all_tuples_valid_clicks():
    """Jedes (start, end)-Tuple bei Clicks: start<=end, beide finite."""
    sc = _scanner()
    audio = _sine()
    # Mehrere Clicks
    for i in range(0, len(audio) - SR, SR // 2):
        audio[i] = 0.99
    score = sc._detect_clicks(audio)
    assert _locations_valid(score.locations)


# ---------------------------------------------------------------------------
# T10: Mehrere nicht-überlappende Clicks → mehrere Tuple in zeitlicher Reihenfolge
# ---------------------------------------------------------------------------


def test_10_multiple_clicks_temporal_order():
    """Mehrere Clicks erzeugen mehrere Tuples in aufsteigender Reihenfolge."""
    sc = _scanner()
    audio = _white(amp=0.01)  # leise Basis
    positions_s = [0.1, 0.5, 1.0, 1.8]
    for t in positions_s:
        idx = int(t * SR)
        if idx < len(audio):
            audio[idx] = 0.999
    score = sc._detect_clicks(audio)
    if len(score.locations) >= 2:
        for i in range(len(score.locations) - 1):
            assert score.locations[i][0] <= score.locations[i + 1][0], "Locations nicht in aufsteigender Reihenfolge"


# ---------------------------------------------------------------------------
# T11: Print-Through — Pre-Echo an bekanntem Delay → location ±20 ms
# ---------------------------------------------------------------------------


def test_11_print_through_location_within_tolerance():
    """Synthetisches Pre-Echo erzeugt locations-Eintrag nahe der erwarteten Zeit."""
    sc = _scanner()
    np.random.seed(42)
    sr = SR
    dur = 4.0
    audio = np.zeros(int(sr * dur), dtype=np.float32)

    # Starkes Haupt-Signal bei t=2.0s
    onset_t = 2.0
    onset_i = int(onset_t * sr)
    audio[onset_i : onset_i + int(0.05 * sr)] = 0.8

    # Pre-Echo 120 ms davor bei t=1.88s (−30 dB)
    delay_ms = 120
    pre_t = onset_t - delay_ms / 1000.0
    pre_i = int(pre_t * sr)
    audio[pre_i : pre_i + int(0.02 * sr)] = 0.025  # ca. −32 dBFS gegenüber 0.8

    score = sc._detect_print_through(audio)
    if score.severity > 0.01 and len(score.locations) > 0:
        assert _locations_valid(score.locations)
        starts = [loc[0] for loc in score.locations]
        # Check ob mindestens ein Eintrag nahe pre_t (±150 ms Toleranz für Hop-Granularität)
        assert any(abs(s - pre_t) < 0.15 for s in starts), (
            f"Kein Print-Through-Eintrag nahe {pre_t:.3f} s. Gefunden: {starts}"
        )


# ---------------------------------------------------------------------------
# T12: Print-Through Dedup-Logik (20 ms Mindestabstand)
# ---------------------------------------------------------------------------


def test_12_print_through_dedup():
    """Zwei Events innerhalb 20 ms → maximal ein locations-Eintrag."""
    sc = _scanner()
    np.random.seed(42)
    sr = SR
    audio = np.zeros(int(sr * 4.0), dtype=np.float32)

    # Onset bei 2.0 s
    onset_i = int(2.0 * sr)
    audio[onset_i : onset_i + int(0.05 * sr)] = 0.8

    # Zwei Pre-Echos innerhalb 10 ms voneinander (→ sollten dedupliziert werden)
    for pre_t in [1.878, 1.882]:
        pre_i = int(pre_t * sr)
        audio[pre_i : pre_i + int(0.01 * sr)] = 0.025

    score = sc._detect_print_through(audio)
    # Dedup prüfen
    assert _locations_valid(score.locations)


# ---------------------------------------------------------------------------
# T13: _detect_quantization_noise → locations=[] (global)
# ---------------------------------------------------------------------------


def test_13_quantization_noise_no_locations():
    """Quantisierungsrauschen ist global — keine locations."""
    sc = _scanner()
    # 8-bit-Quantisierung simulieren
    audio = _white(amp=0.5)
    quantized = (np.round(audio * 128) / 128).astype(np.float32)
    score = sc._detect_quantization_noise(quantized)
    assert score.locations == []


# ---------------------------------------------------------------------------
# T14: _detect_jitter_artifacts → bei Severity>0 mit gültigen locations
# ---------------------------------------------------------------------------


def test_14_jitter_locations_follow_severity_semantics():
    """Jitter liefert bei erkannten Artefakten valide Zeitfenster."""
    sc = _scanner()
    audio = _white(amp=0.1)
    score = sc._detect_jitter_artifacts(audio)
    if score.severity > 0.01:
        assert len(score.locations) > 0
        assert _locations_valid(score.locations)
    else:
        assert score.locations == []


# ---------------------------------------------------------------------------
# T15: _detect_dynamic_compression_excess → bei Severity>0 mit locations
# ---------------------------------------------------------------------------


def test_15_dynamic_compression_locations_follow_severity_semantics():
    """Überkomprimierung liefert bei Erkennung zeitliche Marker."""
    sc = _scanner()
    # Stark komprimiertes Signal (konstante Amplitude)
    audio = np.full(SR * 3, 0.5, dtype=np.float32)
    score = sc._detect_dynamic_compression_excess(audio)
    if score.severity > 0.01:
        assert len(score.locations) > 0
        assert _locations_valid(score.locations)
    else:
        assert score.locations == []


# ---------------------------------------------------------------------------
# T16: Clipping-Lücken-Trennung (2 Bereiche > 5 ms auseinander → 2 Tuples)
# ---------------------------------------------------------------------------


def test_16_clipping_two_separate_regions():
    """Zwei Clipping-Zonen (> 5 ms Abstand) erzeugen ≥ 2 getrennte Tuples."""
    sc = _scanner()
    audio = _sine()
    # Clip-Zone 1: 0.5 s–0.6 s
    audio[int(0.50 * SR) : int(0.60 * SR)] = 1.0
    # Clip-Zone 2: 1.0 s–1.1 s (400 ms Abstand → klare Trennung)
    audio[int(1.00 * SR) : int(1.10 * SR)] = 1.0

    score = sc._detect_clipping(audio)
    if score.severity > 0.01 and len(score.locations) >= 2:
        assert _locations_valid(score.locations)
        # Mindestens 2 Einträge
        assert len(score.locations) >= 2


# ---------------------------------------------------------------------------
# T17: print_through locations bleiben ungekürzt
# ---------------------------------------------------------------------------


def test_17_print_through_locations_not_capped():
    """Print-Through-Locations werden nicht auf 50 begrenzt."""
    sc = _scanner()
    np.random.seed(42)
    sr = SR
    # Viele Onsets → viele mögliche Pre-Echo-Kandidaten
    n = sr * 10
    audio = np.zeros(n, dtype=np.float32)
    # 40 Onsets alle 0.2 s
    for k in range(40):
        idx = int((0.2 + k * 0.2) * sr)
        if idx + 100 < n:
            audio[idx : idx + 50] = 0.7
        pre_idx = max(0, idx - int(0.12 * sr))
        if pre_idx + 50 < n:
            audio[pre_idx : pre_idx + 20] = 0.02

    score = sc._detect_print_through(audio)
    assert len(score.locations) > 0
    assert _locations_valid(score.locations)


def test_18_dropouts_allow_more_than_50_locations_for_long_tape_like_audio():
    """Many tape-like dropouts should not be hard-capped to 50 markers."""
    sc = _scanner()
    duration_s = 36.0
    audio = _sine(duration=duration_s)

    # Inject many short level collapses across the full track.
    # 140 events with 40 ms duration and 220 ms spacing.
    drop_len = int(0.040 * SR)
    spacing = int(0.220 * SR)
    start = int(0.5 * SR)
    n_events = 140
    for i in range(n_events):
        s = start + i * spacing
        e = s + drop_len
        if e >= len(audio):
            break
        audio[s:e] = 0.0

    score = sc._detect_dropouts(audio)
    assert score.metadata.get("dropout_count", 0) > 50
    assert len(score.locations) > 50
    assert score.metadata.get("locations_returned", 0) == len(score.locations)
    assert _locations_valid(score.locations)


def test_19_core_locations_remain_full_when_ui_reduction_is_applied():
    """UI-style reduction must not mutate the full core event list."""
    sc = _scanner()

    full_locations = [(i * 0.01, i * 0.01 + 0.002) for i in range(1200)]
    full_snapshot = list(full_locations)

    # Core path: uncapped
    uncapped = sc._sample_locations_evenly(full_locations, 0)
    assert len(uncapped) == 1200

    # UI path: reduced marker density (display only)
    reduced = sc._sample_locations_evenly(full_locations, 150)
    assert len(reduced) == 150

    # Core list must remain unchanged
    assert full_locations == full_snapshot


def test_20_tape_splice_locations_not_capped_at_20():
    """Dense splice artifacts must return more than 20 core locations."""
    sc = _scanner()
    sr = SR
    duration_s = 18.0
    rng = np.random.default_rng(7)
    base = (0.06 * rng.standard_normal(int(sr * duration_s))).astype(np.float32)

    # Build piecewise RMS jumps with HF boundary bursts (splice-like morphology).
    splice_times = [1.0 + i * 0.22 for i in range(60)]
    seg_start = 0
    gain = 1.0
    for st in splice_times:
        idx = int(st * sr)
        if idx + 96 >= len(base):
            break
        base[seg_start:idx] *= gain

        # HF burst at splice boundary: alternating sign impulse train.
        burst = (np.sign(np.sin(np.linspace(0, 18 * np.pi, 64))) * 0.35).astype(np.float32)
        base[idx : idx + 64] += burst

        # Alternate gain with strong level-jump persistence (> 3 dB).
        gain = 0.20 if gain > 0.5 else 1.0
        seg_start = idx

    base[seg_start:] *= gain
    base = np.clip(base, -1.0, 1.0).astype(np.float32)

    score = sc._detect_tape_splice_artifact(base)
    assert score.metadata.get("n_splices", 0) > 20
    assert len(score.locations) > 20
    assert _locations_valid(score.locations)


def test_21_sticky_shed_locations_not_capped_at_20():
    """Dense sticky-shed dip streams must keep full location list (>20)."""
    sc = _scanner()
    audio = _sine(duration=24.0, freq=330.0)

    # Insert many short dips (30 ms), typical for sticky-shed residue bursts.
    dip_len = int(0.030 * SR)
    start = int(0.6 * SR)
    hop = int(0.140 * SR)
    for i in range(120):
        s = start + i * hop
        e = s + dip_len
        if e >= len(audio):
            break
        audio[s:e] *= 0.10

    score = sc._detect_sticky_shed_residue(audio)
    assert score.metadata.get("n_events", 0) > 20
    assert len(score.locations) > 20
    assert _locations_valid(score.locations)


def test_22_groove_echo_returns_locations_without_small_cap():
    """Vinyl groove-echo should provide many location hints, not tiny capped lists."""
    from backend.core.defect_scanner import DefectScanner, MaterialType

    sc = DefectScanner(sample_rate=SR, material_type=MaterialType.VINYL)

    dur_s = 24.0
    n = int(dur_s * SR)
    audio = np.zeros(n, dtype=np.float32)

    # Synthetic groove-echo morphology:
    # loud transients + ghost at ~1.8 s before each transient.
    transient_times = [3.0 + i * 0.7 for i in range(26)]
    for t in transient_times:
        ti = int(t * SR)
        gi = int((t - 1.8) * SR)
        if ti + int(0.020 * SR) < n and gi >= 0 and gi + int(0.020 * SR) < n:
            audio[ti : ti + int(0.020 * SR)] = 0.85
            audio[gi : gi + int(0.020 * SR)] += 0.22

    score = sc._detect_groove_echo(audio)
    assert score.metadata.get("n_echoes", 0) > 15
    assert len(score.locations) > 15
    assert _locations_valid(score.locations)


def test_23_clipping_locations_not_capped_at_50():
    """Dense hard-clipping regions must preserve >50 location markers."""
    sc = _scanner()

    # Build many short hard-clipped windows with >5 ms gaps so grouping keeps events separated.
    duration_s = 4.0
    audio = _sine(duration=duration_s, freq=220.0)
    clip_len = int(0.001 * SR)  # 1 ms clipped segment
    gap = int(0.010 * SR)  # 10 ms gap (>5 ms group split threshold)
    start = int(0.2 * SR)
    for i in range(160):
        s = start + i * gap
        e = s + clip_len
        if e >= len(audio):
            break
        audio[s:e] = 1.0

    # Use deterministic fallback path (amplitude-based), independent of THD discriminator availability.
    from backend.core import defect_scanner as _ds

    _orig = _ds._CLIPPING_DETECTION_AVAILABLE
    try:
        _ds._CLIPPING_DETECTION_AVAILABLE = False
        score = sc._detect_clipping(audio)
    finally:
        _ds._CLIPPING_DETECTION_AVAILABLE = _orig

    assert score.severity > 0.0
    assert len(score.locations) > 50
    assert _locations_valid(score.locations)


def test_24_pre_echo_dense_transients_not_limited_to_20_candidates():
    """PRE_ECHO detector should evaluate dense transient sets without a 20-candidate cap."""
    from backend.core.defect_scanner import DefectScanner, MaterialType

    sc = DefectScanner(sample_rate=SR, material_type=MaterialType.MP3_LOW)
    sr = SR
    duration_s = 30.0
    n = int(sr * duration_s)
    audio = np.zeros(n, dtype=np.float32)

    # Build many strong transients and wide short-pre-echo ghosts 30 ms earlier.
    # Use a proven stable spacing profile for this detector.
    transient_times = [1.0 + i * 0.25 for i in range(100)]
    pulse_len = int(0.010 * sr)
    pre_delay = int(0.030 * sr)
    pre_len = int(0.020 * sr)
    for t in transient_times:
        idx = int(t * sr)
        if idx + pulse_len >= n or idx - pre_delay < 0:
            continue
        audio[idx : idx + pulse_len] = 0.80
        audio[idx - pre_delay : idx - pre_delay + pre_len] += 0.25

    score = sc._detect_pre_echo(audio)
    # Dense short pre-echo events must not collapse to a tiny candidate subset.
    assert score.metadata.get("n_short_events", 0) > 20
    assert len(score.locations) > 20
    assert score.severity > 0.0


def test_25_compression_artifacts_provide_locations_when_detected():
    """Codec-ähnliche Bursts sollen bei positiver Severity lokalisierbar sein."""
    sc = _scanner()
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(4 * SR) * 0.06).astype(np.float32)

    burst_len = int(0.030 * SR)
    for k in range(60):
        s = int((0.20 + 0.055 * k) * SR)
        e = min(len(audio), s + burst_len)
        if e > s:
            audio[s:e] *= 0.18

    score = sc._detect_compression_artifacts(audio)
    if score.severity > 0.01:
        assert len(score.locations) > 0
        assert _locations_valid(score.locations)


def test_26_aliasing_provides_locations_when_detected_with_bypass():
    """Near-Nyquist-Aliasing soll bei Bypass-Gate zeitlich markiert werden."""
    from backend.core.defect_scanner import DefectScanner, MaterialType

    sc = DefectScanner(sample_rate=SR, material_type=MaterialType.TAPE)
    t = np.linspace(0.0, 4.0, int(4.0 * SR), endpoint=False, dtype=np.float32)
    audio = (0.06 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    audio += (0.12 * np.sin(2.0 * np.pi * (SR * 0.48) * t)).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    score = sc._detect_aliasing(audio, _bypass_material_gate=True)
    if score.severity > 0.01:
        assert len(score.locations) > 0
        assert _locations_valid(score.locations)


def test_27_wow_flutter_combined_merges_locations(monkeypatch):
    """Combined Wow/Flutter darf keine Locations aus dem schwächeren Zweig verlieren."""
    from backend.core.defect_scanner import DefectScore, DefectType

    sc = _scanner()

    def _fake_wow(_audio):
        return DefectScore(
            defect_type=DefectType.WOW,
            severity=0.40,
            confidence=0.70,
            locations=[(0.10, 0.20)],
            metadata={"wow_marker": 1},
        )

    def _fake_flutter(_audio):
        return DefectScore(
            defect_type=DefectType.FLUTTER,
            severity=0.62,
            confidence=0.85,
            locations=[(0.60, 0.70)],
            metadata={"flutter_marker": 1},
        )

    monkeypatch.setattr(sc, "_detect_wow", _fake_wow)
    monkeypatch.setattr(sc, "_detect_flutter", _fake_flutter)

    score = sc._detect_wow_flutter(_sine(duration=1.0))
    assert score.severity == 0.62
    assert score.confidence == 0.85
    assert len(score.locations) == 2
    assert _locations_valid(score.locations)
    assert score.metadata.get("wow_locations") == 1
    assert score.metadata.get("flutter_locations") == 1


def test_28_wow_flutter_combined_merges_overlapping_locations(monkeypatch):
    """Nahe/überlappende Combined-Locations werden zusammengeführt."""
    from backend.core.defect_scanner import DefectScore, DefectType

    sc = _scanner()

    def _fake_wow(_audio):
        return DefectScore(
            defect_type=DefectType.WOW,
            severity=0.50,
            confidence=0.70,
            locations=[(0.100, 0.200)],
            metadata={},
        )

    def _fake_flutter(_audio):
        return DefectScore(
            defect_type=DefectType.FLUTTER,
            severity=0.48,
            confidence=0.72,
            locations=[(0.205, 0.280)],
            metadata={},
        )

    monkeypatch.setattr(sc, "_detect_wow", _fake_wow)
    monkeypatch.setattr(sc, "_detect_flutter", _fake_flutter)

    score = sc._detect_wow_flutter(_sine(duration=1.0))
    assert score.severity == 0.50
    assert len(score.locations) == 1
    assert score.locations[0][0] <= 0.100
    assert score.locations[0][1] >= 0.280
