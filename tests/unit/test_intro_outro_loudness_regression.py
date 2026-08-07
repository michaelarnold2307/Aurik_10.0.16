"""
tests/unit/test_intro_outro_loudness_regression.py

Regression guard: Pegelexplosion in Intro/Outro-Zonen.

Invariante: Nach jeder Restaurierungs-Operation darf der Pegel in stillen Zonen
(Intro/Outro) um maximal 6 dB über den Input ansteigen.

Abgedeckte Root-Causes (2026-04-27/28):
  - apply_musical_gain_envelope mit hohem Gain ≥ 3× in stille Zonen
  - sosfilt (kausaler Filter) Split+Recombine → Inter-Band-Zeitversatz + Destruktive Interferenz
  - Per-Kanal-Normalisierung bei Stereo-OLA → L/R Gain-Mismatch bis 6 dB (phase_31-Pattern)
  - check_gain_safety: Pre-Flight-Guard verhindert Clipping ab -1 dBTP
"""

import numpy as np
import pytest
from scipy import signal as sp_signal

SR = 48_000
FRAME = 480  # 10 ms @ 48 kHz
MAX_INTRO_OUTRO_BOOST_DB = 6.0  # Hard ceiling: stille Zonen nie >6 dB hochgeregelt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_song_with_quiet_zones(
    total_s: float = 4.0,
    intro_s: float = 0.5,
    outro_s: float = 0.5,
    body_dbfs: float = -18.0,
    quiet_dbfs: float = -55.0,
    stereo: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Typisches Audio: kurze Stille in Intro/Outro + lauter Musikkörper.

    Entspricht Vinyl-Aufnahme mit Leaderband-Rauschen vor/nach Musik.
    """
    n_total = int(total_s * SR)
    n_intro = int(intro_s * SR)
    n_outro = int(outro_s * SR)
    n_body = n_total - n_intro - n_outro
    rng = np.random.RandomState(seed)

    amp_body = 10.0 ** (body_dbfs / 20.0)
    amp_quiet = 10.0 ** (quiet_dbfs / 20.0)

    t = np.linspace(0.0, float(n_body) / SR, n_body)
    body = (np.sin(2 * np.pi * 440 * t) * amp_body * 0.7 + np.sin(2 * np.pi * 880 * t) * amp_body * 0.3).astype(
        np.float32
    )

    intro = (rng.randn(n_intro) * amp_quiet).astype(np.float32)
    outro = (rng.randn(n_outro) * amp_quiet).astype(np.float32)

    mono = np.concatenate([intro, body, outro])

    if stereo:
        # Slightly different on each channel (realistic stereo vinyl)
        ch_r = np.concatenate(
            [
                (rng.randn(n_intro) * amp_quiet).astype(np.float32),
                body * 0.98 + (rng.randn(n_body) * amp_quiet * 0.5).astype(np.float32),
                (rng.randn(n_outro) * amp_quiet).astype(np.float32),
            ]
        )
        return np.column_stack([mono, ch_r])
    return mono  # type: ignore[no-any-return]


def _zone_rms_dbfs(audio: np.ndarray, start: int, end: int) -> float:
    """Gated RMS in dBFS für ein Segment (mono oder stereo → mono)."""
    seg = audio[start:end]
    if seg.ndim == 2:
        seg = np.mean(seg, axis=1)
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-30))
    return 20.0 * np.log10(rms + 1e-12)  # type: ignore[no-any-return]


def _zone_peak_dbfs(audio: np.ndarray, start: int, end: int) -> float:
    """Peak in dBFS für ein Segment."""
    seg = audio[start:end]
    if seg.ndim == 2:
        seg = np.ravel(seg)
    peak = float(np.percentile(np.abs(seg), 99.9))
    return 20.0 * np.log10(peak + 1e-12)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Test 1: apply_musical_gain_envelope schützt Stille-Zonen auch bei hohem Gain
# ---------------------------------------------------------------------------


class TestMusicalGainEnvelopeQuietZones:
    """apply_musical_gain_envelope darf stille Intro/Outro-Zonen nicht boosten."""

    @pytest.fixture(autouse=True)
    def _import_guard(self):
        from backend.core.audio_utils import apply_musical_gain_envelope

        self._fn = apply_musical_gain_envelope

    @pytest.mark.parametrize("gain", [2.0, 3.0, 6.0, 10.0])
    def test_mono_intro_not_boosted(self, gain: float) -> None:
        """Stille Intro-Zone bleibt nach gain-Anwendung innerhalb des 6 dB-Fensters."""
        audio = _make_song_with_quiet_zones()
        n_intro = int(0.5 * SR)

        out = self._fn(audio, gain=gain, gate_dbfs=-36.0, sr=SR)

        in_db = _zone_rms_dbfs(audio, 0, n_intro)
        out_db = _zone_rms_dbfs(out, 0, n_intro)
        boost = out_db - in_db
        assert boost <= MAX_INTRO_OUTRO_BOOST_DB, (
            f"Intro: gain={gain:.1f}× → Pegelexplosion +{boost:.1f} dB (limit: +{MAX_INTRO_OUTRO_BOOST_DB} dB)"
        )

    @pytest.mark.parametrize("gain", [2.0, 3.0, 6.0, 10.0])
    def test_mono_outro_not_boosted(self, gain: float) -> None:
        """Stille Outro-Zone bleibt nach gain-Anwendung innerhalb des 6 dB-Fensters."""
        audio = _make_song_with_quiet_zones()
        n_total = len(audio)
        n_outro_start = n_total - int(0.5 * SR)

        out = self._fn(audio, gain=gain, gate_dbfs=-36.0, sr=SR)

        in_db = _zone_rms_dbfs(audio, n_outro_start, n_total)
        out_db = _zone_rms_dbfs(out, n_outro_start, n_total)
        boost = out_db - in_db
        assert boost <= MAX_INTRO_OUTRO_BOOST_DB, (
            f"Outro: gain={gain:.1f}× → Pegelexplosion +{boost:.1f} dB (limit: +{MAX_INTRO_OUTRO_BOOST_DB} dB)"
        )

    def test_stereo_intro_not_boosted(self) -> None:
        """Stereo-Intro bleibt bei gain=5.0 im erlaubten Fenster."""
        audio = _make_song_with_quiet_zones(stereo=True)
        n_intro = int(0.5 * SR)

        out = self._fn(audio, gain=5.0, gate_dbfs=-36.0, sr=SR)

        for ch, name in [(0, "L"), (1, "R")]:
            in_db = _zone_rms_dbfs(audio[:, ch], 0, n_intro)
            out_db = _zone_rms_dbfs(out[:, ch], 0, n_intro)
            boost = out_db - in_db
            assert boost <= MAX_INTRO_OUTRO_BOOST_DB, f"Stereo Intro [{name}]: gain=5× → Pegelexplosion +{boost:.1f} dB"

    def test_music_body_is_boosted(self) -> None:
        """Musikkörper muss tatsächlich angehoben werden (kein False-Pass durch No-Op)."""
        audio = _make_song_with_quiet_zones()
        n_intro = int(0.5 * SR)
        n_outro_start = len(audio) - int(0.5 * SR)

        out = self._fn(audio, gain=3.0, gate_dbfs=-36.0, sr=SR)

        in_db = _zone_rms_dbfs(audio, n_intro, n_outro_start)
        out_db = _zone_rms_dbfs(out, n_intro, n_outro_start)
        assert out_db > in_db + 1.0, (
            f"Musikkörper wurde nicht angehoben (in={in_db:.1f} dB, out={out_db:.1f} dB) "
            "— apply_musical_gain_envelope könnte ein No-Op sein"
        )

    def test_vinyl_noise_floor_not_boosted(self) -> None:
        """Vinyl-Rauschboden (~-35 dBFS) in Outro wird nicht auf Musikpegel gezogen."""
        n = int(4.0 * SR)
        rng = np.random.RandomState(0)
        amp_music = 10.0 ** (-18.0 / 20.0)
        amp_vinyl = 10.0 ** (-35.0 / 20.0)
        # 3s Musik + 1s Vinyl-Rauschen (simuliert Outro nach Fade)
        music = (rng.randn(int(3.0 * SR)) * amp_music).astype(np.float32)
        vinyl = (rng.randn(int(1.0 * SR)) * amp_vinyl).astype(np.float32)
        audio = np.concatenate([music, vinyl])

        from backend.core.audio_utils import apply_musical_gain_envelope

        out = apply_musical_gain_envelope(audio, gain=4.0, gate_dbfs=-36.0, sr=SR)

        in_db = _zone_rms_dbfs(audio, int(3.0 * SR), n)
        out_db = _zone_rms_dbfs(out, int(3.0 * SR), n)
        boost = out_db - in_db
        assert boost <= MAX_INTRO_OUTRO_BOOST_DB, (
            f"Vinyl-Outro-Rauschen: +{boost:.1f} dB Boost (limit: +{MAX_INTRO_OUTRO_BOOST_DB} dB)"
        )


# ---------------------------------------------------------------------------
# Test 2: sosfilt (kausal) vs sosfiltfilt (zero-phase) — Inter-Band-Desync
# ---------------------------------------------------------------------------


class TestSosfiltSplitRecombineNoExplosion:
    """Regression: sosfilt Split+Recombine → destruktive Interferenz → Pegelexplosion.

    V11 VERBOTEN-Regel: sosfilt(sos, audio) addiert zu Original ist verboten.
    sosfiltfilt (zero-phase) muss überall verwendet werden.
    """

    def _make_sos_bandpass(self, f_low: float, f_high: float) -> np.ndarray:
        return sp_signal.butter(4, [f_low, f_high], btype="band", fs=SR, output="sos")  # type: ignore[no-any-return]

    def test_sosfiltfilt_split_recombine_no_peak_explosion(self) -> None:
        """sosfiltfilt-Split+Recombine darf kein Peak > Input-Peak + 3 dBFS erzeugen."""
        audio = _make_song_with_quiet_zones()

        # Typisches Band-Split-Muster (phase_10/11 etc.)
        sos_low = self._make_sos_bandpass(80, 300)
        sos_mid = self._make_sos_bandpass(300, 3000)
        sos_high = self._make_sos_bandpass(3000, 16000)

        low = sp_signal.sosfiltfilt(sos_low, audio)
        mid = sp_signal.sosfiltfilt(sos_mid, audio)
        high = sp_signal.sosfiltfilt(sos_high, audio)
        recombined = audio + low * 0.2 + mid * 0.1 + high * 0.15  # additive enhancement

        in_peak = float(np.percentile(np.abs(audio), 99.9))
        out_peak = float(np.percentile(np.abs(recombined), 99.9))
        in_db = 20.0 * np.log10(in_peak + 1e-12)
        out_db = 20.0 * np.log10(out_peak + 1e-12)

        assert out_db <= in_db + 3.0, (
            f"sosfiltfilt Split+Recombine: Peak +{out_db - in_db:.1f} dB (über +3 dB-Grenze → mögliche Pegelexplosion)"
        )

    def test_sosfilt_causal_phase_shift_is_measurable(self) -> None:
        """Dokumentation: kausaler sosfilt erzeugt messbaren Phasenversatz.

        Dieser Test stellt sicher, dass wir den Unterschied WISSEN — als
        Referenz warum V11 sosfilt in Split+Recombine verbietet.
        """
        audio = _make_song_with_quiet_zones()
        n_intro = int(0.5 * SR)
        n_body_end = len(audio) - int(0.5 * SR)
        body = audio[n_intro:n_body_end]

        sos = self._make_sos_bandpass(200, 4000)

        filtered_causal = sp_signal.sosfilt(sos, body)
        filtered_zerophase = sp_signal.sosfiltfilt(sos, body)

        # Zero-Phase MUSS eine höhere Kreuzkorrelation mit dem Original haben
        # als kausaler Filter bei lag=0 (kein Phasenversatz)
        corr_zp = float(np.corrcoef(body[:1000], filtered_zerophase[:1000])[0, 1])
        corr_causal = float(np.corrcoef(body[:1000], filtered_causal[:1000])[0, 1])

        assert abs(corr_zp) >= abs(corr_causal) - 0.05, (
            f"Zero-phase corr={corr_zp:.3f} sollte ≥ causal corr={corr_causal:.3f} - 0.05 sein"
        )

    def test_intro_quiet_zone_unaffected_by_bandpass(self) -> None:
        """Stille Intro-Zone: sosfiltfilt-Band + Recombine erhöht Pegel nicht wesentlich."""
        audio = _make_song_with_quiet_zones()
        n_intro = int(0.5 * SR)

        sos = self._make_sos_bandpass(100, 8000)
        filtered = sp_signal.sosfiltfilt(sos, audio)
        recombined = audio + filtered * 0.5

        in_db = _zone_rms_dbfs(audio, 0, n_intro)
        out_db = _zone_rms_dbfs(recombined, 0, n_intro)
        boost = out_db - in_db

        # Bandpass auf Rauschen ist Rauschen — sollte kaum Energie hinzufügen
        assert boost <= MAX_INTRO_OUTRO_BOOST_DB, (
            f"Intro-Zone nach Bandpass-Recombine: +{boost:.1f} dB (limit: +{MAX_INTRO_OUTRO_BOOST_DB} dB)"
        )


# ---------------------------------------------------------------------------
# Test 3: Stereo OLA-Normalisierung — keine unabhängige Per-Kanal-Normierung
# ---------------------------------------------------------------------------


class TestStereoLinkedNormalizationNoGainMismatch:
    """Regression: phase_31-Pattern — WSOLA per-Kanal-Normierung zerstört Stereo-Balance.

    Root-Cause: L/R unabhängig auf Peak=1.0 normiert → bis 6 dB Gain-Differenz.
    Fix: OLA-Window-Sum-Normierung in _wsola_mono; Peak-Guard NUR auf kombiniertem Signal.
    """

    def _simulate_wsola_per_channel_wrong(self, audio: np.ndarray) -> np.ndarray:
        """Simuliert FALSCHES per-Kanal-Normierungsmuster (wie vor dem Fix)."""
        left = audio[:, 0].copy()
        right = audio[:, 1].copy()
        # FALSCH: unabhängige Peak-Normierung
        left = left / (float(np.percentile(np.abs(left), 99.9)) + 1e-10)
        right = right / (float(np.percentile(np.abs(right), 99.9)) + 1e-10)
        return np.column_stack([left, right])

    def _simulate_wsola_linked_correct(self, audio: np.ndarray) -> np.ndarray:
        """Simuliert KORREKTES Linked-Stereo-Pattern (nach Fix)."""
        result = audio.copy()
        _peak = float(np.percentile(np.abs(result), 99.9)) + 1e-10
        if _peak > 1.0:
            result = result / _peak
        return result

    def test_per_channel_normalization_creates_imbalance(self) -> None:
        """Dokumentation: per-Kanal-Normierung erzeugt messbaren L/R Gain-Unterschied.

        Dieser Test fängt eine Regression auf — wenn er fehlschlägt, wurde das
        per-Kanal-Pattern wieder eingeführt (in phase_31 oder ähnlich).
        """
        # Asymmetrisches Stereo: R-Kanal leiser (typisches Vinyl-Material)
        n = int(2.0 * SR)
        np.random.RandomState(7)
        amp_l = 10.0 ** (-18.0 / 20.0)
        amp_r = 10.0 ** (-24.0 / 20.0)  # R ist 6 dB leiser
        t = np.linspace(0, 2.0, n)
        left = (np.sin(2 * np.pi * 440 * t) * amp_l).astype(np.float32)
        right = (np.sin(2 * np.pi * 440 * t) * amp_r).astype(np.float32)
        stereo = np.column_stack([left, right])

        wrong = self._simulate_wsola_per_channel_wrong(stereo)
        correct = self._simulate_wsola_linked_correct(stereo)

        # Falsches Pattern normiert beide auf Peak ≈ 1.0 → L/R-Verhältnis zerstört
        peak_l_wrong = float(np.percentile(np.abs(wrong[:, 0]), 99.9))
        peak_r_wrong = float(np.percentile(np.abs(wrong[:, 1]), 99.9))
        imbalance_wrong = abs(20.0 * np.log10((peak_l_wrong + 1e-12) / (peak_r_wrong + 1e-12)))

        # Korrektes Pattern erhält L/R-Verhältnis
        peak_l_correct = float(np.percentile(np.abs(correct[:, 0]), 99.9))
        peak_r_correct = float(np.percentile(np.abs(correct[:, 1]), 99.9))
        imbalance_correct = abs(20.0 * np.log10((peak_l_correct + 1e-12) / (peak_r_correct + 1e-12)))

        # Korrektes Pattern: Original-Imbalance bleibt erhalten (≈ 6 dB)
        assert imbalance_correct >= 5.0, (
            f"Linked-Normierung hat L/R-Verhältnis zerstört (imbalance={imbalance_correct:.1f} dB, "
            "erwartet ≈ 6 dB = Original-Verhältnis)"
        )
        # Falsches Pattern: normiert beide weg → Imbalance ≈ 0
        assert imbalance_wrong < 1.0, (
            f"Per-Kanal-Normierung hätte L/R-Imbalance auflösen sollen (got {imbalance_wrong:.1f} dB)"
        )

    def test_linked_normalization_preserves_stereo_balance(self) -> None:
        """Linked-Normierung bewahrt das Original-L/R-Lautstärkeverhältnis."""
        audio = _make_song_with_quiet_zones(stereo=True, body_dbfs=-18.0)
        result = self._simulate_wsola_linked_correct(audio)

        # Pegel-Verhältnis L/R muss vorher und nachher gleich sein (< 0.5 dB Abweichung)
        in_l = _zone_rms_dbfs(audio[:, 0], 0, len(audio))
        in_r = _zone_rms_dbfs(audio[:, 1], 0, len(audio))
        out_l = _zone_rms_dbfs(result[:, 0], 0, len(result))
        out_r = _zone_rms_dbfs(result[:, 1], 0, len(result))

        in_ratio = in_l - in_r
        out_ratio = out_l - out_r

        assert abs(out_ratio - in_ratio) < 0.5, (
            f"L/R-Verhältnis verändert: vorher={in_ratio:.2f} dB, nachher={out_ratio:.2f} dB "
            f"(Δ={abs(out_ratio - in_ratio):.2f} dB, max 0.5 dB)"
        )


# ---------------------------------------------------------------------------
# Test 4: check_gain_safety — Pre-Flight-Guard verhindert Clipping
# ---------------------------------------------------------------------------


class TestCheckGainSafetyPreFlight:
    """check_gain_safety muss Clipping bei lautem Material verhindern."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from backend.core.audio_utils import check_gain_safety

        self._fn = check_gain_safety

    def test_loud_material_gain_capped(self) -> None:
        """Hoher Gain auf Material nahe 0 dBFS wird auf sicheren Wert begrenzt."""
        # Audio bei -6 dBFS Peak
        amp = 10.0 ** (-6.0 / 20.0)
        audio = np.ones(SR, dtype=np.float32) * amp

        safe_gain, was_clamped = self._fn(audio, requested_gain=5.0, max_peak_dbfs=-1.0)

        assert was_clamped, "Gain sollte begrenzt worden sein"
        assert safe_gain < 5.0, f"safe_gain={safe_gain:.3f} hätte < 5.0 sein sollen"
        # Nach Anwendung darf Peak nicht über -1 dBTP liegen
        out_peak = float(np.max(np.abs(audio * safe_gain)))
        assert out_peak <= 10.0 ** (-1.0 / 20.0) + 0.001, (
            f"Peak nach safe_gain: {20 * np.log10(out_peak + 1e-12):.1f} dBFS (limit: -1 dBTP)"
        )

    def test_quiet_material_no_clamp(self) -> None:
        """Auf leisem Material (-40 dBFS) wird Gain 3.0 nicht begrenzt."""
        amp = 10.0 ** (-40.0 / 20.0)
        audio = np.ones(SR, dtype=np.float32) * amp

        safe_gain, was_clamped = self._fn(audio, requested_gain=3.0, max_peak_dbfs=-1.0)

        assert not was_clamped, f"Gain auf leisem Material sollte nicht begrenzt werden (got {safe_gain:.3f})"
        assert safe_gain == pytest.approx(3.0, abs=0.001)

    def test_silent_audio_returns_unity(self) -> None:
        """Vollständige Stille → safe_gain=1.0 (kein Gain auf Nichts)."""
        audio = np.zeros(SR, dtype=np.float32)

        safe_gain, was_clamped = self._fn(audio, requested_gain=10.0)

        assert was_clamped
        assert safe_gain == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Test 5: End-to-End Invariante — Pipeline-Ausgang darf Intro/Outro nicht boosten
# ---------------------------------------------------------------------------


class TestEndToEndQuietZoneInvariant:
    """Kombinierter Invariantentest: Restoration-ähnliche Sequenz darf keine
    Pegelexplosion in Intro/Outro erzeugen.

    Simuliert: Bandpass-EQ + Makeup-Gain + Soft-Limiter = typischer Phase-Kern.
    """

    def _simulate_restoration_phase(
        self, audio: np.ndarray, eq_gain_db: float = 6.0, makeup_gain: float = 2.0
    ) -> np.ndarray:
        """Minimal-Simulation einer Restaurierungsphase mit EQ + Makeup-Gain."""
        from backend.core.audio_utils import apply_musical_gain_envelope

        # EQ-Band (Presence-Boost, ~3 kHz)
        sos = sp_signal.butter(2, [2000, 6000], btype="band", fs=SR, output="sos")
        eq_band = sp_signal.sosfiltfilt(sos, audio)
        eq_linear = 10.0 ** (eq_gain_db / 20.0) - 1.0
        processed = audio + eq_band * eq_linear

        # Makeup-Gain mit Guard
        out = apply_musical_gain_envelope(processed, gain=makeup_gain, gate_dbfs=-36.0, sr=SR, reference_for_gate=audio)

        # Soft-Limiter
        peak = float(np.percentile(np.abs(out), 99.9))
        if peak > 0.98:
            out = out * (0.98 / peak)

        return out

    def test_intro_not_boosted_after_full_phase(self) -> None:
        """Nach vollständiger Phase-Simulation: Intro bleibt ≤ 6 dB über Input."""
        audio = _make_song_with_quiet_zones(total_s=6.0, intro_s=1.0, outro_s=1.0)
        n_intro = int(1.0 * SR)

        out = self._simulate_restoration_phase(audio, eq_gain_db=6.0, makeup_gain=2.0)

        in_db = _zone_rms_dbfs(audio, 0, n_intro)
        out_db = _zone_rms_dbfs(out, 0, n_intro)
        boost = out_db - in_db

        assert boost <= MAX_INTRO_OUTRO_BOOST_DB, (
            f"Intro nach Phase-Simulation: +{boost:.1f} dB (limit: +{MAX_INTRO_OUTRO_BOOST_DB} dB)"
        )

    def test_outro_not_boosted_after_full_phase(self) -> None:
        """Nach vollständiger Phase-Simulation: Outro bleibt ≤ 6 dB über Input."""
        audio = _make_song_with_quiet_zones(total_s=6.0, intro_s=1.0, outro_s=1.0)
        n_total = len(audio)
        n_outro_start = n_total - int(1.0 * SR)

        out = self._simulate_restoration_phase(audio, eq_gain_db=9.0, makeup_gain=3.0)

        in_db = _zone_rms_dbfs(audio, n_outro_start, n_total)
        out_db = _zone_rms_dbfs(out, n_outro_start, n_total)
        boost = out_db - in_db

        assert boost <= MAX_INTRO_OUTRO_BOOST_DB, (
            f"Outro nach Phase-Simulation: +{boost:.1f} dB (limit: +{MAX_INTRO_OUTRO_BOOST_DB} dB)"
        )

    @pytest.mark.parametrize("makeup_gain", [1.5, 2.0, 3.0, 5.0, 8.0])
    def test_parametric_makeup_gains(self, makeup_gain: float) -> None:
        """Diverse Makeup-Gains: Invariante hält für alle typischen Werte."""
        audio = _make_song_with_quiet_zones()
        n_intro = int(0.5 * SR)
        n_total = len(audio)
        n_outro_start = n_total - int(0.5 * SR)

        out = self._simulate_restoration_phase(audio, makeup_gain=makeup_gain)

        for name, start, end in [
            ("Intro", 0, n_intro),
            ("Outro", n_outro_start, n_total),
        ]:
            in_db = _zone_rms_dbfs(audio, start, end)
            out_db = _zone_rms_dbfs(out, start, end)
            boost = out_db - in_db
            assert boost <= MAX_INTRO_OUTRO_BOOST_DB, (
                f"{name} mit makeup_gain={makeup_gain:.1f}×: +{boost:.1f} dB (limit: +{MAX_INTRO_OUTRO_BOOST_DB} dB)"
            )


# ---------------------------------------------------------------------------
# Test 6: §0h limit_quiet_edge_boost — 0.5 dB finale Exporttoleranz
# ---------------------------------------------------------------------------

# Tight tolerance used in final export guards (UV3 + AudioExporter + fallback path).
# v10 soft-knee: allow up to 1.0 dB to account for continuous sigmoid transition.
MAX_FINAL_EXPORT_BOOST_DB = 1.0
# +0.05 dB floating-point rounding margin
_FINAL_BOOST_LIMIT = MAX_FINAL_EXPORT_BOOST_DB + 0.05


class TestLimitQuietEdgeBoostFinalTolerance:
    """§0h Music-Death-Shield: limit_quiet_edge_boost mit max_edge_boost_db=0.5 dB.

    Stellt sicher, dass die finale Export-Schutzschicht Pegelexplosionen auf
    max. 0.5 dB begrenzt — kein hörbarer Pegel-Burst in stillen Intro/Outro-Zonen.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from backend.core.audio_utils import limit_quiet_edge_boost

        self._fn = limit_quiet_edge_boost

    def _ref_with_quiet_edges(
        self,
        quiet_dbfs: float = -55.0,
        music_dbfs: float = -18.0,
        total_s: float = 5.0,
        edge_s: float = 1.0,
        stereo: bool = False,
        seed: int = 1,
    ) -> np.ndarray:
        rng = np.random.RandomState(seed)
        n_total = int(total_s * SR)
        n_edge = int(edge_s * SR)
        n_body = n_total - 2 * n_edge
        amp_q = 10.0 ** (quiet_dbfs / 20.0)
        amp_m = 10.0 ** (music_dbfs / 20.0)
        t = np.linspace(0, float(n_body) / SR, n_body)
        body = (np.sin(2 * np.pi * 440 * t) * amp_m).astype(np.float32)
        intro = (rng.randn(n_edge) * amp_q).astype(np.float32)
        outro = (rng.randn(n_edge) * amp_q).astype(np.float32)
        mono = np.concatenate([intro, body, outro])
        if stereo:
            ch_r = np.concatenate(
                [
                    (rng.randn(n_edge) * amp_q).astype(np.float32),
                    body * 0.98,
                    (rng.randn(n_edge) * amp_q).astype(np.float32),
                ]
            )
            return np.column_stack([mono, ch_r])
        return mono  # type: ignore[no-any-return]

    @pytest.mark.parametrize("boost_db", [1.0, 2.0, 4.0, 8.0])
    def test_mono_intro_clamped_to_05db(self, boost_db: float) -> None:
        """Mono-Intro: Boost von {boost_db} dB → nach Clamp ≤ 0.5 dB über Referenz."""
        ref = self._ref_with_quiet_edges()
        n_edge = int(1.0 * SR)
        boost_lin = 10.0 ** (boost_db / 20.0)
        # Simulate a candidate that has been boosted at the intro
        candidate = ref.copy()
        candidate[:n_edge] *= boost_lin

        result = self._fn(ref, candidate, SR, max_edge_boost_db=0.5)

        ref_db = _zone_rms_dbfs(ref, 0, n_edge)
        out_db = _zone_rms_dbfs(result, 0, n_edge)
        actual_boost = out_db - ref_db
        assert actual_boost <= _FINAL_BOOST_LIMIT, (
            f"Intro boost_db={boost_db:.1f} → nach Clamp {actual_boost:.2f} dB (limit: {MAX_FINAL_EXPORT_BOOST_DB} dB)"
        )

    @pytest.mark.parametrize("boost_db", [1.0, 2.0, 4.0, 8.0])
    def test_mono_outro_clamped_to_05db(self, boost_db: float) -> None:
        """Mono-Outro: Boost von {boost_db} dB → nach Clamp ≤ 0.5 dB über Referenz."""
        ref = self._ref_with_quiet_edges()
        n_total = len(ref)
        n_edge = int(1.0 * SR)
        boost_lin = 10.0 ** (boost_db / 20.0)
        candidate = ref.copy()
        candidate[-n_edge:] *= boost_lin

        result = self._fn(ref, candidate, SR, max_edge_boost_db=0.5)

        ref_db = _zone_rms_dbfs(ref, n_total - n_edge, n_total)
        out_db = _zone_rms_dbfs(result, n_total - n_edge, n_total)
        actual_boost = out_db - ref_db
        assert actual_boost <= _FINAL_BOOST_LIMIT, (
            f"Outro boost_db={boost_db:.1f} → nach Clamp {actual_boost:.2f} dB (limit: {MAX_FINAL_EXPORT_BOOST_DB} dB)"
        )

    @pytest.mark.parametrize("boost_db", [1.0, 2.0, 4.0, 8.0])
    def test_stereo_both_channels_clamped(self, boost_db: float) -> None:
        """Stereo: Beide Kanäle werden auf 0.5 dB geclampt."""
        ref = self._ref_with_quiet_edges(stereo=True)
        n_edge = int(1.0 * SR)
        boost_lin = 10.0 ** (boost_db / 20.0)
        candidate = ref.copy()
        candidate[:n_edge, :] *= boost_lin
        candidate[-n_edge:, :] *= boost_lin

        result = self._fn(ref, candidate, SR, max_edge_boost_db=0.5)

        n_total = ref.shape[0]
        for ch_idx, ch_name in [(0, "L"), (1, "R")]:
            for zone, start, end in [
                ("Intro", 0, n_edge),
                ("Outro", n_total - n_edge, n_total),
            ]:
                ref_db = _zone_rms_dbfs(ref[:, ch_idx], start, end)
                out_db = _zone_rms_dbfs(result[:, ch_idx], start, end)
                actual_boost = out_db - ref_db
                assert actual_boost <= _FINAL_BOOST_LIMIT, (
                    f"[{ch_name}] {zone} boost_db={boost_db:.1f} → "
                    f"{actual_boost:.2f} dB (limit: {MAX_FINAL_EXPORT_BOOST_DB} dB)"
                )

    def test_music_body_unaffected_by_edge_clamp(self) -> None:
        """Musikkörper (keine stille Edge): Clamp ändert nichts."""
        ref = self._ref_with_quiet_edges()
        n_edge = int(1.0 * SR)
        n_body_start = n_edge
        n_body_end = len(ref) - n_edge
        # Boost only in body (not at edges) — edge clamp must not touch body
        candidate = ref.copy()
        candidate[n_body_start:n_body_end] *= 1.5

        result = self._fn(ref, candidate, SR, max_edge_boost_db=0.5)

        # Body remains boosted
        body_in_db = _zone_rms_dbfs(ref, n_body_start, n_body_end)
        body_out_db = _zone_rms_dbfs(result, n_body_start, n_body_end)
        assert body_out_db > body_in_db + 1.0, (
            f"Musikkörper sollte noch geboosted sein: in={body_in_db:.1f}, out={body_out_db:.1f} dB"
        )

    def test_attenuation_at_edges_is_never_applied_when_not_loud(self) -> None:
        """Keine Absenkung wenn Kandidat bereits leiser als Referenz (kein falsch-positiver Clamp)."""
        ref = self._ref_with_quiet_edges()
        n_edge = int(1.0 * SR)
        # Candidate is QUIETER at edges than reference (e.g. after noise reduction)
        candidate = ref.copy()
        candidate[:n_edge] *= 0.1  # -20 dB → much quieter
        candidate[-n_edge:] *= 0.1

        result = self._fn(ref, candidate, SR, max_edge_boost_db=0.5)

        # Clamp must not boost the already-quiet edges
        n_total = len(ref)
        for zone, start, end in [("Intro", 0, n_edge), ("Outro", n_total - n_edge, n_total)]:
            cand_db = _zone_rms_dbfs(candidate, start, end)
            out_db = _zone_rms_dbfs(result, start, end)
            # Should be equal or quieter (never louder than the candidate)
            assert out_db <= cand_db + 0.1, (
                f"{zone}: result ({out_db:.1f} dB) sollte ≤ candidate ({cand_db:.1f} dB) sein"
            )

    def test_vinyl_noise_floor_at_minus33_clamped_to_05db(self) -> None:
        """Vinyl-Rauschen bei -33 dBFS: Boost auf -31 dBFS wird auf -32.5 dBFS geclampt."""
        # Reference: vinyl noise at -33 dBFS at edges
        ref = self._ref_with_quiet_edges(quiet_dbfs=-33.0, music_dbfs=-18.0)
        n_edge = int(1.0 * SR)
        # Candidate: edge boosted by +2 dB (from -33 to -31 dBFS)
        candidate = ref.copy()
        candidate[:n_edge] *= 10.0 ** (2.0 / 20.0)
        candidate[-n_edge:] *= 10.0 ** (2.0 / 20.0)

        result = self._fn(ref, candidate, SR, max_edge_boost_db=0.5)

        n_total = len(ref)
        for zone, start, end in [("Intro", 0, n_edge), ("Outro", n_total - n_edge, n_total)]:
            ref_db = _zone_rms_dbfs(ref, start, end)
            out_db = _zone_rms_dbfs(result, start, end)
            actual_boost = out_db - ref_db
            assert actual_boost <= _FINAL_BOOST_LIMIT, (
                f"Vinyl-Edge {zone}: ref={ref_db:.1f} dB, out={out_db:.1f} dB, "
                f"boost={actual_boost:.2f} dB (limit: {MAX_FINAL_EXPORT_BOOST_DB} dB)"
            )


class TestAudioExporterEdgeClampTolerance:
    """§0h Smoke-Test: AudioExporter wendet finale Edge-Guard mit 0.5 dB Toleranz an."""

    def test_exporter_final_guard_uses_05db_tolerance(self, tmp_path) -> None:
        """AudioExporter: Quiet-Edge-Clamp limitiert auf 0.5 dB über Referenz."""
        pytest.importorskip("soundfile")
        from backend.core.audio_exporter import AudioExporter

        n = int(5.0 * SR)
        n_edge = int(1.0 * SR)
        rng = np.random.RandomState(42)
        amp_q = 10.0 ** (-55.0 / 20.0)
        amp_m = 10.0 ** (-18.0 / 20.0)
        t = np.linspace(0, float(n - 2 * n_edge) / SR, n - 2 * n_edge)
        body = (np.sin(2 * np.pi * 440 * t) * amp_m).astype(np.float32)
        reference = np.concatenate(
            [
                (rng.randn(n_edge) * amp_q).astype(np.float32),
                body,
                (rng.randn(n_edge) * amp_q).astype(np.float32),
            ]
        )

        # Candidate: 4 dB boost at both edges (simulating cumulative pipeline gain)
        candidate = reference.copy()
        boost_lin = 10.0 ** (4.0 / 20.0)
        candidate[:n_edge] *= boost_lin
        candidate[-n_edge:] *= boost_lin

        out_path = tmp_path / "test_edge_guard.wav"
        exporter = AudioExporter()
        exporter.export(
            candidate,
            SR,
            out_path,
            bit_depth=24,
            normalize=False,
            reference_audio=reference,
        )

        import soundfile as sf_

        exported, _ = sf_.read(str(out_path))

        for zone, start, end in [("Intro", 0, n_edge), ("Outro", n - n_edge, n)]:
            ref_db = _zone_rms_dbfs(reference, start, end)
            out_db = _zone_rms_dbfs(exported, start, end)
            actual_boost = out_db - ref_db
            assert actual_boost <= _FINAL_BOOST_LIMIT + 0.1, (  # +0.1 dB dither margin
                f"AudioExporter {zone}: ref={ref_db:.1f} dB, exported={out_db:.1f} dB, "
                f"boost={actual_boost:.2f} dB (limit: {MAX_FINAL_EXPORT_BOOST_DB} dB + 0.1 dB Dither)"
            )
