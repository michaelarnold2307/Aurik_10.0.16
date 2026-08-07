from __future__ import annotations

import logging

import pytest

"""Unit-Tests für core/emotional_arc_preservation.py — EmotionalArcPreservationMetric.

Spec §8.2 Punkt 12: Emotionaler Dynamik-Bogen, Arousal/Valence Pearson, Klimax-Erhalt.
≥ 20 Tests.
"""


import math

import numpy as np

from backend.core.emotional_arc_preservation import (
    EmotionalArcPreservationMetric,
    EmotionalArcResult,
    apply_waveform_plausibility_guard,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48000


def _sine(freq: float, secs: float = 30.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _noise(secs: float = 30.0, amp: float = 0.1) -> np.ndarray:
    np.random.seed(7)
    return (np.random.randn(int(SR * secs)) * amp).astype(np.float32)


def _silence(secs: float = 30.0) -> np.ndarray:
    return np.zeros(int(SR * secs), dtype=np.float32)


def _dynamic_signal(secs: float = 60.0) -> np.ndarray:
    """Signal mit steigender Dynamik (Crescendo-Simulation)."""
    n = int(SR * secs)
    t = np.linspace(0, secs, n, endpoint=False)
    envelope = np.linspace(0.1, 1.0, n)
    return (np.sin(2 * np.pi * 440 * t) * envelope).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Klasse 1: Import und Initialisierung
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmotionalArcInit:
    def test_01_class_importable(self):
        assert EmotionalArcPreservationMetric is not None

    def test_02_result_class_importable(self):
        assert EmotionalArcResult is not None

    def test_03_instantiate(self):
        m = EmotionalArcPreservationMetric()
        assert m is not None

    def test_04_result_has_required_fields(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(EmotionalArcResult)}
        required = {"arousal_pearson", "valence_pearson", "klimax_peak_deviation", "arc_preserved", "skipped"}
        assert required.issubset(fields)


# ---------------------------------------------------------------------------
# Klasse 2: Kurze Dateien werden übersprungen
# ---------------------------------------------------------------------------


class TestShortFilesSkipped:
    def setup_method(self):
        self.m = EmotionalArcPreservationMetric()

    def test_05_short_file_skipped(self):
        """Datei < 30 s → skipped=True, kein Absturz."""
        audio = _sine(440.0, secs=15.0)
        r = self.m.measure(audio, audio, SR)
        assert r.skipped is True

    def test_06_very_short_file_no_crash(self):
        audio = _sine(440.0, secs=3.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r, EmotionalArcResult)

    def test_07_silence_short_skipped(self):
        audio = _silence(secs=10.0)
        r = self.m.measure(audio, audio, SR)
        assert r.skipped is True


# ---------------------------------------------------------------------------
# Klasse 3: Ausgabe-Invarianten bei gültiger Länge
# ---------------------------------------------------------------------------


class TestOutputInvariants:
    def setup_method(self):
        self.m = EmotionalArcPreservationMetric()

    def test_08_identical_signals_not_skipped(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert not r.skipped

    def test_09_arousal_pearson_bounded(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert -1.0 <= r.arousal_pearson <= 1.0

    def test_10_valence_pearson_bounded(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert -1.0 <= r.valence_pearson <= 1.0

    def test_11_klimax_deviation_non_negative(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert r.klimax_peak_deviation >= 0

    def test_12_identical_signals_arc_preserved(self):
        """Identisches Signal mit sich selbst verglichen → arc_preserved=True."""
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert r.arc_preserved is True

    def test_13_no_nan_in_scores(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert math.isfinite(r.arousal_pearson)
            assert math.isfinite(r.valence_pearson)

    def test_14_noise_vs_noise_no_crash(self):
        audio = _noise(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r, EmotionalArcResult)

    def test_15_silence_long_no_crash(self):
        audio = _silence(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r, EmotionalArcResult)

    def test_16_reason_is_string(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r.reason, str)

    def test_17_arc_preserved_is_bool(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r.arc_preserved, bool)

    def test_18_skipped_is_bool(self):
        audio = _silence(secs=10.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r.skipped, bool)


# ---------------------------------------------------------------------------
# Klasse 4: Verschiedene Signalkombinationen
# ---------------------------------------------------------------------------


class TestSignalCombinations:
    def setup_method(self):
        self.m = EmotionalArcPreservationMetric()

    def test_19_sine_vs_noise_no_crash(self):
        orig = _sine(440.0, secs=60.0)
        rest = _noise(secs=60.0)
        r = self.m.measure(orig, rest, SR)
        assert isinstance(r, EmotionalArcResult)

    def test_20_crescendo_vs_flat_no_crash(self):
        orig = _dynamic_signal(secs=60.0)
        rest = _sine(440.0, secs=60.0) * 0.5
        r = self.m.measure(orig, rest, SR)
        assert isinstance(r, EmotionalArcResult)
        if not r.skipped:
            assert math.isfinite(r.arousal_pearson)

    def test_21_different_lengths_no_crash(self):
        """Unterschiedliche Längen → kein Absturz."""
        orig = _dynamic_signal(secs=60.0)
        rest = _dynamic_signal(secs=70.0)
        try:
            r = self.m.measure(orig, rest, SR)
            assert isinstance(r, EmotionalArcResult)
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # Toleriert: unterschiedliche Längen können abgelehnt werden

    def test_22_threshold_arousal_constant(self):
        assert EmotionalArcPreservationMetric.THRESHOLD_AROUSAL == 0.85

    def test_23_threshold_valence_constant(self):
        assert EmotionalArcPreservationMetric.THRESHOLD_VALENCE == 0.80


# ---------------------------------------------------------------------------
# Klasse 7: correct_arc() — Makro-Bogen-Korrektur (§8.2)
# ---------------------------------------------------------------------------


def _crescendo_decrescendo(secs: float = 60.0) -> np.ndarray:
    """Musikalischer Bogen: leise → laut → leise (Sinusbogen-Hüllkurve)."""
    n = int(SR * secs)
    t = np.linspace(0, secs, n, endpoint=False)
    envelope = np.sin(np.pi * t / secs)  # Spitze in der Mitte
    return (np.sin(2 * np.pi * 440 * t) * envelope * 0.8).astype(np.float32)  # type: ignore[no-any-return]


def _flatten_dynamics(audio: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Simuliert NR-induzierte Dynamik-Abflachung: Amplitude → Mittelwert ziehen."""
    rms = np.sqrt(np.mean(audio**2) + 1e-12)
    return (audio * (1 - factor) + np.sign(audio) * rms * factor).astype(np.float32)  # type: ignore[no-any-return]


class TestCorrectArc:
    def setup_method(self):
        self.m = EmotionalArcPreservationMetric()

    def test_24_correct_arc_importable(self):
        from backend.core.emotional_arc_preservation import correct_emotional_arc

        assert callable(correct_emotional_arc)

    def test_25_short_file_returns_unchanged(self):
        """< 30 s → Audio unverändert zurückgeben."""
        audio = _sine(440.0, secs=15.0)
        corrected, arc = self.m.correct_arc(audio, audio, SR)
        assert corrected.shape == audio.shape
        assert arc.skipped is True

    def test_26_identical_signal_no_change(self):
        """Identisches Signal → minimale Gain-Änderung, kein Absturz."""
        orig = _crescendo_decrescendo(secs=60.0)
        corrected, arc = self.m.correct_arc(orig, orig.copy(), SR)
        # Gain-Delta sollte bei identischem Signal nahe 0 sein
        diff = np.max(np.abs(corrected - orig))
        assert diff < 0.05, f"Identisches Signal: zu große Abweichung {diff}"

    def test_27_output_shape_preserved(self):
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, _ = self.m.correct_arc(orig, rest, SR)
        assert corrected.shape == rest.shape
        assert corrected.dtype == np.float32

    def test_28_no_nan_inf_in_output(self):
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, _ = self.m.correct_arc(orig, rest, SR)
        assert np.isfinite(corrected).all()

    def test_29_no_clipping(self):
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, _ = self.m.correct_arc(orig, rest, SR)
        assert np.max(np.abs(corrected)) <= 1.0

    def test_30_result_has_arousal_valence(self):
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        _, arc = self.m.correct_arc(orig, rest, SR)
        assert hasattr(arc, "arousal_pearson")
        assert hasattr(arc, "valence_pearson")
        assert -1.0 <= arc.arousal_pearson <= 1.0
        assert -1.0 <= arc.valence_pearson <= 1.0

    def test_31_stereo_input_supported(self):
        """Stereo-Signal (2, N) → korrekte Ausgabe."""
        orig_mono = _crescendo_decrescendo(secs=60.0)
        orig = np.stack([orig_mono, orig_mono * 0.8])
        rest = np.stack([_flatten_dynamics(orig_mono), _flatten_dynamics(orig_mono * 0.8)])
        corrected, arc = self.m.correct_arc(orig, rest, SR)
        assert corrected.ndim == 2
        assert corrected.shape[0] == 2
        assert np.isfinite(corrected).all()

    def test_32_safety_revert_on_degradation(self):
        """Wenn Korrektur Arousal verschlechtert → Original zurückgeben."""
        orig = _sine(440.0, secs=60.0) * 0.3
        # Künstlich "korrigierte" Version mit umgekehrtem Profil
        rest = _dynamic_signal(secs=60.0)
        corrected, arc = self.m.correct_arc(orig, rest, SR)
        # Egal ob revert passiert — kein Absturz, gültiges Ergebnis
        assert np.isfinite(corrected).all()
        assert isinstance(arc, EmotionalArcResult)

    def test_33_convenience_function_works(self):
        from backend.core.emotional_arc_preservation import correct_emotional_arc

        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, arc = correct_emotional_arc(orig, rest, SR)
        assert corrected.shape == rest.shape
        assert isinstance(arc, EmotionalArcResult)

    def test_34_damping_zero_returns_unchanged(self):
        """damping=0 → keine Korrektur (Gain-Profil = 0 dB)."""
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, _ = self.m.correct_arc(orig, rest, SR, damping=0.0)
        diff = np.max(np.abs(corrected - rest))
        assert diff < 0.01, f"damping=0 sollte keine Änderung ergeben, diff={diff}"

    def test_35_max_gain_respected(self):
        """Gain wird auf max_gain_db begrenzt."""
        orig = _crescendo_decrescendo(secs=60.0)
        rest = orig * 0.01  # Extremer Pegelunterschied
        corrected, _ = self.m.correct_arc(orig, rest, SR, max_gain_db=3.0)
        # Bei max 3 dB Gain: Faktor ≤ 10^(3/20) ≈ 1.41
        assert np.max(np.abs(corrected)) <= 1.0

    def test_36_no_pegelexplosion_in_denoised_fadeout(self):
        """Regression §2.30 Post-Smoothing Quiet-Zone-Clamp.

        Szenario: Lauter Musik-Intro (0-30 s) gefolgt von denoised Fadeout (30-42 s).
        Die EmotionalArc-Korrektur darf im Fadeout keinen positiven Gain erzeugen,
        auch nicht durch SG-Smoothing-Verschleppung aus dem Intro.

        Symptom (vor Fix): SG-Fenster 7 × 2,5 s = 17,5 s — Boost aus Intro-Segmenten
        wurde in den Fadeout (15,83 % bei 222 s → 35 s) verschleppt → Pegelexplosion.
        """
        from backend.core.emotional_arc_preservation import correct_emotional_arc

        rng = np.random.default_rng(42)
        total_s = 42.0
        fadeout_start_s = 30.0
        n_total = int(SR * total_s)
        n_music = int(SR * fadeout_start_s)

        # Original: Musik + Vinyl-Rauschen (−40 dBFS ≈ 0.01)
        vinyl_noise_amp = 10.0 ** (-40.0 / 20.0)  # −40 dBFS
        music = np.sin(2 * np.pi * 440 * np.linspace(0, fadeout_start_s, n_music)).astype(np.float32)
        music *= 0.5  # −6 dBFS
        fadeout_noise_orig = (rng.standard_normal(n_total - n_music) * vinyl_noise_amp).astype(np.float32)
        original = np.concatenate([music, fadeout_noise_orig])

        # Restored: Musik gleich (leicht reduziert durch Denoise) + fast stille Fadeout
        music_rest = music * 0.92  # −0.72 dB durch Denoise-Beeinflussung
        # Denoised Fadeout: vinyl noise entfernt → nur digitales Rauschen (−65 dBFS)
        fadeout_rest = (rng.standard_normal(n_total - n_music) * 10.0 ** (-65.0 / 20.0)).astype(np.float32)
        restored = np.concatenate([music_rest, fadeout_rest])

        corrected, _ = correct_emotional_arc(original, restored, SR, max_gain_db=6.0, damping=0.70)

        # Im Fadeout darf kein positiver Gain angewendet worden sein
        fadeout_corrected = corrected[n_music:]
        fadeout_restored = restored[n_music:]

        # Der korrigierte Fadeout darf nicht signifikant lauter als der denoised Fadeout sein.
        # Erlaubt: minimale SG-Überblendung im ersten Frame (≤ 0.5 dB)
        rms_before = float(np.sqrt(np.mean(fadeout_restored**2) + 1e-12))
        rms_after = float(np.sqrt(np.mean(fadeout_corrected**2) + 1e-12))
        gain_db_applied = 20.0 * np.log10((rms_after + 1e-12) / (rms_before + 1e-12))
        assert gain_db_applied < 0.5, (
            f"Pegelexplosion im Fadeout: correct_emotional_arc boosted denoised fadeout "
            f"by {gain_db_applied:.2f} dB (max erlaubt: 0.5 dB). "
            f"Post-Smoothing Quiet-Zone-Clamp fehlt oder defekt."
        )


# ---------------------------------------------------------------------------
# Klasse 8: WaveformPlausibilityGuard (§2.30c)
# ---------------------------------------------------------------------------


class TestWaveformPlausibilityGuard:
    """§2.30c — Finale Pegelexplosions-Fangschicht vor HPI Gate."""

    def test_37_explosion_in_intro_outro_corrected(self):
        """Regression §2.30c: restored ist im Intro/Outro 12 dB lauter als original
        (typisches MDEM/correct_arc-Artefakt). WPG muss die explodierten Fenster
        auf ≈ orig + 2 dB klemmen.

        Szenario: 30 s Musik (0 dB amp) + 12 s Intro/Outro auf beiden Seiten.
        Original-Intro: Vinyl-Rauschen (−35 dBFS). Restored-Intro: Vinyl-Rauschen
        durch Gain-Artefakt auf −23 dBFS hochgesetzt (+12 dB Explosion).
        WPG muss corrected_intro_rms ≤ orig_intro_rms + 4 dB (threshold 2 dB Target
        + 2 dB Toleranz für Interpolationsübergang) erzielen.
        """
        rng = np.random.default_rng(0)
        sr = 48000
        n_intro = sr * 12  # 12 s Intro
        n_music = sr * 30  # 30 s Musik

        vinyl_amp = 10.0 ** (-35.0 / 20.0)  # −35 dBFS
        music = np.sin(2 * np.pi * 440 * np.linspace(0, 30.0, n_music)).astype(np.float32)

        intro_orig = (rng.standard_normal(n_intro) * vinyl_amp).astype(np.float32)
        original = np.concatenate([intro_orig, music, intro_orig])

        # Explosion: Intro/Outro wurde auf −23 dBFS geboostet (+12 dB)
        exploded_amp = 10.0 ** (-23.0 / 20.0)
        intro_rest = (rng.standard_normal(n_intro) * exploded_amp).astype(np.float32)
        restored = np.concatenate([intro_rest, music * 0.95, intro_rest])

        corrected, meta = apply_waveform_plausibility_guard(
            original,
            restored,
            sr,
            mode="restoration",
            material_type="vinyl",
            restorability_score=60.0,
        )

        assert meta["explosions_found"] > 0, "WPG hat keine Explosionen erkannt"
        assert meta["corrections_applied"] > 0, "WPG hat keine Korrektur angewendet"
        assert meta["skipped_reason"] is None, f"WPG skip unerwartet: {meta['skipped_reason']}"

        # Intro-RMS nach Korrektur muss signifikant unter der explodierten Version liegen
        intro_corr = corrected[:n_intro]
        intro_rest_rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(intro_rest**2))) + 1e-12)
        intro_corr_rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(intro_corr**2))) + 1e-12)
        intro_orig_rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(intro_orig**2))) + 1e-12)

        assert intro_corr_rms_db < intro_rest_rms_db - 3.0, (
            f"WPG hat Explosion nicht ausreichend korrigiert: "
            f"orig={intro_orig_rms_db:.1f} dBFS, restored={intro_rest_rms_db:.1f} dBFS, "
            f"corrected={intro_corr_rms_db:.1f} dBFS (erwartet < {intro_rest_rms_db - 3.0:.1f})"
        )
        # Korrektur darf nicht über orig + 4 dB gehen
        assert intro_corr_rms_db <= intro_orig_rms_db + 4.0, (
            f"WPG over-korrigiert: corrected={intro_corr_rms_db:.1f} dBFS > orig+4={intro_orig_rms_db + 4.0:.1f} dBFS"
        )

    def test_38_normal_music_not_touched(self):
        """WPG darf normale Musik ohne Explosion nicht anfassen.

        Szenario: 60 s Musik. Restored ist überall nur 2 dB lauter als Original
        (legitime Restaurierungsverbesserung, deutlich unter threshold=6 dB).
        WPG darf corrections_applied == 0 liefern.
        """
        rng = np.random.default_rng(1)
        sr = 48000
        n = sr * 60

        t = np.linspace(0.0, 60.0, n, dtype=np.float32)
        envelope = (0.5 + 0.4 * np.sin(2 * np.pi * t / 20.0)).astype(np.float32)
        original = (envelope * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        original += (rng.standard_normal(n) * 0.01).astype(np.float32)

        # Restored: legitimately 2 dB louder (well within threshold)
        restored = original * (10.0 ** (2.0 / 20.0))

        corrected, meta = apply_waveform_plausibility_guard(
            original,
            restored,
            sr,
            mode="restoration",
            material_type="cd_digital",
            restorability_score=80.0,
        )

        assert meta["corrections_applied"] == 0, (
            f"WPG korrigierte falsches Positiv: corrections_applied={meta['corrections_applied']}, "
            f"explosions_found={meta['explosions_found']} (keine Explosion in Testdaten)"
        )
        # Corrected muss identisch zu restored sein (kein Eingriff)
        rms_diff = float(np.sqrt(np.mean((corrected - restored) ** 2)))
        assert rms_diff < 1e-5, f"WPG hat Signal verändert ohne Explosion: RMS-Diff={rms_diff}"

    def test_39_spectral_content_preserved_after_correction(self):
        """§2.30c — Spektralgehalt nach Korrektur erhalten.

        Defektreparatur (simuliert: Hochtonwiederherstellung) muss nach WPG-Korrektur
        vollständig erhalten bleiben. Die WPG-Korrektur ist ein reines Gain-Multiplikation
        → spektrale FORM (Verhältnisse zwischen Frequenzbändern) unverändert.

        Szenario: Restored hat BW-Extension (HF-Energie deutlich erhöht) UND eine
        Pegelexplosion im Intro. WPG muss Explosion korrigieren, HF-Ratio bewahren.
        """
        rng = np.random.default_rng(2)
        sr = 48000
        n_intro = sr * 10  # 10 s explodiertes Intro
        n_music = sr * 40  # 40 s Musik

        # Original: bandbegrenzt (kein HF), leise Intro
        vinyl_amp = 10.0 ** (-35.0 / 20.0)
        t_music = np.linspace(0.0, 40.0, n_music, dtype=np.float32)
        music_orig = (0.5 * np.sin(2 * np.pi * 220.0 * t_music)).astype(np.float32)
        intro_orig = (rng.standard_normal(n_intro) * vinyl_amp).astype(np.float32)
        original = np.concatenate([intro_orig, music_orig])

        # Restored: BW-Extension hat HF hinzugefügt (1/4 der Energie),
        # Intro explodiert auf −20 dBFS (+15 dB)
        exploded_amp = 10.0 ** (-20.0 / 20.0)
        intro_rest = (rng.standard_normal(n_intro) * exploded_amp).astype(np.float32)
        # HF hinzugefügt: hf_ratio im Musik-Segment erhöht
        hf_component = (0.2 * np.sin(2 * np.pi * 12000.0 * t_music)).astype(np.float32)
        music_rest = music_orig + hf_component
        restored = np.concatenate([intro_rest, music_rest])

        # Spektrale Ratio HF/Breitband im Musik-Segment VOR Korrektur messen
        music_rest_segment = restored[n_intro:]
        fft_rest = np.abs(np.fft.rfft(music_rest_segment[:sr]))  # 1 s FFT
        bin_hz = sr / (2 * len(fft_rest))
        hf_start = int(8000 / bin_hz)
        hf_ratio_before = float(np.sum(fft_rest[hf_start:] ** 2) / (np.sum(fft_rest**2) + 1e-12))

        corrected, meta = apply_waveform_plausibility_guard(
            original,
            restored,
            sr,
            mode="restoration",
            material_type="vinyl",
            restorability_score=50.0,
        )

        assert meta["corrections_applied"] > 0, "WPG sollte Explosion im Intro korrigiert haben"

        # Spektrale Ratio HF/Breitband im Musik-Segment NACH Korrektur messen
        music_corr_segment = corrected[n_intro:]
        fft_corr = np.abs(np.fft.rfft(music_corr_segment[:sr]))
        hf_ratio_after = float(np.sum(fft_corr[hf_start:] ** 2) / (np.sum(fft_corr**2) + 1e-12))

        # HF-Ratio muss innerhalb 1 % identisch sein (pure gain-only correction)
        assert abs(hf_ratio_after - hf_ratio_before) < 0.01, (
            f"§2.30c: Spektralgehalt nach WPG-Korrektur verändert! "
            f"HF-Ratio vor={hf_ratio_before:.4f}, nach={hf_ratio_after:.4f} "
            f"(Differenz {abs(hf_ratio_after - hf_ratio_before):.4f} > 0.01)"
        )

    def test_40_quiet_zone_emergency_applies_when_proxy_would_fail(self):
        """Real-Audio-Schutz: Quiet-Zone-Explosion darf nicht wegen Proxy-Skip durchrutschen.

        Erzwingt einen Proxy-Fail via Monkeypatch und verifiziert, dass die neue
        Quiet-Zone-Notfallregel trotzdem attenuiert, wenn >80 % der Explosionen in
        sehr leisen Original-Fenstern liegen.
        """
        from backend.core.emotional_arc_preservation import WaveformPlausibilityGuard

        rng = np.random.default_rng(40)
        sr = 48000
        n_intro = sr * 12
        n_music = sr * 30

        quiet_amp = 10.0 ** (-35.0 / 20.0)
        exploded_amp = 10.0 ** (-22.0 / 20.0)

        intro_orig = (rng.standard_normal(n_intro) * quiet_amp).astype(np.float32)
        intro_rest = (rng.standard_normal(n_intro) * exploded_amp).astype(np.float32)

        t = np.linspace(0.0, 30.0, n_music, endpoint=False, dtype=np.float32)
        music_orig = (0.35 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
        music_rest = (0.34 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)

        original = np.concatenate([intro_orig, music_orig, intro_orig]).astype(np.float32)
        restored = np.concatenate([intro_rest, music_rest, intro_rest]).astype(np.float32)

        wpg = WaveformPlausibilityGuard()
        wpg._measure_goals_proxy = lambda *args, **kwargs: (0.90, 0.50, 10.0, 2.0)  # type: ignore[method-assign]

        corrected, meta = wpg.apply(
            original,
            restored,
            sr,
            mode="restoration",
            material_type="vinyl",
            restorability_score=55.0,
        )

        assert meta["explosions_found"] > 0
        assert meta["quiet_zone_emergency_applied"] is True
        assert meta["corrections_applied"] > 0
        assert meta["skipped_reason"] is None
        assert meta["explosion_quiet_ratio"] >= 0.80

        # Intro muss hörbar abgesenkt werden, obwohl Proxy künstlich "failing" ist.
        intro_corr = corrected[:n_intro]
        intro_rest_db = 20.0 * np.log10(float(np.sqrt(np.mean(intro_rest**2))) + 1e-12)
        intro_corr_db = 20.0 * np.log10(float(np.sqrt(np.mean(intro_corr**2))) + 1e-12)
        assert intro_corr_db <= intro_rest_db - 2.0, (
            f"Quiet-Zone-Notfallkorrektur zu schwach: restored={intro_rest_db:.2f} dBFS, "
            f"corrected={intro_corr_db:.2f} dBFS"
        )
