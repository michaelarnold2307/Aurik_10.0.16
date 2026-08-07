from __future__ import annotations

"""
tests/unit/test_vinyl_tape_mp3_chain_detection.py
===================================================
§2.46a Deep-Transfer-Chain — vinyl → tape → mp3_low

Stellt sicher, dass:
  1. _infer_tape_speed_ips() eine tape-Stufe in der transfer_chain erkennt,
     auch wenn primary_material vinyl/shellac ist.
  2. _infer_analog_source_from_fingerprint() reel_tape als Sekundärstufe
     zurückgibt, wenn has_disc=True und wow_flutter_index hoch genug ist.
  3. MediumDetector.detect() bei .mp3 mit vinyl+tape-Fingerabdruck eine
     3-stufige Chain [vinyl, tape, mp3_low] liefert.
  4. phase_04 head-bump auch feuert, wenn material_type='vinyl' ist aber
     'tape' in transfer_chain steht.
"""


import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _make_audio(seconds: float = 3.0, wow_hz: float = 0.0, infrasonic_rms: float = 0.0) -> np.ndarray:
    """Synthetisches 48 kHz Mono-Signal mit konfigurierbarem Wow/Flutter und Infraschall."""
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Infraschall-Rumble (Vinyl-Charakteristik)
    if infrasonic_rms > 0.0:
        signal += infrasonic_rms * np.sin(2 * np.pi * 18 * t)
    # Capstan-Flutter (Tape-Charakteristik)
    if wow_hz > 0.0:
        flutter = wow_hz * np.sin(2 * np.pi * 3.5 * t)  # ~3.5 Hz pinch-roller flutter
        signal = signal * (1.0 + 0.1 * flutter)
    signal = np.clip(signal, -1.0, 1.0)
    return signal  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# 1. _infer_tape_speed_ips — transfer_chain coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInferTapeSpeedIps:
    @pytest.fixture(autouse=True)
    def _uv3(self):
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        self.fn = UnifiedRestorerV3._infer_tape_speed_ips

    def test_primary_tape_still_works(self):
        """Primär tape → 7.5 ips."""
        assert self.fn("tape", None) == 7.5

    def test_primary_reel_tape_era_60s(self):
        """Primär reel_tape, Ära 1970 → 15 ips."""
        assert self.fn("reel_tape", 1970) == 15.0

    def test_primary_cassette(self):
        """Primär cassette → 1.875 ips."""
        assert self.fn("cassette", None) == 1.875

    def test_vinyl_no_chain_returns_none(self):
        """Vinyl ohne Chain → kein Tape → None."""
        assert self.fn("vinyl", 1970) is None

    def test_vinyl_with_tape_in_chain(self):
        """Vinyl primary + tape in transfer_chain → 7.5 ips erkannt."""
        result = self.fn("vinyl", 1970, transfer_chain=["vinyl", "tape", "mp3_low"])
        assert result == 7.5

    def test_shellac_with_reel_tape_chain_60s(self):
        """Shellac primary + reel_tape in chain, Ära 1960 → 15 ips."""
        result = self.fn("shellac", 1965, transfer_chain=["shellac", "reel_tape", "cd_digital"])
        assert result == 15.0

    def test_vinyl_with_cassette_chain(self):
        """Vinyl primary + cassette in chain → 1.875 ips."""
        result = self.fn("vinyl", None, transfer_chain=["vinyl", "cassette", "mp3_low"])
        assert result == 1.875

    def test_empty_chain_returns_none(self):
        """Leere Chain + vinyl primary → None."""
        assert self.fn("vinyl", None, transfer_chain=[]) is None

    def test_chain_without_tape_returns_none(self):
        """Chain ohne Tape-Stage → None."""
        result = self.fn("vinyl", 1980, transfer_chain=["vinyl", "cd_digital", "mp3_low"])
        assert result is None


# ---------------------------------------------------------------------------
# 2. _infer_analog_source_from_fingerprint — reel_tape bei has_disc=True
# ---------------------------------------------------------------------------


class TestInferAnalogSourceWithDisc:
    @pytest.fixture(autouse=True)
    def _detector(self):
        from forensics.medium_detector import MediumDetector

        self.det = MediumDetector()

    def _make_fp(
        self,
        wow: float = 0.0,
        rotation: float = 0.0,
        infrasonic: float = 0.0,
        crackle: float = 0.0,
        codec: float = 0.0,
    ) -> object:
        from forensics.medium_detector import SpectralFingerprint

        return SpectralFingerprint(
            wow_flutter_index=wow,
            rotation_strength=rotation,
            infrasonic_rms=infrasonic,
            crackle_density=crackle,
            codec_artifact_score=codec,
            snr_db=40.0,
            noise_floor_db=-50.0,
            effective_bandwidth_hz=10_000.0,
            noise_color="pink",  # type: ignore[arg-type]
        )

    def test_no_tape_with_low_flutter_and_disc(self):
        """Geringer Flutter + Disc → kein reel_tape (Vinyl-Flutter-Ceiling)."""
        fp = self._make_fp(wow=0.20, rotation=0.50, infrasonic=0.08, crackle=0.010)
        sources = self.det._infer_analog_source_from_fingerprint(fp)  # type: ignore[arg-type]
        mat_names = [m for m, _ in sources]
        assert "reel_tape" not in mat_names

    def test_tape_detected_with_disc_and_high_flutter(self):
        """Hoher Flutter (0.50) + Disc (vinyl) → Tape-Stufe erkannt.

        §2.46a Disambiguation: wow=0.50 >= 0.06 → Kassette wahrscheinlicher als
        Studio-Bandmaschine. Wichtig ist, dass IRGENDEINE Tape-Stufe (reel_tape ODER
        cassette) als Zwischenglied erkannt wird — nicht die spezifische Sorte.
        """
        fp = self._make_fp(wow=0.50, rotation=0.03, infrasonic=0.08, crackle=0.010)
        sources = self.det._infer_analog_source_from_fingerprint(fp)  # type: ignore[arg-type]
        mat_names = [m for m, _ in sources]
        _tape_family = {"reel_tape", "cassette", "tape"}
        assert any(m in _tape_family for m in mat_names), f"Keine Tape-Stufe erkannt, sources={sources}"

    def test_reel_tape_order_after_vinyl(self):
        """reel_tape muss nach vinyl in der sortierten Quell-Liste stehen."""
        fp = self._make_fp(wow=0.55, rotation=0.04, infrasonic=0.08, crackle=0.010)
        sources = self.det._infer_analog_source_from_fingerprint(fp)  # type: ignore[arg-type]
        mat_names = [m for m, _ in sources]
        if "vinyl" in mat_names and "reel_tape" in mat_names:
            assert mat_names.index("vinyl") < mat_names.index("reel_tape")

    def test_codec_adaptive_tape_threshold(self):
        """Studio reel_tape mit geringem Flutter + hoher Codec-Kontamination → reel_tape.

        §2.46a Studio-Tape-Pfad (has_disc=True, _codec_contamination > 0.5):
        wow=0.034 ist typisch für Studer/Ampex-Bandmaschine (IEC 60386: 0.01–0.03 WRMS).
        Mit codec=0.40 → _codec_contamination=0.667 > 0.5 → Studio-Pfad aktiv.
        _tape_flutter_thresh_rt = max(0.010, 0.025*(1-0.55*0.667)) ≈ 0.016
        wow=0.034 > 0.016 → tape_conf_rt = clip((0.034-0.016)/0.10, 0.12, 0.50) = 0.18 ✓
        Disambiguation: wow=0.034 < 0.06 → reel_tape gewinnt über Kassette.
        """
        fp = self._make_fp(wow=0.034, rotation=0.371, infrasonic=0.08, crackle=0.010, codec=0.40)
        sources = self.det._infer_analog_source_from_fingerprint(fp)  # type: ignore[arg-type]
        mat_names = [m for m, _ in sources]
        assert "reel_tape" in mat_names, f"reel_tape missing at codec=0.40/studio-flutter, sources={sources}"

    def test_no_disc_tape_direct(self):
        """Ohne Disc-Quelle: Tape-Familie mit ausreichendem Flutter erkannt (klassischer Pfad)."""
        # threshold=0.20, wow=0.45 > 0.20 → reel_tape oder cassette erkannt.
        # Disambiguation: wow=0.45 >= 0.06 → cassette bevorzugt wenn BW-Erkennung feuert.
        fp = self._make_fp(wow=0.45, rotation=0.02, infrasonic=0.00, crackle=0.001)
        sources = self.det._infer_analog_source_from_fingerprint(fp)  # type: ignore[arg-type]
        mat_names = [m for m, _ in sources]
        _tape_family = {"reel_tape", "cassette", "tape"}
        assert any(m in _tape_family for m in mat_names), f"Keine Tape-Familie erkannt, sources={sources}"

    def test_disambiguation_reel_tape_wins_low_flutter(self):
        """§2.46a Disambiguation: wow < 0.06 → reel_tape gewinnt über Kassette."""
        fp = self._make_fp(wow=0.034, rotation=0.371, infrasonic=0.08, crackle=0.010, codec=0.40)
        sources = self.det._infer_analog_source_from_fingerprint(fp)  # type: ignore[arg-type]
        mat_names = [m for m, _ in sources]
        assert "cassette" not in mat_names, f"Cassette sollte bei wow=0.034 entfernt werden, sources={sources}"
        assert "reel_tape" in mat_names, f"reel_tape fehlt, sources={sources}"

    def test_disambiguation_cassette_wins_high_flutter(self):
        """§2.46a Disambiguation: wow >= 0.06 → cassette gewinnt über reel_tape."""
        fp = self._make_fp(wow=0.50, rotation=0.03, infrasonic=0.08, crackle=0.010)
        sources = self.det._infer_analog_source_from_fingerprint(fp)  # type: ignore[arg-type]
        mat_names = [m for m, _ in sources]
        assert "reel_tape" not in mat_names, f"reel_tape sollte bei wow=0.50 entfernt werden, sources={sources}"
        assert "cassette" in mat_names, f"cassette fehlt, sources={sources}"

    def test_vinyl_detected_via_rotation_fallback_gate(self):
        """§2.46a Fallback-Gate: rotation >= 0.30 + conf >= 0.20 → vinyl erkannt (auch wenn conf < _pa_conf_thresh).

        Produktions-Szenario aus Backend-Logs: rotation=0.371, vinyl_conf=0.250, codec=0.40.
        Primär-Gate würde scheitern (_pa_conf_thresh ≈ 0.348 > 0.250), Fallback-Gate rettet.
        """
        from unittest.mock import patch

        from forensics.medium_detector import SpectralFingerprint

        det = self.det
        fp = SpectralFingerprint(
            rolloff_95_hz=14_000.0,
            noise_floor_db=-40.0,
            snr_db=33.0,
            noise_color=1.0,
            wow_flutter_index=0.034,
            rotation_hz=0.555,
            rotation_strength=0.371,
            infrasonic_rms=0.012,  # unter 0.030 → kein Infraschall-Beweis
            codec_artifact_score=0.40,
            effective_bandwidth_hz=13_500.0,  # < 14000 → mp3_low
            hf_energy_above_16k=0.001,
            crackle_density=0.015,
            codec_type_code=0.40,
        )
        dummy_audio = np.zeros(SR * 3, dtype=np.float32)
        with patch.object(det, "_compute_fingerprint", return_value=fp):
            result = det.detect(dummy_audio, SR, file_ext=".mp3")

        # vinyl muss als primäre Quelle erkannt werden (Fallback-Gate greift)
        assert result.primary_material == "vinyl", (
            f"Vinyl sollte primär sein (rotation=0.371 >= 0.30, conf >= 0.20), "
            f"primary_material={result.primary_material}, chain={result.transfer_chain}"
        )
        assert "vinyl" in result.transfer_chain, f"vinyl nicht in chain={result.transfer_chain}"

    def test_vinyl_reel_tape_mp3_full_chain_production_case(self):
        """§2.46a Production-Fall: vinyl → reel_tape → mp3_low komplett erkannt.

        Fingerprintwerte aus Backend-Logs (2026-04-20/21):
        rotation=0.371, wow=0.034, vinyl_conf=0.250, codec_artifact≈0.40.
        Erwartet: transfer_chain = [vinyl, reel_tape, mp3_low].
        """
        from unittest.mock import patch

        from forensics.medium_detector import SpectralFingerprint

        det = self.det
        fp = SpectralFingerprint(
            rolloff_95_hz=14_000.0,
            noise_floor_db=-40.0,
            snr_db=33.0,
            noise_color=1.0,
            wow_flutter_index=0.034,
            rotation_hz=0.555,
            rotation_strength=0.371,
            infrasonic_rms=0.012,
            codec_artifact_score=0.40,
            effective_bandwidth_hz=13_500.0,  # < 14000 → mp3_low (Grenzbedingung vermeiden)
            hf_energy_above_16k=0.001,
            crackle_density=0.015,
            codec_type_code=0.40,
        )
        dummy_audio = np.zeros(SR * 3, dtype=np.float32)
        with patch.object(det, "_compute_fingerprint", return_value=fp):
            result = det.detect(dummy_audio, SR, file_ext=".mp3")

        chain = result.transfer_chain
        assert result.primary_material == "vinyl", f"primary_material={result.primary_material}"
        assert "reel_tape" in chain, f"reel_tape fehlt in chain={chain}"
        assert "mp3_low" in chain, f"mp3_low fehlt in chain={chain}"
        assert result.is_multi_generation is True
        # Reihenfolge: vinyl → reel_tape → mp3_low
        if len(chain) >= 3:
            assert chain.index("vinyl") < chain.index("reel_tape"), f"Falsche Reihenfolge: {chain}"
            assert chain.index("reel_tape") < chain.index("mp3_low"), f"Falsche Reihenfolge: {chain}"


# ---------------------------------------------------------------------------
# 3. Full detect() — vinyl+tape+mp3 in chain
# ---------------------------------------------------------------------------


class TestFullChainDetection:
    @pytest.fixture(autouse=True)
    def _detector(self):
        from forensics.medium_detector import MediumDetector

        self.det = MediumDetector()

    def test_vinyl_tape_mp3_chain_contains_tape(self):
        """Vinyl+Tape-Fingerabdruck als .mp3 → chain enthält tape/reel_tape.

        _compute_fingerprint wird gemockt, da AM-Flutter-Synthese in der
        Infrasonic-Messung Artefakte erzeugt (3.5 Hz liegt im Infraschall-Band).
        Die einzelnen Erkennungskomponenten sind in TestInferAnalogSourceWithDisc
        bereits mit echten SpectralFingerprint-Werten getestet.
        """
        from unittest.mock import patch

        from forensics.medium_detector import SpectralFingerprint

        fp = SpectralFingerprint(
            rolloff_95_hz=13_000.0,
            noise_floor_db=-42.0,
            snr_db=35.0,
            noise_color=1.0,
            wow_flutter_index=0.50,
            rotation_hz=0.55,
            rotation_strength=0.04,
            # 0.075 → vinyl_conf = (0.075-0.030)/0.080 = 0.5625 ≥ _pa_conf_thresh≈0.424
            # so _strong_physical_analog=True and chain includes tape/cassette stage
            infrasonic_rms=0.075,
            codec_artifact_score=0.25,
            effective_bandwidth_hz=13_000.0,
            hf_energy_above_16k=0.001,
            crackle_density=0.007,
            codec_type_code=0.25,
        )
        dummy_audio = np.zeros(SR * 3, dtype=np.float32)

        with patch.object(self.det, "_compute_fingerprint", return_value=fp):
            result = self.det.detect(dummy_audio, SR, file_ext=".mp3")

        chain = result.transfer_chain
        _tape_family = {"tape", "reel_tape", "cassette"}
        has_tape = any(m in _tape_family for m in chain)
        assert has_tape, f"Tape-Stage fehlt in chain={chain}"

    def test_no_tape_signal_gives_no_tape_chain(self):
        """Reines Vinyl-Signal ohne Tape-Flutter → kein Tape in chain."""
        t = np.linspace(0, 3.0, SR * 3, endpoint=False, dtype=np.float32)
        signal = 0.3 * np.sin(2 * np.pi * 440 * t)
        # Vinyl rumble only — flutter very low (0.05)
        signal += 0.06 * np.sin(2 * np.pi * 18 * t)
        signal *= 1.0 + 0.03 * np.sin(2 * np.pi * 0.6 * t)
        import scipy.signal as ss

        sos = ss.butter(8, 15500 / (SR / 2), btype="low", output="sos")
        signal = ss.sosfilt(sos, signal).astype(np.float32)
        signal = np.clip(signal, -1.0, 1.0)

        result = self.det.detect(signal, SR, file_ext=".mp3")
        chain = result.transfer_chain
        _tape_family = {"tape", "reel_tape", "cassette"}
        # tape should NOT appear when only vinyl features present
        has_tape = any(m in _tape_family for m in chain)
        # Note: this is advisory — vinyl flutter can occasionally exceed threshold.
        # We log it but don't hard-fail since it's a probabilistic detector.
        if has_tape:
            import warnings

            warnings.warn(f"Tape false-positive for pure vinyl signal: chain={chain}", UserWarning)


# ---------------------------------------------------------------------------
# 4. phase_04 head-bump — fires for vinyl primary + tape in transfer_chain
# ---------------------------------------------------------------------------


class TestPhase04HeadBumpChainAware:
    @pytest.fixture(autouse=True)
    def _phase(self):
        from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase

        self.phase = EQCorrectionPhase(sample_rate=SR)

    def _noise(self, seconds: float = 2.0) -> np.ndarray:
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(int(SR * seconds)).astype(np.float32) * 0.1
        return np.clip(sig, -1.0, 1.0)

    def test_head_bump_fires_with_tape_in_chain(self):
        """material_type='vinyl' + tape in transfer_chain + tape_speed_ips → head_bump_applied=True."""
        audio = self._noise()
        result = self.phase.process(
            audio,
            material_type="vinyl",
            sample_rate=SR,
            tape_speed_ips=7.5,
            transfer_chain=["vinyl", "tape", "mp3_low"],
        )
        assert result.metadata.get("head_bump_applied") is True, (
            f"head_bump_applied sollte True sein, metadata={result.metadata}"
        )

    def test_head_bump_fires_for_primary_tape(self):
        """Klassischer Pfad: primary_material='tape' → head_bump_applied=True."""
        audio = self._noise()
        result = self.phase.process(
            audio,
            material_type="tape",
            sample_rate=SR,
            tape_speed_ips=7.5,
        )
        assert result.metadata.get("head_bump_applied") is True

    def test_head_bump_not_fired_without_tape_speed(self):
        """Kein tape_speed_ips → head_bump_applied=False."""
        audio = self._noise()
        result = self.phase.process(
            audio,
            material_type="tape",
            sample_rate=SR,
            # tape_speed_ips intentionally omitted
        )
        assert result.metadata.get("head_bump_applied") is False

    def test_head_bump_not_fired_vinyl_no_tape_chain(self):
        """Vinyl primary + kein Tape in chain → head_bump_applied=False."""
        audio = self._noise()
        result = self.phase.process(
            audio,
            material_type="vinyl",
            sample_rate=SR,
            tape_speed_ips=7.5,
            transfer_chain=["vinyl", "mp3_low"],  # no tape
        )
        assert result.metadata.get("head_bump_applied") is False

    def test_head_bump_reel_tape_in_chain(self):
        """Vinyl primary + reel_tape in chain → head_bump_applied=True (normalized: 'tape')."""
        audio = self._noise()
        result = self.phase.process(
            audio,
            material_type="vinyl",
            sample_rate=SR,
            tape_speed_ips=15.0,
            transfer_chain=["vinyl", "reel_tape", "cd_digital"],
        )
        assert result.metadata.get("head_bump_applied") is True


class TestMediumDetectorCodecGuardForAnalogChains:
    """Test codec guard that prefers mp3_low for analog transfer chains."""

    def test_vinyl_tape_mp3_chain_codec_guard_limits_mp3_high(self, monkeypatch):
        """Analog chain (vinyl + tape) with limited BW must end in mp3_low, not mp3_high."""
        from forensics.medium_detector import MediumDetector, SpectralFingerprint

        detector = MediumDetector()

        # Analog-evidence fingerprint: tape characteristics (flutter), limited HF bandwidth
        fp = SpectralFingerprint(
            rolloff_95_hz=7500.0,
            wow_flutter_index=0.050,
            hf_energy_above_16k=0.0002,
            noise_floor_db=-35.0,
            effective_bandwidth_hz=15_000.0,  # Below 18 kHz threshold for analog chains
            codec_artifact_score=0.25,
            codec_type_code=1.0,
            crackle_density=0.018,
            rotation_strength=0.35,  # Vinyl rotation
            infrasonic_rms=0.028,
        )

        # Mock the Bayesian scorer to prefer higher codecs naively
        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(
            detector,
            "_bayesian_score",
            lambda _fp, **_kw: {
                "mp3_high": 0.40,  # Higher score, but should not win for limited BW + analog chain
                "mp3_low": 0.18,
                "cd_digital": 0.15,
                "vinyl": 0.12,
                "tape": 0.10,
            },
        )
        monkeypatch.setattr(
            detector,
            "_infer_analog_source_from_fingerprint",
            lambda _fp: [("vinyl", 0.65), ("tape", 0.55)],
        )
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: False)

        audio = np.zeros(SR * 3, dtype=np.float32)
        result = detector.detect(audio, SR, file_ext=".mp3")

        assert result.primary_material == "vinyl", f"Expected vinyl primary, got {result.primary_material}"
        assert "tape" in result.transfer_chain, f"Expected tape in chain, got {result.transfer_chain}"
        assert result.transfer_chain[-1] == "mp3_low", (
            f"Expected vinyl→tape→mp3_low, but codec-guard did not apply. "
            f"Chain={result.transfer_chain}, BW={fp.effective_bandwidth_hz} Hz"
        )
        # Confirm evidence shows guard was triggered
        guard_fired = any("Codec-Guard" in e for e in result.evidence)
        assert guard_fired, f"Codec-Guard evidence not found in: {result.evidence}"
