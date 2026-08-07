from __future__ import annotations

import pytest

"""Unit-Tests für MediumDetectionResult — neue Felder §6.7 (v10.0.0)."""


from forensics.medium_detector import MediumDetectionResult, SpectralFingerprint


def _make_result(**kwargs) -> MediumDetectionResult:
    """Helper — minimales MediumDetectionResult."""
    defaults = {
        "transfer_chain": ["vinyl"],
        "is_multi_generation": False,
        "primary_material": "vinyl",
        "confidence": 0.95,
        "spectral_fingerprint": SpectralFingerprint(
            effective_bandwidth_hz=15000.0,
            noise_floor_db=-55.0,
            rolloff_95_hz=14000.0,
            wow_flutter_index=0.01,
        ),
    }
    defaults.update(kwargs)
    return MediumDetectionResult(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestMediumDetectionResultNewFields:
    """§6.7 — tape_speed_ips, riaa_curve_type, riaa_curve_confidence."""

    def test_defaults(self):
        r = _make_result()
        assert r.tape_speed_ips is None
        assert r.riaa_curve_type == "unknown"
        assert r.riaa_curve_confidence == 0.0

    def test_tape_speed_ips_set(self):
        r = _make_result(tape_speed_ips=7.5)
        assert r.tape_speed_ips == 7.5

    def test_riaa_curve_type_set(self):
        r = _make_result(riaa_curve_type="columbia", riaa_curve_confidence=0.85)
        assert r.riaa_curve_type == "columbia"
        assert r.riaa_curve_confidence == 0.85

    def test_as_dict_contains_new_fields(self):
        r = _make_result(tape_speed_ips=15.0, riaa_curve_type="nab", riaa_curve_confidence=0.72)
        d = r.as_dict()
        assert d["tape_speed_ips"] == 15.0
        assert d["riaa_curve_type"] == "nab"
        assert d["riaa_curve_confidence"] == 0.72

    def test_as_dict_defaults(self):
        d = _make_result().as_dict()
        assert d["tape_speed_ips"] is None
        assert d["riaa_curve_type"] == "unknown"
        assert d["riaa_curve_confidence"] == 0.0

    def test_dolby_fields_still_present(self):
        """Backward compat — bestehende Felder unverändert."""
        r = _make_result(dolby_nr_type="dolby_b", dolby_nr_confidence=0.9)
        assert r.dolby_nr_type == "dolby_b"
        d = r.as_dict()
        assert d["dolby_nr_type"] == "dolby_b"
        assert d["dolby_nr_confidence"] == 0.9
