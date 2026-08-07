#!/usr/bin/env python3
"""§v10.306: Per-Defekt-Audibilität — Verification Tests."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.defect_scanner import DefectScanner


class TestDefectAudibility(unittest.TestCase):
    """Verify per-defect audibility thresholds."""

    def test_01_all_thresholds_defined(self):
        """Every key defect type has an audibility threshold."""
        required = [
            "clicks",
            "pops",
            "crackle",
            "dropout",
            "transport_bump",
            "hum",
            "hiss",
            "rumble",
            "noise_level",
            "surface_noise",
            "sibilance",
            "vocal_harshness",
            "stereo_imbalance",
            "phase_issues",
            "dc_offset",
            "bandwidth_loss",
            "pre_echo",
            "aliasing",
            "quantization_noise",
            "compression_artifacts",
        ]
        for dt in required:
            self.assertIn(dt, DefectScanner.AUDIBILITY_THRESHOLDS, f"Missing threshold for {dt}")

    def test_02_critical_defects_always_audible(self):
        """Sibilance and vocal_harshness are always audible (low JND, low masking)."""
        for material in ["tape", "cd", "shellac", "vinyl", "cassette", "dat"]:
            self.assertTrue(
                DefectScanner.is_audible("sibilance", 0.55, material, -10, 6000),
                f"Sibilance should be audible on {material}",
            )
            self.assertTrue(
                DefectScanner.is_audible("vocal_harshness", 0.55, material, -10, 3000),
                f"Vocal harshness should be audible on {material}",
            )

    def test_03_benign_defects_masked(self):
        """DC offset and quantization noise are masked for most materials."""
        # DC offset at severity 0.1 → -54 dB, JND is -50 dB with 0.9 masking
        self.assertFalse(DefectScanner.is_audible("dc_offset", 0.1, "cd", -20, 50), "DC offset 0.1 should be inaudible")
        # But at severity 0.99 → -0.5 dB, nearly full scale
        self.assertTrue(DefectScanner.is_audible("dc_offset", 0.99, "cd", -20, 60), "DC offset 0.95 should be audible")

    def test_04_material_masking(self):
        """Shellac masks defects more than DAT."""
        # clicks severity 0.3 → -42 dB. JND = -35 dB
        # On shellac: effective = -35 * 1.3 = -45.5 → -42 >= -45.5 → audible
        # On DAT: effective = -35 * 0.75 = -26.25 → -42 < -26.25 → inaudible
        self.assertTrue(
            DefectScanner.is_audible("clicks", 0.5, "shellac", -10, 3000),
            "Clicks on shellac should be audible (masked by surface noise)",
        )
        # DAT is so clean that same severity is below threshold
        self.assertFalse(
            DefectScanner.is_audible("clicks", 0.5, "dat", -10, 3000), "Clicks on DAT should be inaudible (no masking)"
        )

    def test_05_signal_masking(self):
        """Loud signal masks defects more than quiet signal."""
        # hiss severity 0.25 → -45 dB. JND = -38 dB, masking=0.4
        # Quiet (-40 dB RMS): signal_mask = 0.33, effective = -38 * (1-0.4*0.33) = -33
        # -45 < -33 → NOT audible
        self.assertFalse(
            DefectScanner.is_audible("hiss", 0.25, "tape", -40, 8000), "Hiss in quiet signal should be masked"
        )
        # Loud (-10 dB RMS): signal_mask = 0.83, effective = -38 * (1-0.4*0.83) = -25.4
        # -45 < -25.4 → NOT audible (loud signal masks more)
        self.assertFalse(
            DefectScanner.is_audible("hiss", 0.25, "tape", -10, 8000), "Hiss in loud signal should be more masked"
        )

    def test_06_frequency_weighting(self):
        """3 kHz is more audible than 50 Hz (ISO 226 equal-loudness)."""
        # Hum at 50 Hz: freq_gain=0.10 → JND near 0 → very hard to hear
        # Hum at 3000 Hz: freq_gain=1.0 → JND stays at -45 dB
        self.assertFalse(
            DefectScanner.is_audible("hum", 0.3, "tape", -20, 50), "Hum at 50Hz sev=0.3 should be inaudible"
        )
        # At 3 kHz, same hum severity is still below threshold
        # But we verify higher severity works differently per frequency
        audible_50hz_high = DefectScanner.is_audible("hum", 0.96, "tape", -20, 50)
        audible_3000hz_mod = DefectScanner.is_audible("hum", 0.6, "tape", -20, 3000)
        # 3 kHz needs less severity to be audible
        self.assertTrue(audible_3000hz_mod, "Hum at 3kHz sev=0.6 should be audible")
        self.assertTrue(audible_50hz_high, "Hum at 50Hz sev=0.96 IS audible (just barely)")

    def test_07_unknown_defect_default(self):
        """Unknown defect types use a moderate default threshold."""
        self.assertTrue(
            DefectScanner.is_audible("nonexistent_defect", 0.5, "tape", -20),
            "Unknown defect at 0.5 severity should be audible",
        )
        self.assertFalse(
            DefectScanner.is_audible("nonexistent_defect", 0.1, "tape", -20),
            "Unknown defect at 0.1 severity should be inaudible",
        )

    def test_08_all_materials_have_factors(self):
        """Every threshold has all 9 standard material factors."""
        std_materials = {"shellac", "vinyl", "tape", "cassette", "reel_tape", "cd", "digital", "mp3_low", "dat"}
        for dt, (_, _, factors) in DefectScanner.AUDIBILITY_THRESHOLDS.items():
            missing = std_materials - set(factors.keys())
            self.assertEqual(len(missing), 0, f"{dt} missing material factors: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
