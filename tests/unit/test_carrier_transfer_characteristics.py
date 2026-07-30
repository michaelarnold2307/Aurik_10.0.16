from __future__ import annotations

"""Unit-Tests für carrier_transfer_characteristics.py (§4.8, §6.2b, §6.2c)."""


import pytest

from backend.core.carrier_transfer_characteristics import (
    CARRIER_TRANSFER_CHARACTERISTICS,
    MATERIAL_BW_CEILING_HZ,
    MATERIAL_DR_CEILING_DB,
    MATERIAL_SNR_FLOOR_DB,
    compute_cumulative_generation_loss,
    get_bw_ceiling_hz,
    get_dr_ceiling_db,
)


class TestCarrierTransferCharacteristics:
    """Tabellen-Integrität und Wert-Korrektheit."""

    def test_all_16_materials_present(self):
        assert len(CARRIER_TRANSFER_CHARACTERISTICS) == 16

    def test_all_tuples_have_4_elements(self):
        for mat, vals in CARRIER_TRANSFER_CHARACTERISTICS.items():
            assert len(vals) == 4, f"{mat}: erwartet 4 Elemente, hat {len(vals)}"

    @pytest.mark.parametrize(
        "material,expected_bw,expected_dr",
        [
            ("wax_cylinder", 5000, 35),
            ("shellac", 8000, 45),
            ("vinyl", 16000, 70),
            ("reel_tape", 18000, 72),
            ("cd_digital", 22050, 96),
            ("cassette", 16000, 60),  # §BW-CEIL v10.13: 14→16 kHz
            ("tape", 15000, 62),
            ("mp3_low", 16000, 90),
        ],
    )
    def test_spot_check_values(self, material, expected_bw, expected_dr):
        bw, _, _, dr = CARRIER_TRANSFER_CHARACTERISTICS[material]
        assert bw == expected_bw
        assert dr == expected_dr

    def test_bw_ceiling_hz_sync_with_main_table(self):
        """§6.2c — Abgeleitetes Dict muss synchron sein."""
        for mat, vals in CARRIER_TRANSFER_CHARACTERISTICS.items():
            assert MATERIAL_BW_CEILING_HZ[mat] == vals[0], f"BW-Sync-Fehler: {mat}"

    def test_dr_ceiling_db_sync_with_main_table(self):
        """§6.2b — Abgeleitetes Dict muss synchron sein."""
        for mat, vals in CARRIER_TRANSFER_CHARACTERISTICS.items():
            assert MATERIAL_DR_CEILING_DB[mat] == vals[3], f"DR-Sync-Fehler: {mat}"

    def test_snr_floor_db_sync_with_main_table(self):
        for mat, vals in CARRIER_TRANSFER_CHARACTERISTICS.items():
            assert MATERIAL_SNR_FLOOR_DB[mat] == vals[1], f"SNR-Sync-Fehler: {mat}"

    def test_bw_ceiling_all_positive(self):
        for mat, bw in MATERIAL_BW_CEILING_HZ.items():
            assert bw > 0, f"{mat}: BW muss positiv sein"

    def test_dr_ceiling_all_positive(self):
        for mat, dr in MATERIAL_DR_CEILING_DB.items():
            assert dr > 0, f"{mat}: DR muss positiv sein"

    def test_generation_loss_all_non_positive(self):
        """Generationsverlust ist immer ≤ 0 (Energieverlust)."""
        for mat, vals in CARRIER_TRANSFER_CHARACTERISTICS.items():
            assert vals[2] <= 0.0, f"{mat}: generation_loss muss ≤ 0 sein"


class TestComputeCumulativeGenerationLoss:
    def test_single_shellac(self):
        loss = compute_cumulative_generation_loss(["shellac"])
        assert loss == pytest.approx(-5.0)

    def test_multi_gen_chain(self):
        loss = compute_cumulative_generation_loss(["shellac", "reel_tape", "cd_digital"])
        expected = -5.0 + -1.5 + -0.1
        assert loss == pytest.approx(expected)

    def test_empty_chain(self):
        assert compute_cumulative_generation_loss([]) == 0.0

    def test_unknown_material_in_chain(self):
        loss = compute_cumulative_generation_loss(["vinyl", "unknown"])
        expected = -2.0 + -2.0
        assert loss == pytest.approx(expected)

    def test_completely_unknown_material(self):
        loss = compute_cumulative_generation_loss(["nonexistent_material"])
        assert loss == 0.0


class TestConvenienceFunctions:
    def test_get_bw_ceiling_known(self):
        assert get_bw_ceiling_hz("shellac") == 8000

    def test_get_bw_ceiling_unknown(self):
        assert get_bw_ceiling_hz("nonexistent") == 20000

    def test_get_dr_ceiling_known(self):
        assert get_dr_ceiling_db("vinyl") == 70

    def test_get_dr_ceiling_unknown(self):
        assert get_dr_ceiling_db("nonexistent") == 70
