from __future__ import annotations

import logging

from backend.core.phase_fazit import log_restoration_summary


def test_restoration_summary_surfaces_degraded_status_and_no_effect_count(caplog):
    caplog.set_level(logging.INFO, logger="backend.core.phase_fazit")

    log_restoration_summary(  # type: ignore[call-arg]
        total_time_s=1500.0,
        rt_factor=50.0,
        quality_pct=69.0,
        chain=["reel_tape", "lacquer_disc"],
        genre="Deutscher Schlager",
        era_decade=1960,
        phases_count=39,
        mushra_score=38.9,
        hpi_score=0.60,
        degradation_status="degraded",
        fail_reason="RESTORATION_OQS_GATE_DEGRADED",
        oqs_threshold=72.0,
        residual_audible_defects=34,
        no_effect_count=12,
    )

    text = caplog.text
    assert "AURIK RESTAURATION MIT EINSCHRÄNKUNG" in text
    assert "RESTORATION_OQS_GATE_DEGRADED" in text
    assert "OQS-Floor: 72" in text
    assert "34" in text
    assert "12 ohne Wirkung" in text
