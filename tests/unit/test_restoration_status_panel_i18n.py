from __future__ import annotations

from pathlib import Path

import pytest

PANEL_FILE = Path("Aurik10/ui/restoration_status_panel.py")
I18N_FILE = Path("Aurik10/i18n/__init__.py")


@pytest.mark.unit
def test_restoration_status_panel_visible_texts_are_i18n_controlled() -> None:
    src = PANEL_FILE.read_text(encoding="utf-8")

    assert "from Aurik10.i18n import t" in src
    assert 't("status_panel.initializing")' in src
    assert 't("status_panel.quality_header")' in src
    assert 't("status_panel.phase_counter"' in src
    assert 't("status_panel.complete")' in src
    assert '"status_panel.dimension_tooltip"' in src


@pytest.mark.unit
def test_restoration_status_panel_i18n_keys_exist() -> None:
    i18n = I18N_FILE.read_text(encoding="utf-8")
    for key in (
        "status_panel.initializing",
        "status_panel.complete",
        "status_panel.quality_header",
        "status_panel.phase_counter",
        "status_panel.dimension.tonal_center",
        "status_panel.dimension.timbre_authentizitaet",
        "status_panel.dimension.natuerlichkeit",
        "status_panel.dimension.authentizitaet",
        "status_panel.dimension_tooltip",
    ):
        assert key in i18n
