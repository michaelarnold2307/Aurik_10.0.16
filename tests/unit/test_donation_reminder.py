from __future__ import annotations

import json
from pathlib import Path

import pytest

import backend.core.donation_reminder as donation


@pytest.mark.unit
def test_open_donation_link_uses_fallback_when_primary_browser_open_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    def fake_open(url: str) -> bool:
        opened.append(url)
        return url == donation.PAYPAL_FALLBACK

    monkeypatch.setattr(donation.webbrowser, "open", fake_open)

    assert donation.open_donation_link() is True
    assert opened == [donation.PAYPAL_URL, donation.PAYPAL_FALLBACK]


@pytest.mark.unit
def test_open_donation_link_returns_false_when_both_browser_paths_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(donation.webbrowser, "open", lambda _url: False)

    assert donation.open_donation_link() is False


@pytest.mark.unit
def test_validate_donation_configuration_reports_verification_limit() -> None:
    status = donation.validate_donation_configuration()  # type: ignore[attr-defined]

    assert status["primary_ok"] is True
    assert status["fallback_ok"] is True
    assert status["email_present"] is True
    assert status["payment_verification"] == "external_paypal_required"
    assert status["guaranteed_capture"] is False


@pytest.mark.unit
def test_extend_grace_period_persists_trust_grace_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grace_file = tmp_path / "donation_grace.json"
    monkeypatch.setattr(donation, "_GRACE_FILE", grace_file)

    days = donation.extend_grace_period(5.0)  # type: ignore[attr-defined]

    assert days >= 30
    state = json.loads(grace_file.read_text(encoding="utf-8"))
    assert state["last_donation_eur"] == 5.0
    assert state["count"] == 1
    assert state["total_donated_eur"] == 5.0


@pytest.mark.unit
def test_get_donation_info_includes_configuration_status() -> None:
    info = donation.get_donation_info()

    assert info["url"] == donation.PAYPAL_URL
    assert info["fallback"] == donation.PAYPAL_FALLBACK
    assert info["email"] == donation.PAYPAL_EMAIL
    assert info["configuration"]["guaranteed_capture"] is False
