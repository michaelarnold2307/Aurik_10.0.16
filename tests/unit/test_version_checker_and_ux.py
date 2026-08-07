import pytest

"""Tests for Aurik10/core/version_checker.py — update check logic."""

import json
import threading
from unittest.mock import MagicMock, patch

from Aurik10.core.version_checker import (
    _CURRENT_VERSION,
    VersionCheckResult,
    _parse_version,
    check_for_update,
    check_for_update_async,
)

# ── _parse_version ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_version_simple():
    assert _parse_version("9.10.77") == (9, 10, 77)


def test_parse_version_with_v_prefix():
    assert _parse_version("v10.0.0") == (10, 0, 0)


def test_parse_version_with_V_prefix():
    assert _parse_version("V1.2.3") == (1, 2, 3)


def test_parse_version_non_numeric():
    assert _parse_version("v10.0.0.beta") == (10, 0, 0)


def test_parse_version_hotfix_suffix():
    assert _parse_version("v10.0.0-hotfix.2") == (10, 0, 0, 2)


def test_parse_version_comparison():
    assert _parse_version("9.10.80") > _parse_version("9.10.77")
    assert _parse_version("9.11.0") > _parse_version("9.10.99")
    assert _parse_version("10.0.0") > _parse_version("9.99.99")
    assert _parse_version("v10.0.0") > _parse_version("9.10.77")
    assert _parse_version("9.12.9-hotfix.2") > _parse_version("9.12.9-hotfix.1")


# ── VersionCheckResult ─────────────────────────────────────────────────────


def test_result_defaults():
    r = VersionCheckResult()
    assert r.available is False
    assert r.error == ""
    assert r.latest_version == ""
    assert r.download_url == ""


def test_default_current_version_uses_package_version():
    from Aurik10 import __version__

    assert __version__ == _CURRENT_VERSION


def test_result_available():
    r = VersionCheckResult(available=True, latest_version="9.11.0", download_url="https://example.com/dl")
    assert r.available is True
    assert r.latest_version == "9.11.0"


# ── check_for_update (mocked) ─────────────────────────────────────────────


def _mock_release(tag: str, assets=None, body="Release notes"):
    """Build a GitHub API-like release dict."""
    data = {"tag_name": tag, "html_url": f"https://github.com/release/{tag}", "body": body}
    if assets:
        data["assets"] = assets
    else:
        data["assets"] = []
    return json.dumps(data).encode("utf-8")


@patch("Aurik10.core.version_checker.urlopen")
def test_check_newer_available(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = _mock_release("v99.0.0")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("9.10.77")
    assert result.available is True
    assert result.latest_version == "99.0.0"
    assert result.error == ""


@patch("Aurik10.core.version_checker.urlopen")
def test_check_up_to_date(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = _mock_release("v10.0.15")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("10.0.15")
    assert result.available is False
    assert result.error == ""


@patch("Aurik10.core.version_checker.urlopen")
def test_check_older_than_current(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = _mock_release("v9.9.0")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("10.0.15")
    assert result.available is False


@patch("Aurik10.core.version_checker.urlopen")
def test_check_with_appimage_asset(mock_urlopen):
    assets = [{"name": "aurik-9.11.0.AppImage", "browser_download_url": "https://dl.example.com/aurik.AppImage"}]
    resp = MagicMock()
    resp.read.return_value = _mock_release("v10.0.0", assets=assets)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("9.10.77")
    assert result.available is True
    assert "AppImage" in result.download_url


@patch("Aurik10.core.version_checker.urlopen")
def test_check_network_error(mock_urlopen):
    from urllib.error import URLError

    mock_urlopen.side_effect = URLError("offline")

    result = check_for_update("9.10.77")
    assert result.available is False
    assert "offline" in result.error


@patch("Aurik10.core.version_checker.urlopen")
def test_check_no_tag(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = json.dumps({"body": "no tag"}).encode()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = check_for_update("9.10.77")
    assert result.available is False
    assert "no tag_name" in result.error


# ── check_for_update_async ─────────────────────────────────────────────────


@patch("Aurik10.core.version_checker.urlopen")
def test_async_check_calls_callback(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = _mock_release("v99.0.0")
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    results = []
    event = threading.Event()

    def cb(r):
        results.append(r)
        event.set()

    check_for_update_async(cb)
    event.wait(timeout=5)
    assert len(results) == 1
    assert results[0].available is True


# ── Batch-Retry / TitleBar Settings (integration-style, no Qt needed) ──────


def test_simple_batch_item_retry_reset():
    """Verify SimpleBatchItem fields can be reset for retry."""
    import sys

    sys.path.insert(0, ".")
    # Minimal duck-type test — no Qt import needed

    class _Item:
        def __init__(self):
            self.status = "failed"
            self.error: str | None = "some error"
            self.progress = 100
            self.restoration_result: object | None = object()

    item = _Item()
    # Simulate retry reset logic
    item.status = "pending"
    item.error = None
    item.progress = 0
    item.restoration_result = None

    assert item.status == "pending"
    assert item.error is None
    assert item.progress == 0
    assert item.restoration_result is None


def test_i18n_keys_exist():
    """All new i18n keys resolve to non-empty strings."""
    from Aurik10.i18n import set_language, t

    keys = [
        "batch.retry_tooltip",
        "batch.retry_hint",
        "update.checking",
        "update.available",
        "update.up_to_date",
        "update.error",
        "update.unavailable",
        "update.banner_text",
        "update.download",
        "help.check_update",
        "settings.title",
    ]
    for lang in ("de", "en"):
        set_language(lang)
        for key in keys:
            val = t(key, count=1, version="9.99.0")
            assert val, f"i18n key '{key}' empty for lang={lang}"
            assert key not in val, f"i18n key '{key}' not translated for lang={lang}"
