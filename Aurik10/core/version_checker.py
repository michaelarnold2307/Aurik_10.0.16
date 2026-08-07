"""Background version checker — compares local version against latest GitHub release.

Non-blocking, offline-safe, no telemetry. Respects the RELEASE_MUST offline constraint:
this module never transmits user data; it only fetches a single JSON endpoint.
"""

import json
import logging
import re
import threading
from urllib.error import URLError
from urllib.request import Request, urlopen

from Aurik10 import __version__ as _AURIK_VERSION

logger = logging.getLogger(__name__)

_GITHUB_RELEASES_URL = "https://api.github.com/repos/AURIK-audio/aurik-professional/releases/latest"
_TIMEOUT_S = 8
_CURRENT_VERSION = _AURIK_VERSION

# Singleton
_instance = None
_lock = threading.Lock()


class VersionCheckResult:
    """Immutable result of a version check."""

    __slots__ = ("available", "current_version", "download_url", "error", "latest_version", "release_notes")

    def __init__(
        self,
        available: bool = False,
        latest_version: str = "",
        current_version: str = _CURRENT_VERSION,
        download_url: str = "",
        release_notes: str = "",
        error: str = "",
    ):
        self.available = available
        self.latest_version = latest_version
        self.current_version = current_version
        self.download_url = download_url
        self.release_notes = release_notes
        self.error = error


def _parse_version(v: str) -> tuple[int, ...]:
    """Parst 'v10.0.0' oder '9.12.9-hotfix.2' in ein vergleichbares Tupel."""
    return tuple(int(part) for part in re.findall(r"\d+", v.lstrip("vV").strip()))


def check_for_update(current_version: str | None = None) -> VersionCheckResult:
    """Synchronous update check. Call from a background thread.

    Returns VersionCheckResult with .available=True if a newer version exists.
    Never raises — errors are captured in .error field.
    """
    cur = current_version or _CURRENT_VERSION
    try:
        req = Request(_GITHUB_RELEASES_URL, headers={"Accept": "application/vnd.github.v3+json"})
        with urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tag = data.get("tag_name", "")
        if not tag:
            return VersionCheckResult(current_version=cur, error="no tag_name in response")

        latest = _parse_version(tag)
        current = _parse_version(cur)

        if latest > current:
            # Extract download URL for Linux AppImage or Windows exe
            download_url = ""
            for asset in data.get("assets", []):
                name = asset.get("name", "").lower()
                if name.endswith(".appimage") or name.endswith(".exe"):
                    download_url = asset.get("browser_download_url", "")
                    break
            if not download_url:
                download_url = data.get("html_url", "")

            notes = data.get("body", "")[:500]
            return VersionCheckResult(
                available=True,
                latest_version=tag.lstrip("vV"),
                current_version=cur,
                download_url=download_url,
                release_notes=notes,
            )
        return VersionCheckResult(current_version=cur, latest_version=tag.lstrip("vV"))
    except URLError as exc:
        logger.debug("Aktualisierung Pruefung fehlgeschlagen (offline?): %s", exc)
        return VersionCheckResult(current_version=cur, error=str(exc))
    except Exception as exc:
        logger.debug("Aktualisierung Pruefung error: %s", exc)
        return VersionCheckResult(current_version=cur, error=str(exc))


def check_for_update_async(callback) -> None:
    """Fire-and-forget background update check.

    Args:
        callback: Called with VersionCheckResult on the calling thread's context.
                  For Qt, wrap with QTimer.singleShot(0, ...) in the callback.
    """

    def _worker():
        result = check_for_update()
        if callback:
            callback(result)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
