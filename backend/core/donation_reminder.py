"""DonationReminder — §GRATITUDE

Zeigt nach jeder erfolgreichen Restaurierung eine freundliche
Spenden-Erinnerung mit PayPal-Link.
"""

from __future__ import annotations

import logging
import time
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

PAYPAL_URL = "https://www.paypal.com/donate?business=michael.arnold2307@gmail.com&currency_code=EUR"
PAYPAL_EMAIL = "michael.arnold2307@gmail.com"
PAYPAL_FALLBACK = "https://paypal.me/michaelarnold2307"

# Rate-limit: maximal alle 24 Stunden (pro Session oder via Disk-Stamp)
_RATE_LIMIT_SECONDS = 86400  # 24 Stunden
_STAMP_FILE = Path(__file__).parent.parent / "logs" / ".donation_last_shown"

_MESSAGES = [
    "🎵 Dein Song wurde erfolgreich restauriert!",
    "",
    "Aurik ist das Ergebnis tausender Stunden Entwicklungsarbeit —",
    "kostenlos, werbefrei und mit Weltspitze-Qualität.",
    "",
    "Wenn Dir Aurik geholfen hat, freue ich mich über Deine Unterstützung:",
    f"👉 {PAYPAL_URL}",
    "",
    "Jeder Betrag hilft, Aurik weiter zu verbessern. Danke! ❤️",
    "",
    "— Michael (Aurik-Entwickler)",
]


def show_reminder(quality_score: float = 0.0) -> str:
    """Zeigt Spenden-Erinnerung mit personalisiertem Qualitäts-Hinweis."""

    if quality_score > 0.8:
        personal = "🌟 Hervorragende Restaurierung! Aurik hat hier ganze Arbeit geleistet."
    elif quality_score > 0.5:
        personal = "✨ Gute Restaurierung! Aurik konnte den Klang spürbar verbessern."
    else:
        personal = "🎧 Dein Song wurde restauriert. Aurik hat sein Bestes gegeben."

    lines = [personal] + _MESSAGES

    message = "\n".join(lines)
    logger.info(message)
    return message


def open_donation_link() -> bool:
    """Öffnet den Spenden-Link im Browser. Verifiziert via Fallback."""
    try:
        webbrowser.open(PAYPAL_URL)
        logger.debug("Donation link opened: %s", PAYPAL_URL)
        return True
    except Exception:
        try:
            webbrowser.open(PAYPAL_FALLBACK)
            return True
        except Exception:
            return False


def get_donation_info() -> dict:
    """Gibt Spenden-Informationen als Dict zurück."""
    return {
        "url": PAYPAL_URL,
        "fallback": PAYPAL_FALLBACK,
        "email": PAYPAL_EMAIL,
    }


def should_show_reminder() -> bool:
    """Prüft, ob genug Zeit seit der letzten Anzeige vergangen ist (Rate-Limit 24h)."""
    try:
        now = time.time()
        if _STAMP_FILE.exists():
            last = float(_STAMP_FILE.read_text().strip())
            if now - last < _RATE_LIMIT_SECONDS:
                return False
        _STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STAMP_FILE.write_text(str(now))
        return True
    except Exception:
        return True  # Bei Fehler lieber anzeigen als nie
