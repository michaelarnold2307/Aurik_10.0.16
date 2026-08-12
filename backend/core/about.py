"""About — Aurik Entwickler-Informationen."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

AURIK_ABOUT = {
    "name": "Aurik",
    "version": "10.0.15",
    "codename": "Weltspitze",
    "developer": "Michael Arnold",
    "email": "michael.arnold2307@gmail.com",
    "year": "2026",
    "description": (
        "Aurik ist ein KI-gestütztes Audio-Restaurierungssystem, "
        "das alte und beschädigte Aufnahmen auf Weltspitze-Niveau "
        "wiederherstellt — mit 66 adaptiven Phasen, Closed-Loop-Optimierung "
        "und Self-Supervised Learning."
    ),
    "donation_url": "https://www.paypal.com/donate?business=michael.arnold2307@gmail.com&currency_code=EUR",
    "license": "Apache 2.0 — Kostenlos für private und kommerzielle Nutzung",
}


def get_about_text() -> str:
    """Gibt formatierten Über-Dialog-Text zurück."""
    return f"""
╔══════════════════════════════════════════╗
║             AURIK {AURIK_ABOUT["version"]} — {AURIK_ABOUT["codename"]}            ║
╠══════════════════════════════════════════╣
║                                          ║
║  Entwickler: {AURIK_ABOUT["developer"]}                    ║
║  E-Mail:     {AURIK_ABOUT["email"]}  ║
║  Jahr:       {AURIK_ABOUT["year"]}                          ║
║                                          ║
║  {AURIK_ABOUT["description"][:40]}  ║
║  {AURIK_ABOUT["description"][40:80]}   ║
║  {AURIK_ABOUT["description"][80:120]}  ║
║  {AURIK_ABOUT["description"][120:160]} ║
║                                          ║
║  Lizenz: {AURIK_ABOUT["license"][:42]} ║
║  {AURIK_ABOUT["license"][42:]:<42} ║
║                                          ║
║  ❤️  Danke für Deine Unterstützung!     ║
╚══════════════════════════════════════════╝
"""
