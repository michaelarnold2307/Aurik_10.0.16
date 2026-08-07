"""cli/i18n_cli.py — §v10.700 J5: CLI Internationalisierung.

Nutzt dieselben Übersetzungsdateien wie die GUI (Aurik10/i18n/).
Unterstützt --lang de|en|fr für mehrsprachige CLI-Ausgaben.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_I18N_DIR = Path(__file__).parent.parent / "Aurik10" / "i18n"
_CURRENT_LANG = "de"
_TRANSLATIONS: dict[str, dict[str, str]] = {}


def set_language(lang: str) -> None:
    """Setzt die aktuelle Sprache für CLI-Ausgaben."""
    global _CURRENT_LANG
    _CURRENT_LANG = lang[:2].lower()
    _load_translations()


def _(key: str, **kwargs) -> str:
    """Übersetzt einen Key in die aktuelle Sprache. Fallback: Key selbst."""
    if not _TRANSLATIONS:
        _load_translations()

    lang_dict = _TRANSLATIONS.get(_CURRENT_LANG, {})
    text = lang_dict.get(key, key)

    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text


def _load_translations() -> None:
    """Lädt Übersetzungsdateien für die aktuelle Sprache."""
    global _TRANSLATIONS
    if _TRANSLATIONS.get(_CURRENT_LANG):
        return

    lang_file = _I18N_DIR / f"{_CURRENT_LANG}.json"
    if lang_file.exists():
        try:
            with open(lang_file, encoding="utf-8") as f:
                _TRANSLATIONS[_CURRENT_LANG] = json.load(f)
            logger.debug("CLI i18n: %s geladen (%d Keys)", _CURRENT_LANG, len(_TRANSLATIONS[_CURRENT_LANG]))
        except Exception:
            _TRANSLATIONS[_CURRENT_LANG] = {}
    else:
        _TRANSLATIONS[_CURRENT_LANG] = {}
