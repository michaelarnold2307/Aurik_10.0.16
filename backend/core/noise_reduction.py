"""
Noise Reduction Facade — Routes to DeepFilterNet v3.II Plugin.

Provides ``get_noise_reducer()`` as bridge-compatible accessor (§9.7.4).
Delegates to ``plugins.deepfilternet_v3_ii_plugin.get_deepfilternet_plugin()``.

Author: Aurik Development Team
Version: 10.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

logger = logging.getLogger(__name__)

try:
    from plugins.deepfilternet_v3_ii_plugin import get_deepfilternet_plugin as _get_dfn
    from plugins.deepfilternet_v3_ii_plugin import get_loaded_deepfilternet_plugin as _get_loaded_dfn

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    logger.debug("noise_reduction: DeepFilterNet plugin not verfuegbar — DSP Ersatzpfad only")


def get_noise_reducer() -> DeepFilterNetV3Plugin | None:
    """Gibt the DeepFilterNet v3.II singleton, or None if unavailable zurück."""
    if not _AVAILABLE:
        return None
    _loaded = _get_loaded_dfn()
    if _loaded is not None:
        return _loaded
    return _get_dfn()
