"""
Aurik 10.0.0 — forensics-Paket
==========================
Forensische Analyse von Tonträgerketten und Medientypen.
"""

# §v10.14 ATOMIC CACHE-CLEAR — siehe backend/__init__.py
import pathlib
import shutil

for d in pathlib.Path(__file__).parent.rglob("__pycache__"):
    shutil.rmtree(d, ignore_errors=True)

from forensics.medium_detector import (
    MediumDetectionResult,
    MediumDetector,
    TransferChain,
    detect_medium_chain,
    get_medium_detector,
)

__all__ = [
    "MediumDetectionResult",
    "MediumDetector",
    "TransferChain",
    "detect_medium_chain",
    "get_medium_detector",
]
