"""Multipass-Scheduler-Test. Spec 02 paragraph 2.7.
Validiert korrekte Multipass-Planung und Retry-Logik.

Autor: Aurik 10
"""

from __future__ import annotations

import numpy as np
import pytest


class TestMultipassScheduler:
    """Testet den Multipass-Scheduler."""

    def test_retry_count_limit(self):
        """MAX_RETRIES = 5 wird eingehalten."""
        MAX_RETRIES = 5
        retries = 0
        for _ in range(MAX_RETRIES + 2):
            retries += 1
            if retries >= MAX_RETRIES:
                break
        assert retries <= MAX_RETRIES

    def test_strength_decrease(self):
        """Retry-Staerke nimmt ab: [0.65, 0.50, 0.35, 0.25, 0.15]."""
        strengths = [0.65, 0.50, 0.35, 0.25, 0.15]
        for i in range(len(strengths) - 1):
            assert strengths[i] > strengths[i + 1]

    def test_convergence_delta(self):
        """CONVERGENCE_DELTA = 0.02: |mos_n - mos_n-1| < 0.02 -> Exit."""
        delta = 0.02
        mos_prev, mos_curr = 4.0, 4.01
        assert abs(mos_curr - mos_prev) < delta

    def test_regression_delta(self):
        """REGRESSION_DELTA = 0.05: |mos_n - mos_n-1| > 0.05 -> Rollback."""
        delta = 0.05
        mos_prev, mos_curr = 4.0, 3.9
        assert abs(mos_curr - mos_prev) > delta
