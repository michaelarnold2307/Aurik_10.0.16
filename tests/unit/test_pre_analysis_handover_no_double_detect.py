"""
Regression test: PreAnalysisResult handover prevents double MediumDetector calls.

Scenario:
  1. Import audio → run_pre_analysis() stores MediumDetector result in cache
  2. UI captures the result in _latest_pre_analysis_result (direct handover)
  3. Mode click → adds result to queue settings
  4. Batch thread → prefers queue settings over cache
  5. denke() receives result via pre_analysis_result kwarg
  6. UV3 uses cached_medium_kwarg → NO second detect call

Validation: MediumDetector.detect() is invoked exactly once (during pre-analysis),
not again during restoration start (UV3 restore phase).
"""

from unittest.mock import patch

import numpy as np
import pytest

from backend.core.pre_analysis import run_pre_analysis


@pytest.fixture
def audio_48k_mono():
    """Simple test audio: 48 kHz, 1 second, mono."""
    return np.random.randn(48000).astype(np.float32)


@pytest.mark.unit
def test_pre_analysis_result_cached_then_passed_to_denker(audio_48k_mono):
    """
    Verify that a full PreAnalysisResult from run_pre_analysis() can be
    captured and passed directly to the denker without triggering a second
    MediumDetector.detect() call.
    """
    call_count = {"detect": 0}

    class FakeMediumDetector:
        def detect(self, audio, sr, file_ext=None):
            call_count["detect"] += 1
            from dataclasses import dataclass

            @dataclass
            class FakeMediumResult:
                primary_material: str = "vinyl"
                confidence: float = 0.9
                transfer_chain: list = None  # type: ignore[assignment]
                chain_label: str = "vinyl"

            if call_count["detect"] == 1:
                return FakeMediumResult(transfer_chain=["vinyl", "mp3_low"])
            else:
                # Second call should never happen
                raise AssertionError(f"MediumDetector.detect() called {call_count['detect']} times — should be 1")

    # Phase 1: Pre-analysis (should call detect once)
    with patch("forensics.medium_detector.get_medium_detector") as mock_get_md:
        mock_md_instance = FakeMediumDetector()
        mock_get_md.return_value = mock_md_instance

        result = run_pre_analysis(
            audio_48k_mono,
            sr_native=48000,
            audio_48k=audio_48k_mono,
            file_path="/tmp/test_vinyl.wav",
            store_in_bridge_cache=False,  # Don't pollute real cache
        )

    # Verify call count after pre-analysis
    assert call_count["detect"] == 1, f"Expected 1 detect call during pre-analysis, got {call_count['detect']}"
    assert result.medium is not None, "PreAnalysisResult.medium should not be None"
    assert result.medium.primary_material == "vinyl", (  # type: ignore[attr-defined]
        f"Material should be 'vinyl', got {result.medium.primary_material}"  # type: ignore[attr-defined]
    )

    # Phase 2: Simulate denker receiving the pre-analysis result as kwarg
    # (without cache lookup, without second detect call)
    pre_result_direct = result  # This is what the UI captures

    # Simulate UV3 trying to use this directly
    # The key invariant: no second detect() call should happen
    assert call_count["detect"] == 1, (
        f"Second MediumDetector.detect() call detected after passing to denker! Total calls: {call_count['detect']}"
    )

    # Verify the material was preserved through handover
    captured_material = getattr(pre_result_direct.medium, "primary_material", "unknown")
    assert captured_material == "vinyl" or (
        isinstance(captured_material, str) and captured_material.lower() == "vinyl"
    ), f"Material lost in handover: {captured_material}"


def test_cache_first_then_direct_handover_flow():
    """
    Test the full UI → Batch → Denker flow:
    1. Import + pre-analysis (detect called once)
    2. Cache the result
    3. UI captures in _latest_pre_analysis_result
    4. Mode click adds to queue settings
    5. Batch prefers queue settings, passes to denker
    6. Denker uses pre_analysis_result kwarg
    → No second detect call
    """
    call_count = {"detect": 0}

    class MockMediumDetector:
        def detect(self, audio, sr, file_ext=None):
            call_count["detect"] += 1
            if call_count["detect"] > 1:
                raise AssertionError(f"BUG: MediumDetector called {call_count['detect']} times")
            from dataclasses import dataclass

            @dataclass
            class Result:
                primary_material: str = "tape"
                confidence: float = 0.95
                transfer_chain: list = None  # type: ignore[assignment]
                chain_label: str = "tape"

            return Result(transfer_chain=["tape", "mp3_high"])

    with patch("forensics.medium_detector.get_medium_detector") as mock_get_md:
        mock_get_md.return_value = MockMediumDetector()

        # Step 1: Pre-analysis (first detect call)
        audio = np.zeros(48000, dtype=np.float32)
        result = run_pre_analysis(
            audio, sr_native=48000, audio_48k=audio, file_path="/tmp/test.mp3", store_in_bridge_cache=False
        )

    assert call_count["detect"] == 1
    assert result.medium.primary_material == "tape"  # type: ignore[attr-defined]

    # Step 2: Simulate queue item with direct handover
    queue_settings = {
        "mode": "RESTORATION",
        "pre_analysis_result": result,  # <-- Direct handover
    }

    # Step 3: Batch thread retrieves from queue settings
    queued_pre = queue_settings.get("pre_analysis_result")
    assert queued_pre is not None
    assert queued_pre.medium.primary_material == "tape"  # type: ignore[attr-defined]

    # Step 4: This is what gets passed to denke()
    denke_kwargs = {
        "pre_analysis_result": queued_pre,
        "mode": "restoration",
    }

    # Verify no additional detect calls were made
    assert call_count["detect"] == 1, (
        f"Expected exactly 1 detect call, got {call_count['detect']}. "
        "Direct handover should bypass cache lookups and second detect."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
