from typing import Any

import pytest

"""
Test suite for forensics/analysis_and_modules.py - PolicyManager
Tests policy escalation, reset, thresholds, and logging
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.forensics.analysis_and_modules import PolicyManager


@pytest.mark.unit
def test_policy_manager_initialization():
    """Test PolicyManager initializes with default escalation levels"""
    policy: dict[Any, Any] = {}
    pm = PolicyManager(policy)

    assert pm.policy == {}
    assert pm.escalation_levels == {"warn": 3, "bypass": 5, "hard_bypass": 7}
    assert pm.callback is None


def test_policy_manager_custom_escalation():
    """Test PolicyManager with custom escalation levels"""
    policy: dict[Any, Any] = {}
    custom_levels = {"warn": 2, "bypass": 4, "hard_bypass": 6}
    pm = PolicyManager(policy, escalation_levels=custom_levels)

    assert pm.escalation_levels == custom_levels


def test_policy_update_single_failure():
    """Test policy update with single gate failure"""
    policy: dict[Any, Any] = {}
    pm = PolicyManager(policy)

    feedback = {"gate1": False}
    pm.update(feedback)

    # Gate should be initialized with fail_count=1
    assert "gate1" in pm.policy
    assert pm.policy["gate1"]["fail_count"] == 1
    assert pm.policy["gate1"]["escalated"] is False


def test_policy_escalation_warn_level():
    """Test escalation to warn level after 3 failures"""
    policy: dict[Any, Any] = {}
    pm = PolicyManager(policy)

    # 3 failures → warn level
    for _ in range(3):
        pm.update({"gate1": False})

    assert pm.policy["gate1"]["fail_count"] == 3
    assert pm.policy["gate1"]["escalation_level"] == "warn"
    assert pm.policy["gate1"]["action"] == "warn"
    assert pm.policy["gate1"]["escalated"] is True


def test_policy_escalation_bypass_level():
    """Test escalation to bypass level after 5 failures"""
    policy: dict[Any, Any] = {}
    pm = PolicyManager(policy)

    # 5 failures → bypass level
    for _ in range(5):
        pm.update({"gate1": False})

    assert pm.policy["gate1"]["fail_count"] == 5
    assert pm.policy["gate1"]["escalation_level"] == "bypass"
    assert pm.policy["gate1"]["action"] == "bypass_or_notify"


def test_policy_escalation_hard_bypass_level():
    """Test escalation to hard_bypass level after 7 failures"""
    policy: dict[Any, Any] = {}
    pm = PolicyManager(policy)

    # 7 failures → hard_bypass level
    for _ in range(7):
        pm.update({"gate1": False})

    assert pm.policy["gate1"]["fail_count"] == 7
    assert pm.policy["gate1"]["escalation_level"] == "hard_bypass"
    assert pm.policy["gate1"]["action"] == "hard_bypass"


def test_policy_reset_after_success():
    """Test policy resets after success"""
    policy: dict[Any, Any] = {}
    pm = PolicyManager(policy)

    # 3 failures → warn
    for _ in range(3):
        pm.update({"gate1": False})

    assert pm.policy["gate1"]["fail_count"] == 3
    assert pm.policy["gate1"]["escalated"] is True

    # 1 success → reset
    pm.update({"gate1": True})

    assert pm.policy["gate1"]["fail_count"] == 0
    assert pm.policy["gate1"]["escalated"] is False
    assert pm.policy["gate1"]["escalation_level"] is None


def test_policy_logging():
    """Test policy logs all events"""
    policy: dict[Any, Any] = {}
    pm = PolicyManager(policy)

    # Update triggers logging
    pm.update({"gate1": False})

    assert "_log" in pm.policy
    assert len(pm.policy["_log"]) > 0
    # Check log entry has timestamp
    assert "timestamp" in pm.policy["_log"][0]


def test_policy_callback_invoked():
    """Test callback is invoked on escalation"""
    policy: dict[Any, Any] = {}
    callback_events = []

    def callback(event):
        callback_events.append(event)

    pm = PolicyManager(policy, callback=callback)

    # 3 failures → warn → callback
    for _ in range(3):
        pm.update({"gate1": False})

    # Check callback was called with escalation event
    assert len(callback_events) > 0
    assert callback_events[0]["event"] == "escalation"
    assert callback_events[0]["gate"] == "gate1"
    assert callback_events[0]["level"] == "warn"


def test_policy_threshold_adjustment():
    """Test adaptive threshold adjustment on failures"""
    policy = {
        "gate1": {"threshold": 1.0, "fail_count": 0, "escalated": False, "escalation_level": None, "action": None}
    }
    pm = PolicyManager(policy)

    # 3 failures → warn → threshold reduced
    for _ in range(3):
        pm.update({"gate1": False})

    # Threshold should be reduced to 0.95 * original
    assert pm.policy["gate1"]["threshold"] == 0.95


def test_policy_reset_policy_method():
    """Test reset_policy() clears all gate states"""
    policy: dict[Any, Any] = {}
    pm = PolicyManager(policy)

    # Create some failures
    for _ in range(5):
        pm.update({"gate1": False, "gate2": False})

    assert pm.policy["gate1"]["fail_count"] == 5
    assert pm.policy["gate2"]["fail_count"] == 5

    # Reset
    pm.reset_policy()

    assert pm.policy["gate1"]["fail_count"] == 0
    assert pm.policy["gate2"]["fail_count"] == 0
    # Log should still exist
    assert "_log" in pm.policy


def test_policy_multiple_gates():
    """Test policy handles multiple gates independently"""
    policy: dict[Any, Any] = {}
    pm = PolicyManager(policy)

    # gate1 fails 3 times, gate2 fails 5 times
    for _ in range(3):
        pm.update({"gate1": False, "gate2": False})
    for _ in range(2):
        pm.update({"gate2": False})

    assert pm.policy["gate1"]["fail_count"] == 3
    assert pm.policy["gate1"]["escalation_level"] == "warn"
    assert pm.policy["gate2"]["fail_count"] == 5
    assert pm.policy["gate2"]["escalation_level"] == "bypass"
