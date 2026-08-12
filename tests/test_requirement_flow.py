"""
Tests for requirement_collector and requirement_validator using a mocked
LLM, so the merge/validation logic can be verified without a live API call.
"""

from __future__ import annotations

from unittest.mock import patch

from app.agent.nodes.requirement_collector import collect_requirements
from app.agent.nodes.requirement_validator import validate_requirements
from app.agent.state import AgentState
from app.models.schemas import TrafficPattern, UserRequirements, WorkloadType


def _make_state(message: str | None) -> AgentState:
    return {
        "requirements": UserRequirements(),
        "latest_user_message": message,
        "next_question": None,
        "pending_field": None,
        "technical_needs": None,
        "instance_candidates": None,
        "recommendation": None,
    }


@patch("app.agent.nodes.requirement_collector.invoke_structured")
def test_collect_requirements_merges_extracted_fields(mock_invoke):
    mock_invoke.return_value = UserRequirements(
        workload_type=WorkloadType.API_SERVICE,
        registered_users=10000,
        traffic_pattern=TrafficPattern.BURSTY,
        peak_hours="5 PM to 8 PM",
    )

    state = _make_state("10000 users, bursty traffic 5-8pm, FastAPI backend")
    state = collect_requirements(state)

    assert state["requirements"].workload_type == WorkloadType.API_SERVICE
    assert state["requirements"].registered_users == 10000
    assert state["requirements"].traffic_pattern == TrafficPattern.BURSTY
    assert state["latest_user_message"] is None


def test_validate_requirements_asks_for_missing_field():
    state = validate_requirements(_make_state(None))
    assert state["next_question"] is not None


def test_validate_requirements_passes_when_complete():
    state = _make_state(None)
    state["requirements"] = UserRequirements(
        workload_type=WorkloadType.WEB_APP,
        registered_users=5000,
        traffic_pattern=TrafficPattern.STEADY,
    )
    state = validate_requirements(state)
    assert state["next_question"] is None
    assert state["pending_field"] is None


def test_validator_allows_batch_concurrency_without_registered_users():
    state = _make_state(None)
    state["requirements"] = UserRequirements(
        workload_type=WorkloadType.BATCH_PROCESSING,
        estimated_concurrent_users=200,
        traffic_pattern=TrafficPattern.STEADY,
    )
    state = validate_requirements(state)
    assert state["next_question"] is None


def test_validator_batch_expected_scale_question_mentions_jobs():
    state = _make_state(None)
    state["requirements"] = UserRequirements(
        workload_type=WorkloadType.BATCH_PROCESSING,
        traffic_pattern=TrafficPattern.STEADY,
    )
    state = validate_requirements(state)
    assert state["next_question"] is not None
    assert state["pending_field"] == "expected_scale"
    assert "[" not in state["next_question"]
    assert "job" in state["next_question"].lower()


@patch("app.agent.nodes.requirement_collector.invoke_structured")
def test_multi_turn_numeric_answer_uses_pending_field_guidance(mock_invoke):
    """
    Regression for a follow-up loop: when the validator asks for
    expected_scale, the collector must interpret a bare number correctly.
    """
    state = _make_state(None)
    state["requirements"] = UserRequirements(
        workload_type=WorkloadType.API_SERVICE,
        traffic_pattern=TrafficPattern.STEADY,
    )
    # Simulate validator asking for expected_scale.
    state["next_question"] = "Roughly expected scale"
    state["pending_field"] = "expected_scale"
    state["latest_user_message"] = "2000"

    def _capture_schema(_schema, prompt: str):
        # Collector prompt should include the pending field token.
        assert "expected_scale" in prompt
        return UserRequirements(estimated_concurrent_users=2000)

    mock_invoke.side_effect = _capture_schema
    state = collect_requirements(state)
    assert state["latest_user_message"] is None
    assert state["pending_field"] is None

    state = validate_requirements(state)
    assert state["next_question"] is None


@patch("app.agent.nodes.requirement_collector.invoke_structured")
def test_collect_requirements_interprets_concurrent_jobs_for_batch(mock_invoke):
    state = _make_state(
        "CPU-intensive backend processing service with 200 concurrent jobs and steady traffic."
    )

    def _capture(schema, prompt):
        assert "concurrent jobs/workers/worker slots" in prompt
        return UserRequirements(
            workload_type=WorkloadType.BATCH_PROCESSING,
            estimated_concurrent_users=200,
            traffic_pattern=TrafficPattern.STEADY,
        )

    mock_invoke.side_effect = _capture
    state = collect_requirements(state)
    assert state["requirements"].workload_type == WorkloadType.BATCH_PROCESSING
    assert state["requirements"].estimated_concurrent_users == 200
    assert state["requirements"].traffic_pattern == TrafficPattern.STEADY
