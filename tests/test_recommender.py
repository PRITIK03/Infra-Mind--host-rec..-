"""
Tests for recommender candidate-set enforcement.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agent.nodes.recommender import RecommendationError, recommend_instance
from app.models.schemas import (
    InstanceCandidate,
    InstanceRecommendation,
    ResourceProfile,
    TechnicalNeeds,
    TrafficPattern,
    UserRequirements,
    WorkloadType,
)


def _base_state(recommendation: InstanceRecommendation):
    return {
        "requirements": UserRequirements(
            workload_type=WorkloadType.API_SERVICE,
            estimated_concurrent_users=100,
            traffic_pattern=TrafficPattern.BURSTY,
            budget_constraint_usd_monthly=200.0,
        ),
        "latest_user_message": None,
        "next_question": None,
        "technical_needs": TechnicalNeeds(
            estimated_concurrency=100,
            resource_profile=ResourceProfile.BALANCED,
            traffic_pattern=TrafficPattern.BURSTY,
            requires_gpu=False,
            scaling_recommendation="horizontal with auto scaling",
            reasoning="bursty API",
        ),
        "instance_candidates": [
            InstanceCandidate(instance_type="m5.large", vcpu=2, memory_gib=8),
            InstanceCandidate(instance_type="m5.xlarge", vcpu=4, memory_gib=16),
        ],
        "recommendation": None,
        "_expected": recommendation,
    }


def _rec(
    recommended: str,
    *,
    alternative: str | None = None,
    why: str = "ok",
) -> InstanceRecommendation:
    return InstanceRecommendation(
        recommended_instance=recommended,
        why=why,
        assumptions=["test assumption"],
        confidence="medium",
        alternative_instance=alternative,
        trade_off="n/a" if alternative else None,
    )


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_accepts_candidate_from_live_set(mock_invoke):
    recommendation = _rec("m5.large", alternative="m5.xlarge", why="Fits share of bursty load")
    mock_invoke.return_value = recommendation

    state = recommend_instance(_base_state(recommendation))
    assert state["recommendation"].recommended_instance == "m5.large"
    assert mock_invoke.call_count == 1


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_retries_once_then_accepts_valid_retry(mock_invoke):
    invalid = _rec("m5.fantasy", why="made up")
    valid = _rec("m5.large", alternative="m5.xlarge", why="corrected")
    mock_invoke.side_effect = [invalid, valid]

    state = recommend_instance(_base_state(invalid))
    assert state["recommendation"].recommended_instance == "m5.large"
    assert state["recommendation"].alternative_instance == "m5.xlarge"
    assert mock_invoke.call_count == 2

    retry_prompt = mock_invoke.call_args_list[1].args[1]
    assert "m5.fantasy" in retry_prompt
    assert "not one of the available options" in retry_prompt
    assert "m5.large" in retry_prompt
    assert "m5.xlarge" in retry_prompt
    assert "You MUST choose recommended_instance and alternative_instance" in retry_prompt


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_raises_after_two_invalid_attempts(mock_invoke):
    first = _rec("m5.fantasy", why="made up")
    second = _rec("m5.unicorn", why="still made up")
    mock_invoke.side_effect = [first, second]

    with pytest.raises(RecommendationError, match="not in the live candidate set"):
        recommend_instance(_base_state(first))
    assert mock_invoke.call_count == 2


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_retries_when_alternative_is_invented(mock_invoke):
    invalid_alt = _rec("m5.large", alternative="c6i.madeup")
    valid = _rec("m5.large", alternative="m5.xlarge")
    mock_invoke.side_effect = [invalid_alt, valid]

    state = recommend_instance(_base_state(invalid_alt))
    assert state["recommendation"].recommended_instance == "m5.large"
    assert state["recommendation"].alternative_instance == "m5.xlarge"
    assert mock_invoke.call_count == 2
    retry_prompt = mock_invoke.call_args_list[1].args[1]
    assert "c6i.madeup" in retry_prompt


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_raises_when_retry_still_has_invented_alternative(
    mock_invoke,
):
    bad = _rec("m5.large", alternative="c6i.madeup")
    mock_invoke.return_value = bad

    with pytest.raises(RecommendationError, match="c6i.madeup"):
        recommend_instance(_base_state(bad))
    assert mock_invoke.call_count == 2


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_rejects_invented_instance_type(mock_invoke):
    """Two invented recommendations in a row still fail after the single retry."""
    recommendation = _rec("m5.fantasy", why="made up")
    mock_invoke.return_value = recommendation

    with pytest.raises(RecommendationError):
        recommend_instance(_base_state(recommendation))
    assert mock_invoke.call_count == 2


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_prompt_requires_exact_instance_recommendation_schema(
    mock_invoke,
):
    recommendation = _rec(
        "m5.large",
        alternative="m5.xlarge",
        why="Fits the live candidate set",
    )
    mock_invoke.return_value = recommendation

    recommend_instance(_base_state(recommendation))

    prompt = mock_invoke.call_args.args[1]
    assert "InstanceRecommendation schema" in prompt
    assert "exact field names" in prompt
    assert "recommended_instance" in prompt
    assert "why" in prompt
    assert "assumptions" in prompt
    assert "confidence" in prompt
    assert "assumptions MUST be a JSON array/list of strings" in prompt
    assert "Use confidence, NOT confidence_level" in prompt
    assert "Do not add reasoning unless it actually exists in the schema" in prompt
    assert "Do not wrap the recommendation inside any outer key" in prompt
    assert "No wrapper object" in prompt
    assert "single JSON object" in prompt
