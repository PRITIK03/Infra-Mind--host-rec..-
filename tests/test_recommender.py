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


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_accepts_candidate_from_live_set(mock_invoke):
    recommendation = InstanceRecommendation(
        recommended_instance="m5.large",
        why="Fits share of bursty load",
        assumptions=["4 replicas"],
        confidence="medium",
        alternative_instance="m5.xlarge",
        trade_off="More headroom, higher cost",
    )
    mock_invoke.return_value = recommendation

    state = _base_state(recommendation)
    state = recommend_instance(state)
    assert state["recommendation"].recommended_instance == "m5.large"


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_rejects_invented_instance_type(mock_invoke):
    recommendation = InstanceRecommendation(
        recommended_instance="m5.fantasy",
        why="made up",
        assumptions=[],
        confidence="low",
        alternative_instance=None,
        trade_off=None,
    )
    mock_invoke.return_value = recommendation

    with pytest.raises(RecommendationError):
        recommend_instance(_base_state(recommendation))


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_clears_invalid_alternative(mock_invoke):
    recommendation = InstanceRecommendation(
        recommended_instance="m5.large",
        why="ok",
        assumptions=[],
        confidence="medium",
        alternative_instance="c6i.madeup",
        trade_off="n/a",
    )
    mock_invoke.return_value = recommendation

    state = recommend_instance(_base_state(recommendation))
    assert state["recommendation"].recommended_instance == "m5.large"
    assert state["recommendation"].alternative_instance is None
    assert state["recommendation"].trade_off is None


@patch("app.agent.nodes.recommender.invoke_structured")
def test_recommend_instance_prompt_requires_exact_instance_recommendation_schema(
    mock_invoke,
):
    recommendation = InstanceRecommendation(
        recommended_instance="m5.large",
        why="Fits the live candidate set",
        assumptions=["steady traffic"],
        confidence="medium",
        alternative_instance="m5.xlarge",
        trade_off="More headroom, more cost",
    )
    mock_invoke.return_value = recommendation

    state = _base_state(recommendation)
    recommend_instance(state)

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
