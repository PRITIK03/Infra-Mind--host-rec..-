"""
Tests for optional Tavily research and deterministic requirement bridges
in the system design reasoner. LLM / search calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agent.nodes.system_design_reasoner import (
    _apply_deterministic_requirement_bridges,
    _maybe_collect_research,
    ReasoningError,
    reason_system_design,
)
from app.models.schemas import (
    ResourceProfile,
    TechnicalNeeds,
    TrafficPattern,
    UserRequirements,
    WorkloadType,
)

from langchain_core.messages import AIMessage


def _needs(**kwargs) -> TechnicalNeeds:
    base = dict(
        estimated_concurrency=10,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical",
        reasoning="test",
    )
    base.update(kwargs)
    return TechnicalNeeds(**base)


def test_gpu_bridge_forces_requires_gpu_and_profile():
    requirements = UserRequirements(gpu_required=True)
    bridged = _apply_deterministic_requirement_bridges(
        requirements,
        _needs(resource_profile=ResourceProfile.CPU_BOUND, requires_gpu=False),
    )
    assert bridged.requires_gpu is True
    assert bridged.resource_profile == ResourceProfile.GPU_BOUND


def test_gpu_false_clears_incorrect_gpu_bound_profile():
    requirements = UserRequirements(gpu_required=False)
    bridged = _apply_deterministic_requirement_bridges(
        requirements,
        _needs(resource_profile=ResourceProfile.GPU_BOUND, requires_gpu=True),
    )
    assert bridged.requires_gpu is False
    assert bridged.resource_profile == ResourceProfile.BALANCED


def test_user_resource_profile_is_honored_when_not_unknown():
    requirements = UserRequirements(
        gpu_required=False,
        resource_profile=ResourceProfile.MEMORY_BOUND,
    )
    bridged = _apply_deterministic_requirement_bridges(
        requirements,
        _needs(resource_profile=ResourceProfile.CPU_BOUND),
    )
    assert bridged.resource_profile == ResourceProfile.MEMORY_BOUND
    assert bridged.requires_gpu is False


def test_user_traffic_pattern_is_honored():
    requirements = UserRequirements(traffic_pattern=TrafficPattern.BURSTY)
    bridged = _apply_deterministic_requirement_bridges(
        requirements,
        _needs(traffic_pattern=TrafficPattern.STEADY),
    )
    assert bridged.traffic_pattern == TrafficPattern.BURSTY


@patch("app.agent.nodes.system_design_reasoner.try_get_search_tool", return_value=None)
def test_maybe_collect_research_skips_when_tavily_unavailable(_mock_tool):
    requirements = UserRequirements(
        workload_type=WorkloadType.API_SERVICE,
        estimated_concurrent_users=200,
        traffic_pattern=TrafficPattern.BURSTY,
    )
    assert _maybe_collect_research(requirements) is None


@patch("app.agent.nodes.system_design_reasoner.try_get_search_tool")
def test_maybe_collect_research_skips_when_search_invoke_raises(mock_try_tool):
    search_tool = MagicMock()
    search_tool.invoke.side_effect = RuntimeError("network down")
    mock_try_tool.return_value = search_tool

    with patch("app.agent.nodes.system_design_reasoner.get_chat_model") as mock_get_model:
        decision = MagicMock()
        decision.tool_calls = [{"args": {"query": "AWS sizing"}}]
        bound = MagicMock()
        bound.invoke.return_value = decision
        mock_get_model.return_value.bind_tools.return_value = bound

        requirements = UserRequirements(
            workload_type=WorkloadType.API_SERVICE,
            estimated_concurrent_users=200,
            traffic_pattern=TrafficPattern.BURSTY,
        )
        assert _maybe_collect_research(requirements) is None


@patch("app.agent.nodes.system_design_reasoner.try_get_search_tool")
@patch("app.agent.nodes.system_design_reasoner.get_chat_model")
def test_maybe_collect_research_returns_none_when_model_skips_search(
    mock_get_model, mock_try_tool
):
    search_tool = MagicMock()
    mock_try_tool.return_value = search_tool

    decision = MagicMock()
    decision.tool_calls = []
    bound = MagicMock()
    bound.invoke.return_value = decision
    mock_get_model.return_value.bind_tools.return_value = bound

    requirements = UserRequirements(
        workload_type=WorkloadType.WEB_APP,
        registered_users=1000,
        traffic_pattern=TrafficPattern.STEADY,
    )
    assert _maybe_collect_research(requirements) is None
    search_tool.invoke.assert_not_called()


@patch("app.agent.nodes.system_design_reasoner.try_get_search_tool")
@patch("app.agent.nodes.system_design_reasoner.get_chat_model")
def test_maybe_collect_research_runs_one_search_when_model_requests_it(
    mock_get_model, mock_try_tool
):
    search_tool = MagicMock()
    search_tool.invoke.return_value = {
        "results": [
            {
                "title": "AWS scaling guide",
                "url": "https://example.com/aws",
                "content": "Prefer horizontal scaling for bursty APIs.",
            }
        ]
    }
    mock_try_tool.return_value = search_tool

    decision = MagicMock()
    decision.tool_calls = [{"args": {"query": "AWS EC2 sizing bursty API"}}]
    bound = MagicMock()
    bound.invoke.return_value = decision
    mock_get_model.return_value.bind_tools.return_value = bound

    requirements = UserRequirements(
        workload_type=WorkloadType.API_SERVICE,
        estimated_concurrent_users=500,
        traffic_pattern=TrafficPattern.BURSTY,
    )
    findings = _maybe_collect_research(requirements)
    assert findings is not None
    assert "AWS EC2 sizing bursty API" in findings
    assert "AWS scaling guide" in findings
    search_tool.invoke.assert_called_once_with({"query": "AWS EC2 sizing bursty API"})


@patch("app.agent.nodes.system_design_reasoner._maybe_collect_research", return_value=None)
@patch("app.agent.nodes.system_design_reasoner.invoke_structured")
def test_reason_system_design_applies_gpu_bridge(mock_invoke, _mock_research):
    mock_invoke.return_value = TechnicalNeeds(
        estimated_concurrency=8,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical",
        reasoning="forgot gpu flag",
    )

    state = {
        "requirements": UserRequirements(
            workload_type=WorkloadType.ML_INFERENCE,
            estimated_concurrent_users=8,
            traffic_pattern=TrafficPattern.STEADY,
            gpu_required=True,
        ),
        "latest_user_message": None,
        "next_question": None,
        "technical_needs": None,
        "instance_candidates": None,
        "recommendation": None,
    }
    state = reason_system_design(state)
    assert state["technical_needs"] is not None
    assert state["technical_needs"].requires_gpu is True
    assert state["technical_needs"].resource_profile == ResourceProfile.GPU_BOUND


@patch("app.agent.nodes.system_design_reasoner._maybe_collect_research", return_value=None)
@patch("app.llm.client.get_chat_model")
def test_reason_system_design_accepts_technical_needs_wrapper(mock_get_model, _mock_research):
    """
    Regression for provider returning a wrapper:
      { reasoning: "...", technical_needs: { ... } }
    """
    wrapper_json = (
        '{"reasoning":"wrapped ok",'
        '"technical_needs":{'
        '"estimated_concurrency":500,'
        '"resource_profile":"cpu_bound",'
        '"traffic_pattern":"bursty",'
        '"requires_gpu":false,'
        '"scaling_recommendation":"horizontal with auto scaling",'
        '"reasoning":"inner"}'
        "}"
    )

    model = MagicMock()
    model.invoke.return_value = AIMessage(content=wrapper_json)
    mock_get_model.return_value = model

    state = {
        "requirements": UserRequirements(
            workload_type=WorkloadType.API_SERVICE,
            registered_users=10000,
            traffic_pattern=TrafficPattern.BURSTY,
            gpu_required=False,
        ),
        "latest_user_message": None,
        "next_question": None,
        "technical_needs": None,
        "instance_candidates": None,
        "recommendation": None,
    }

    state = reason_system_design(state)
    assert state["technical_needs"].estimated_concurrency == 500
    assert state["technical_needs"].resource_profile == ResourceProfile.CPU_BOUND
    assert state["technical_needs"].traffic_pattern == TrafficPattern.BURSTY


@patch("app.agent.nodes.system_design_reasoner._maybe_collect_research", return_value=None)
@patch("app.llm.client.get_chat_model")
def test_reason_system_design_invalid_wrapper_fails_with_domain_error(mock_get_model, _mock_research):
    model = MagicMock()
    model.invoke.return_value = AIMessage(
        content='{"reasoning":"x","technical_needs":{"traffic_pattern":"bursty"}}'
    )
    mock_get_model.return_value = model

    state = {
        "requirements": UserRequirements(
            workload_type=WorkloadType.API_SERVICE,
            registered_users=10000,
            traffic_pattern=TrafficPattern.BURSTY,
            gpu_required=False,
        ),
        "latest_user_message": None,
        "next_question": None,
        "technical_needs": None,
        "instance_candidates": None,
        "recommendation": None,
    }

    with pytest.raises(ReasoningError):
        reason_system_design(state)
