"""
End-to-end recommendation-quality scenarios for the V1 pipeline.

All external services (LLM, Tavily, live EC2) are mocked so tests are
deterministic and do not require paid API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from app.agent.graph import build_graph
from app.agent.nodes.recommender import RecommendationError, _PROMPT_TEMPLATE
from app.agent.nodes.requirement_collector import collect_requirements
from app.agent.nodes.requirement_validator import validate_requirements
from app.models.schemas import (
    InstanceCandidate,
    InstanceRecommendation,
    ResourceProfile,
    TechnicalNeeds,
    TrafficPattern,
    UserRequirements,
    WorkloadType,
)
from app.tools.aws_instance_data import InstanceDataUnavailableError, fetch_ec2_instance_data


def _empty_state(**overrides: Any) -> dict[str, Any]:
    state = {
        "requirements": UserRequirements(),
        "latest_user_message": None,
        "next_question": None,
        "technical_needs": None,
        "instance_candidates": None,
        "recommendation": None,
    }
    state.update(overrides)
    return state


def _candidate(
    instance_type: str,
    vcpu: int,
    memory_gib: float,
    *,
    gpu_count: int = 0,
    gpu_model: str | None = None,
    gpu_memory_gib: float | None = None,
    network: str = "Up to 10 Gigabit",
) -> InstanceCandidate:
    return InstanceCandidate(
        instance_type=instance_type,
        vcpu=vcpu,
        memory_gib=memory_gib,
        gpu_count=gpu_count,
        gpu_model=gpu_model,
        gpu_memory_gib=gpu_memory_gib,
        network_performance=network,
        hourly_price_usd=None,
    )


REALISTIC_EC2_POOL = [
    _candidate("t3.medium", 2, 4),
    _candidate("t3.large", 2, 8),
    _candidate("m5.large", 2, 8),
    _candidate("m5.xlarge", 4, 16),
    _candidate("m5.2xlarge", 8, 32),
    _candidate("m6i.large", 2, 8),
    _candidate("m6i.xlarge", 4, 16),
    _candidate("c6i.large", 2, 4),
    _candidate("c6i.xlarge", 4, 8),
    _candidate("c7i.xlarge", 4, 8),
    _candidate("r6i.large", 2, 16),
    _candidate("r6i.xlarge", 4, 32),
    _candidate("r6i.2xlarge", 8, 64),
    _candidate("g4dn.xlarge", 4, 16, gpu_count=1, gpu_model="NVIDIA T4", gpu_memory_gib=16),
    _candidate("g5.xlarge", 4, 16, gpu_count=1, gpu_model="NVIDIA A10G", gpu_memory_gib=24),
    _candidate("p4d.24xlarge", 96, 1152, gpu_count=8, gpu_model="NVIDIA A100", gpu_memory_gib=320),
]


def _run_graph_to_recommendation(
    *,
    initial_requirements: UserRequirements,
    technical_needs: TechnicalNeeds,
    recommendation: InstanceRecommendation,
    ec2_pool: list[InstanceCandidate] | None = None,
) -> dict[str, Any]:
    """
    Run collect→validate→reason→research→recommend with a complete
    requirements object already present (skips multi-turn collection).
    """
    state = _empty_state(
        requirements=initial_requirements,
        latest_user_message=None,
    )

    # Bypass collector LLM: requirements already complete.
    with (
        patch(
            "app.agent.nodes.system_design_reasoner.try_get_search_tool",
            return_value=None,
        ),
        patch(
            "app.agent.nodes.system_design_reasoner.invoke_structured",
            return_value=technical_needs,
        ),
        patch(
            "app.agent.nodes.instance_researcher.fetch_ec2_instance_data",
            return_value=ec2_pool or REALISTIC_EC2_POOL,
        ),
        patch(
            "app.agent.nodes.recommender.invoke_structured",
            return_value=recommendation,
        ),
    ):
        graph = build_graph()
        # Start after collection by invoking from validate path via full graph
        # with empty message so collector is a no-op.
        return graph.invoke(state)


# ---------------------------------------------------------------------------
# Scenario 1 — Small steady web application
# ---------------------------------------------------------------------------


def test_scenario_1_small_steady_web_app_non_gpu_and_live_candidate():
    requirements = UserRequirements(
        workload_type=WorkloadType.WEB_APP,
        registered_users=2000,
        estimated_concurrent_users=80,
        traffic_pattern=TrafficPattern.STEADY,
        resource_profile=ResourceProfile.BALANCED,
        gpu_required=False,
    )
    needs = TechnicalNeeds(
        estimated_concurrency=80,
        resource_profile=ResourceProfile.CPU_BOUND,  # wrong on purpose; bridge should fix
        traffic_pattern=TrafficPattern.BURSTY,  # wrong on purpose; bridge should fix
        requires_gpu=True,  # wrong on purpose; bridge should fix
        scaling_recommendation="vertical / fixed size for steady load",
        reasoning="steady small web app",
    )
    recommendation = InstanceRecommendation(
        recommended_instance="m5.large",
        why="Balanced general-purpose size for steady modest concurrency",
        assumptions=["concurrency near 80"],
        confidence="medium",
        alternative_instance="t3.large",
        trade_off="Burstable is cheaper but less consistent CPU",
    )

    final = _run_graph_to_recommendation(
        initial_requirements=requirements,
        technical_needs=needs,
        recommendation=recommendation,
    )

    assert final["next_question"] is None
    assert final["technical_needs"] is not None
    assert final["technical_needs"].requires_gpu is False
    assert final["technical_needs"].resource_profile == ResourceProfile.BALANCED
    assert final["technical_needs"].traffic_pattern == TrafficPattern.STEADY
    assert "horizontal" not in final["technical_needs"].scaling_recommendation.lower()
    assert final["instance_candidates"]
    assert all(c.gpu_count == 0 for c in final["instance_candidates"])
    assert final["recommendation"].recommended_instance == "m5.large"
    assert final["recommendation"].recommended_instance in {
        c.instance_type for c in final["instance_candidates"]
    }


# ---------------------------------------------------------------------------
# Scenario 2 — Bursty API workload
# ---------------------------------------------------------------------------


def test_scenario_2_bursty_api_horizontal_and_registered_users_not_concurrency():
    requirements = UserRequirements(
        workload_type=WorkloadType.API_SERVICE,
        registered_users=10000,
        traffic_pattern=TrafficPattern.BURSTY,
        peak_hours="5 PM to 8 PM",
        resource_profile=ResourceProfile.CPU_BOUND,
        gpu_required=False,
    )
    needs = TechnicalNeeds(
        estimated_concurrency=250,  # not 10000
        resource_profile=ResourceProfile.CPU_BOUND,
        traffic_pattern=TrafficPattern.BURSTY,
        requires_gpu=False,
        scaling_recommendation="horizontal with auto scaling for bursty peaks",
        reasoning=(
            "Registered users are 10000 but concurrency is estimated much lower "
            "from bursty peak windows, not equal to registered users."
        ),
    )
    recommendation = InstanceRecommendation(
        recommended_instance="c6i.large",
        why="CPU-oriented share of peak behind an ASG",
        assumptions=["~4 replicas; ~60 concurrent per instance"],
        confidence="medium",
        alternative_instance="c6i.xlarge",
        trade_off="More headroom per node, fewer replicas",
    )

    final = _run_graph_to_recommendation(
        initial_requirements=requirements,
        technical_needs=needs,
        recommendation=recommendation,
    )

    tn = final["technical_needs"]
    assert tn is not None
    assert tn.estimated_concurrency != requirements.registered_users
    assert tn.estimated_concurrency < requirements.registered_users
    assert "registered" in tn.reasoning.lower() or "10000" in tn.reasoning
    assert "horizontal" in tn.scaling_recommendation.lower()
    assert all(c.gpu_count == 0 for c in final["instance_candidates"] or [])
    assert final["recommendation"].recommended_instance.startswith("c")
    assert any("replica" in a.lower() or "per instance" in a.lower() or "concurrent" in a.lower()
               for a in final["recommendation"].assumptions)


# ---------------------------------------------------------------------------
# Scenario 3 — CPU-heavy workload
# ---------------------------------------------------------------------------


def test_scenario_3_cpu_heavy_prefers_compute_candidates():
    requirements = UserRequirements(
        workload_type=WorkloadType.BATCH_PROCESSING,
        estimated_concurrent_users=120,
        traffic_pattern=TrafficPattern.STEADY,
        resource_profile=ResourceProfile.CPU_BOUND,
        gpu_required=False,
    )
    needs = TechnicalNeeds(
        estimated_concurrency=120,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical / fixed workers for steady batch slots",
        reasoning="CPU-heavy batch",
    )
    recommendation = InstanceRecommendation(
        recommended_instance="c6i.xlarge",
        why="Compute-optimized for sustained CPU work",
        assumptions=["concurrency means worker slots"],
        confidence="high",
        alternative_instance="c7i.xlarge",
        trade_off="Newer generation, similar shape",
    )

    final = _run_graph_to_recommendation(
        initial_requirements=requirements,
        technical_needs=needs,
        recommendation=recommendation,
    )

    assert final["technical_needs"].resource_profile == ResourceProfile.CPU_BOUND
    types = {c.instance_type for c in final["instance_candidates"]}
    assert any(t.startswith("c") for t in types)
    assert not any(t.startswith("g") or t.startswith("p") for t in types)
    assert final["recommendation"].recommended_instance in types


# ---------------------------------------------------------------------------
# Scenario 4 — Memory-heavy workload
# ---------------------------------------------------------------------------


def test_scenario_4_memory_heavy_keeps_memory_candidates():
    requirements = UserRequirements(
        workload_type=WorkloadType.WEB_APP,
        estimated_concurrent_users=100,
        traffic_pattern=TrafficPattern.STEADY,
        resource_profile=ResourceProfile.MEMORY_BOUND,
        gpu_required=False,
    )
    needs = TechnicalNeeds(
        estimated_concurrency=100,
        resource_profile=ResourceProfile.MEMORY_BOUND,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical for steady memory-heavy service",
        reasoning="in-memory working set dominates",
    )
    recommendation = InstanceRecommendation(
        recommended_instance="r6i.xlarge",
        why="High memory per vCPU matches memory-bound profile",
        assumptions=["working set fits in 32 GiB with headroom"],
        confidence="medium",
        alternative_instance="r6i.2xlarge",
        trade_off="More memory headroom at higher cost",
    )

    final = _run_graph_to_recommendation(
        initial_requirements=requirements,
        technical_needs=needs,
        recommendation=recommendation,
    )

    candidates = final["instance_candidates"]
    assert candidates
    assert all(c.gpu_count == 0 for c in candidates)
    assert any(c.instance_type.startswith("r") for c in candidates)
    assert all(c.memory_gib / c.vcpu >= 6.0 for c in candidates)
    assert final["recommendation"].recommended_instance.startswith("r")


# ---------------------------------------------------------------------------
# Scenario 5 — Explicit GPU-required workload
# ---------------------------------------------------------------------------


def test_scenario_5_gpu_required_filters_to_live_gpu_only():
    requirements = UserRequirements(
        workload_type=WorkloadType.ML_INFERENCE,
        estimated_concurrent_users=20,
        traffic_pattern=TrafficPattern.STEADY,
        gpu_required=True,
    )
    needs = TechnicalNeeds(
        estimated_concurrency=20,
        resource_profile=ResourceProfile.BALANCED,  # LLM forgot GPU
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical for steady inference",
        reasoning="model forgot GPU",
    )
    recommendation = InstanceRecommendation(
        recommended_instance="g4dn.xlarge",
        why="Single T4 GPU fits modest inference concurrency",
        assumptions=["one model replica per instance"],
        confidence="high",
        alternative_instance="g5.xlarge",
        trade_off="Newer GPU, higher cost",
    )

    final = _run_graph_to_recommendation(
        initial_requirements=requirements,
        technical_needs=needs,
        recommendation=recommendation,
    )

    assert final["technical_needs"].requires_gpu is True
    assert final["technical_needs"].resource_profile == ResourceProfile.GPU_BOUND
    assert final["instance_candidates"]
    assert all(c.gpu_count > 0 for c in final["instance_candidates"])
    # GPU metadata reaches recommender candidate set
    assert any(c.gpu_model for c in final["instance_candidates"])
    assert final["recommendation"].recommended_instance in {
        c.instance_type for c in final["instance_candidates"]
    }
    assert final["recommendation"].recommended_instance != "m5.large"


# ---------------------------------------------------------------------------
# Scenario 6 — Missing critical information
# ---------------------------------------------------------------------------


def test_scenario_6_missing_critical_fields_asks_follow_up_and_stops():
    state = _empty_state(
        requirements=UserRequirements(
            workload_type=WorkloadType.WEB_APP,
            # missing scale + traffic still UNKNOWN
        ),
        latest_user_message=None,
    )
    state = validate_requirements(state)
    assert state["next_question"] is not None
    assert "users" in state["next_question"].lower() or "request" in state["next_question"].lower()

    # UNKNOWN traffic is not treated as complete even with scale present.
    state["requirements"] = UserRequirements(
        workload_type=WorkloadType.WEB_APP,
        registered_users=1000,
        traffic_pattern=TrafficPattern.UNKNOWN,
    )
    state = validate_requirements(state)
    assert state["next_question"] is not None
    assert "steady" in state["next_question"].lower() or "bursty" in state["next_question"].lower()

    with patch(
        "app.agent.nodes.requirement_collector.invoke_structured",
        return_value=UserRequirements(workload_type=WorkloadType.API_SERVICE),
    ):
        graph = build_graph()
        incomplete = graph.invoke(
            _empty_state(latest_user_message="I have an API")
        )
    assert incomplete.get("recommendation") is None
    assert incomplete.get("next_question")
    assert incomplete["requirements"].workload_type == WorkloadType.API_SERVICE


# ---------------------------------------------------------------------------
# Scenario 7 — Requirement merge over multiple turns
# ---------------------------------------------------------------------------


@patch("app.agent.nodes.requirement_collector.invoke_structured")
def test_scenario_7_multi_turn_merge_preserves_and_fills(mock_invoke):
    mock_invoke.side_effect = [
        UserRequirements(
            workload_type=WorkloadType.API_SERVICE,
            registered_users=10000,
        ),
        UserRequirements(
            traffic_pattern=TrafficPattern.BURSTY,
            peak_hours="5 PM to 8 PM",
        ),
        UserRequirements(
            gpu_required=False,
            latency_requirement_ms=100,
        ),
    ]

    state = _empty_state()
    messages = [
        "I have a FastAPI backend with around 10,000 registered users.",
        "Traffic is bursty between 5 PM and 8 PM.",
        "No GPU needed and latency should be low.",
    ]
    for message in messages:
        state["latest_user_message"] = message
        state = collect_requirements(state)
        state = validate_requirements(state)

    req = state["requirements"]
    assert req.workload_type == WorkloadType.API_SERVICE
    assert req.registered_users == 10000
    assert req.traffic_pattern == TrafficPattern.BURSTY
    assert req.peak_hours == "5 PM to 8 PM"
    assert req.gpu_required is False
    assert req.latency_requirement_ms == 100
    assert req.missing_critical_fields() == []
    assert state["next_question"] is None


# ---------------------------------------------------------------------------
# Scenario 8 — Horizontal scaling consistency
# ---------------------------------------------------------------------------


def test_scenario_8_horizontal_scaling_context_reaches_recommender():
    requirements = UserRequirements(
        workload_type=WorkloadType.API_SERVICE,
        estimated_concurrent_users=400,
        traffic_pattern=TrafficPattern.BURSTY,
        gpu_required=False,
        resource_profile=ResourceProfile.BALANCED,
    )
    needs = TechnicalNeeds(
        estimated_concurrency=400,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.BURSTY,
        requires_gpu=False,
        scaling_recommendation="horizontal with auto scaling due to bursty peaks",
        reasoning="bursty API",
    )
    recommendation = InstanceRecommendation(
        recommended_instance="m5.large",
        why="Sized for a share of peak behind ASG, not full 400 on one box",
        assumptions=["4 replicas; ~100 concurrent per instance"],
        confidence="medium",
        alternative_instance="m5.xlarge",
        trade_off="Larger node reduces replica count",
    )

    captured: dict[str, str] = {}

    def _capture_recommend(schema, prompt: str):
        captured["prompt"] = prompt
        return recommendation

    with (
        patch(
            "app.agent.nodes.system_design_reasoner.try_get_search_tool",
            return_value=None,
        ),
        patch(
            "app.agent.nodes.system_design_reasoner.invoke_structured",
            return_value=needs,
        ),
        patch(
            "app.agent.nodes.instance_researcher.fetch_ec2_instance_data",
            return_value=REALISTIC_EC2_POOL,
        ),
        patch(
            "app.agent.nodes.recommender.invoke_structured",
            side_effect=_capture_recommend,
        ),
    ):
        final = build_graph().invoke(_empty_state(requirements=requirements))

    assert "horizontal" in captured["prompt"].lower()
    assert "share of the peak" in captured["prompt"].lower()
    assert final["recommendation"].recommended_instance == "m5.large"
    assert any("per instance" in a.lower() or "replica" in a.lower()
               for a in final["recommendation"].assumptions)
    # Filtering should not force only huge boxes for horizontal load
    assert max(c.vcpu for c in final["instance_candidates"]) < 96


# ---------------------------------------------------------------------------
# Scenario 9 — Strict latency requirement
# ---------------------------------------------------------------------------


def test_scenario_9_latency_reaches_recommender_without_false_guarantees():
    requirements = UserRequirements(
        workload_type=WorkloadType.API_SERVICE,
        estimated_concurrent_users=150,
        traffic_pattern=TrafficPattern.STEADY,
        latency_requirement_ms=50,
        gpu_required=False,
    )
    needs = TechnicalNeeds(
        estimated_concurrency=150,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical with latency-sensitive right-sizing",
        reasoning="50ms latency target noted as constraint",
    )
    recommendation = InstanceRecommendation(
        recommended_instance="m5.xlarge",
        why="Enough CPU headroom for responsive API; latency treated as design constraint",
        assumptions=[
            "EC2 specs alone cannot guarantee 50ms application latency",
            "Application and network path dominate measured latency",
        ],
        confidence="medium",
        alternative_instance="m6i.xlarge",
        trade_off="Newer generation networking",
    )

    captured: dict[str, str] = {}

    def _capture_recommend(schema, prompt: str):
        captured["prompt"] = prompt
        return recommendation

    with (
        patch(
            "app.agent.nodes.system_design_reasoner.try_get_search_tool",
            return_value=None,
        ),
        patch(
            "app.agent.nodes.system_design_reasoner.invoke_structured",
            return_value=needs,
        ),
        patch(
            "app.agent.nodes.instance_researcher.fetch_ec2_instance_data",
            return_value=REALISTIC_EC2_POOL,
        ),
        patch(
            "app.agent.nodes.recommender.invoke_structured",
            side_effect=_capture_recommend,
        ),
    ):
        final = build_graph().invoke(_empty_state(requirements=requirements))

    assert "latency_requirement_ms" in captured["prompt"]
    assert "50" in captured["prompt"]
    assert "do not claim an exact latency guarantee" in _PROMPT_TEMPLATE.lower()
    assert "cannot guarantee" in " ".join(final["recommendation"].assumptions).lower()


# ---------------------------------------------------------------------------
# Scenario 10 — Budget constraint
# ---------------------------------------------------------------------------


def test_scenario_10_budget_reaches_recommender_without_invented_prices():
    requirements = UserRequirements(
        workload_type=WorkloadType.WEB_APP,
        estimated_concurrent_users=60,
        traffic_pattern=TrafficPattern.STEADY,
        budget_constraint_usd_monthly=50.0,
        gpu_required=False,
    )
    needs = TechnicalNeeds(
        estimated_concurrency=60,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical with budget-aware right-sizing",
        reasoning="tight monthly budget",
    )
    recommendation = InstanceRecommendation(
        recommended_instance="m5.large",
        why="Smallest comfortable balanced fit under a tight budget preference",
        assumptions=[
            "Authoritative hourly prices are unavailable in candidate data",
            "Cost guidance is qualitative only",
        ],
        confidence="medium",
        alternative_instance="m6i.large",
        trade_off="Similar size, newer generation, cost still qualitative",
    )

    captured: dict[str, str] = {}

    def _capture_recommend(schema, prompt: str):
        captured["prompt"] = prompt
        return recommendation

    with (
        patch(
            "app.agent.nodes.system_design_reasoner.try_get_search_tool",
            return_value=None,
        ),
        patch(
            "app.agent.nodes.system_design_reasoner.invoke_structured",
            return_value=needs,
        ),
        patch(
            "app.agent.nodes.instance_researcher.fetch_ec2_instance_data",
            return_value=REALISTIC_EC2_POOL,
        ),
        patch(
            "app.agent.nodes.recommender.invoke_structured",
            side_effect=_capture_recommend,
        ),
    ):
        final = build_graph().invoke(_empty_state(requirements=requirements))

    assert "budget_constraint_usd_monthly" in captured["prompt"]
    assert "50" in captured["prompt"]
    prompt_l = _PROMPT_TEMPLATE.lower()
    assert "do not invent dollar" in prompt_l
    assert "qualitative only" in prompt_l
    assert all(c.hourly_price_usd is None for c in final["instance_candidates"])
    assert "unavailable" in " ".join(final["recommendation"].assumptions).lower()


# ---------------------------------------------------------------------------
# Scenario 11 — Tavily unavailable / failures
# ---------------------------------------------------------------------------


@patch("app.agent.nodes.system_design_reasoner.invoke_structured")
@patch("app.agent.nodes.system_design_reasoner.try_get_search_tool", return_value=None)
def test_scenario_11_tavily_missing_still_produces_technical_needs(
    _mock_tool, mock_invoke
):
    needs = TechnicalNeeds(
        estimated_concurrency=40,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical",
        reasoning="no search needed",
    )
    mock_invoke.return_value = needs
    state = _empty_state(
        requirements=UserRequirements(
            workload_type=WorkloadType.WEB_APP,
            estimated_concurrent_users=40,
            traffic_pattern=TrafficPattern.STEADY,
        )
    )
    from app.agent.nodes.system_design_reasoner import reason_system_design

    state = reason_system_design(state)
    assert state["technical_needs"] is not None
    assert state["technical_needs"].estimated_concurrency == 40


@patch("app.tools.web_search.get_tavily_settings")
def test_scenario_11_search_tool_init_failure_is_soft(mock_settings):
    from app.config import ConfigError
    from app.tools.web_search import WebSearchUnavailableError, get_search_tool, try_get_search_tool

    mock_settings.side_effect = ConfigError(
        "Missing required environment variable: TAVILY_API_KEY"
    )
    with pytest.raises(WebSearchUnavailableError):
        get_search_tool()
    assert try_get_search_tool() is None


# ---------------------------------------------------------------------------
# Scenario 12 — Live EC2 data failure
# ---------------------------------------------------------------------------


def test_scenario_12_live_ec2_network_failure_raises_clear_error():
    with patch(
        "app.tools.aws_instance_data._get_json",
        side_effect=InstanceDataUnavailableError("Failed to reach vantage: network"),
    ):
        with pytest.raises(InstanceDataUnavailableError, match="Failed to reach"):
            fetch_ec2_instance_data()


def test_scenario_12_live_ec2_empty_response_raises_clear_error():
    with patch("app.tools.aws_instance_data._get_json", return_value=[]):
        with pytest.raises(InstanceDataUnavailableError, match="no results"):
            fetch_ec2_instance_data()


def test_scenario_12_live_ec2_invalid_structure_raises_clear_error():
    with patch("app.tools.aws_instance_data._get_json", return_value={"not": "a list"}):
        with pytest.raises(InstanceDataUnavailableError, match="Unexpected"):
            fetch_ec2_instance_data()


def test_scenario_12_research_node_does_not_invent_candidates_on_fetch_failure():
    from app.agent.nodes.instance_researcher import research_instances

    state = _empty_state(
        technical_needs=TechnicalNeeds(
            estimated_concurrency=10,
            resource_profile=ResourceProfile.BALANCED,
            traffic_pattern=TrafficPattern.STEADY,
            requires_gpu=False,
            scaling_recommendation="vertical",
            reasoning="x",
        )
    )
    with patch(
        "app.agent.nodes.instance_researcher.fetch_ec2_instance_data",
        side_effect=InstanceDataUnavailableError("network down"),
    ):
        with pytest.raises(InstanceDataUnavailableError):
            research_instances(state)
    assert state["instance_candidates"] is None


# ---------------------------------------------------------------------------
# Scenario 13 — LLM invents unsupported instance
# ---------------------------------------------------------------------------


def test_scenario_13_invented_instance_is_rejected():
    requirements = UserRequirements(
        workload_type=WorkloadType.WEB_APP,
        estimated_concurrent_users=50,
        traffic_pattern=TrafficPattern.STEADY,
        gpu_required=False,
    )
    needs = TechnicalNeeds(
        estimated_concurrency=50,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical",
        reasoning="ok",
    )
    recommendation = InstanceRecommendation(
        recommended_instance="m5.unicorn",
        why="invented",
        assumptions=[],
        confidence="low",
        alternative_instance=None,
        trade_off=None,
    )

    with pytest.raises(RecommendationError, match="not in the live candidate set"):
        _run_graph_to_recommendation(
            initial_requirements=requirements,
            technical_needs=needs,
            recommendation=recommendation,
        )
