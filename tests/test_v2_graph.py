"""
Tests for the V2 pipeline LangGraph definition.

Verifies the graph structure, node sequencing, and end-to-end flow
through reason_system_design -> research_instances -> research_database
-> research_cache -> holistic_recommend -> END.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.agent.graph import build_graph, build_v1_graph
from app.models.schemas import (
    CacheCandidate,
    CacheEngine,
    CacheRecommendation,
    DatabaseCandidate,
    DatabaseRecommendation,
    InstanceCandidate,
    InstanceRecommendation,
    LoadBalancerRecommendation,
    ResourceProfile,
    SystemDesignRecommendation,
    TechnicalNeeds,
    TrafficPattern,
    UserRequirements,
    WorkloadType,
)


def _sample_needs(needs_db: bool = True, needs_cache: bool = True) -> TechnicalNeeds:
    return TechnicalNeeds(
        estimated_concurrency=20,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="Single instance",
        needs_database=needs_db,
        needs_cache=needs_cache,
        load_balancer_needed=False,
        reasoning="Test system design reasoning",
    )


def _sample_system_recommendation() -> SystemDesignRecommendation:
    return SystemDesignRecommendation(
        compute=InstanceRecommendation(
            recommended_instance="t3.medium",
            why="Good balanced general purpose instance",
            assumptions=["steady load"],
            confidence="high",
        ),
        database=DatabaseRecommendation(
            needed=True,
            recommended_instance="db.t3.medium",
            engine_suggestion="PostgreSQL",
            why="Standard relational storage",
            confidence="high",
        ),
        cache=CacheRecommendation(
            needed=True,
            recommended_instance="cache.t3.medium",
            engine=CacheEngine.REDIS,
            why="Session cache",
            confidence="high",
        ),
        load_balancer=LoadBalancerRecommendation(
            needed=False,
            why="Single instance workload",
        ),
        architecture_summary="Balanced architecture with compute, DB, and Redis cache.",
    )


def test_v2_graph_node_structure():
    """Verify that build_graph contains all V2 nodes and excludes recommend_instance."""
    graph = build_graph()
    nodes = set(graph.nodes.keys())

    assert "collect_requirements" in nodes
    assert "validate_requirements" in nodes
    assert "reason_system_design" in nodes
    assert "research_instances" in nodes
    assert "research_database" in nodes
    assert "research_cache" in nodes
    assert "holistic_recommend" in nodes
    assert "recommend_instance" not in nodes


def test_v1_graph_node_structure():
    """Verify that build_v1_graph contains the legacy V1 nodes."""
    v1_graph = build_v1_graph()
    nodes = set(v1_graph.nodes.keys())

    assert "collect_requirements" in nodes
    assert "validate_requirements" in nodes
    assert "reason_system_design" in nodes
    assert "research_instances" in nodes
    assert "recommend_instance" in nodes
    assert "holistic_recommend" not in nodes


def test_v2_graph_e2e_execution():
    """Verify full V2 graph execution produces system_design_recommendation."""
    mock_compute = [
        InstanceCandidate(instance_type="t3.medium", vcpu=2, memory_gib=4.0),
        InstanceCandidate(instance_type="m5.large", vcpu=2, memory_gib=8.0),
    ]
    mock_db = [
        DatabaseCandidate(instance_type="db.t3.medium", family="General purpose", vcpu=2, memory_gib=4.0),
    ]
    mock_cache = [
        CacheCandidate(
            instance_type="cache.t3.medium",
            family="Standard",
            engine=CacheEngine.REDIS,
            vcpu=2,
            memory_gib=3.14,
        ),
    ]

    needs = _sample_needs(needs_db=True, needs_cache=True)
    expected_rec = _sample_system_recommendation()

    with (
        patch("app.agent.nodes.system_design_reasoner.try_get_search_tool", return_value=None),
        patch("app.agent.nodes.system_design_reasoner.invoke_structured", return_value=needs),
        patch("app.agent.nodes.instance_researcher.fetch_ec2_instance_data", return_value=mock_compute),
        patch("app.agent.nodes.database_researcher.fetch_rds_instance_data", return_value=mock_db),
        patch("app.agent.nodes.cache_researcher.fetch_cache_instance_data", return_value=mock_cache),
        patch("app.agent.nodes.holistic_recommender.invoke_structured", return_value=expected_rec),
    ):
        graph = build_graph()
        initial_state = {
            "requirements": UserRequirements(
                workload_type=WorkloadType.WEB_APP,
                registered_users=1000,
                traffic_pattern=TrafficPattern.STEADY,
            ),
            "latest_user_message": None,
            "next_question": None,
            "pending_field": None,
            "technical_needs": None,
            "instance_candidates": None,
            "database_candidates": None,
            "cache_candidates": None,
            "recommendation": None,
            "system_design_recommendation": None,
        }
        final_state = graph.invoke(initial_state)

    assert final_state.get("system_design_recommendation") is not None
    rec = final_state["system_design_recommendation"]
    assert rec.compute.recommended_instance == "t3.medium"
    assert rec.database.needed is True
    assert rec.database.recommended_instance == "db.t3.medium"
    assert rec.cache.needed is True
    assert rec.cache.recommended_instance == "cache.t3.medium"
    assert rec.cache.engine == CacheEngine.REDIS


def test_v2_graph_e2e_batch_skip_paths():
    """Verify that when database and cache are not needed, their fetch calls are not invoked."""
    mock_compute = [
        InstanceCandidate(instance_type="c5.xlarge", vcpu=4, memory_gib=8.0),
    ]

    batch_needs = TechnicalNeeds(
        estimated_concurrency=1,
        resource_profile=ResourceProfile.CPU_BOUND,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="Single batch instance",
        needs_database=False,
        needs_cache=False,
        load_balancer_needed=False,
        reasoning="Nightly CSV batch processor",
    )
    batch_rec = SystemDesignRecommendation(
        compute=InstanceRecommendation(
            recommended_instance="c5.xlarge",
            why="Compute optimized for batch workloads",
            assumptions=["batch runs once a day"],
            confidence="high",
        ),
        database=DatabaseRecommendation(
            needed=False,
            why="No persistent relational state needed",
            confidence="high",
        ),
        cache=CacheRecommendation(
            needed=False,
            why="No repetitive reads to cache",
            confidence="high",
        ),
        load_balancer=LoadBalancerRecommendation(
            needed=False,
            why="Batch job requires no inbound load balancing",
        ),
        architecture_summary="Single batch compute instance reading and processing CSV files.",
    )

    with (
        patch("app.agent.nodes.system_design_reasoner.try_get_search_tool", return_value=None),
        patch("app.agent.nodes.system_design_reasoner.invoke_structured", return_value=batch_needs),
        patch("app.agent.nodes.instance_researcher.fetch_ec2_instance_data", return_value=mock_compute),
        patch("app.agent.nodes.database_researcher.fetch_rds_instance_data") as mock_db_fetch,
        patch("app.agent.nodes.cache_researcher.fetch_cache_instance_data") as mock_cache_fetch,
        patch("app.agent.nodes.holistic_recommender.invoke_structured", return_value=batch_rec),
    ):
        graph = build_graph()
        initial_state = {
            "requirements": UserRequirements(
                workload_type=WorkloadType.BATCH_PROCESSING,
                registered_users=0,
                traffic_pattern=TrafficPattern.STEADY,
            ),
            "latest_user_message": None,
            "next_question": None,
            "pending_field": None,
            "technical_needs": None,
            "instance_candidates": None,
            "database_candidates": None,
            "cache_candidates": None,
            "recommendation": None,
            "system_design_recommendation": None,
        }
        final_state = graph.invoke(initial_state)

    mock_db_fetch.assert_not_called()
    mock_cache_fetch.assert_not_called()
    assert final_state["database_candidates"] == []
    assert final_state["cache_candidates"] == []
    assert final_state["system_design_recommendation"].database.needed is False
    assert final_state["system_design_recommendation"].cache.needed is False
