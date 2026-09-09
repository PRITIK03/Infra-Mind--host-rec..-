"""
Unit tests for the database researcher node.

Uses synthetic DatabaseCandidate pools only — no live Vantage calls.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agent.nodes.database_researcher import (
    DatabaseResearchError,
    _filter_by_family,
    _target_families,
    research_database,
)
from app.models.schemas import (
    DatabaseCandidate,
    ResourceProfile,
    TechnicalNeeds,
    TrafficPattern,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _needs(**kwargs) -> TechnicalNeeds:
    base = dict(
        estimated_concurrency=100,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical / fixed size for steady load",
        needs_database=True,
        needs_cache=False,
        min_instances=1,
        max_instances=1,
        load_balancer_needed=False,
        reasoning="test",
    )
    base.update(kwargs)
    return TechnicalNeeds(**base)


def _db_candidate(
    instance_type: str,
    family: str,
    vcpu: int,
    memory_gib: float,
) -> DatabaseCandidate:
    return DatabaseCandidate(
        instance_type=instance_type,
        family=family,
        vcpu=vcpu,
        memory_gib=memory_gib,
    )


def _make_pool() -> list[DatabaseCandidate]:
    return [
        _db_candidate("db.t3.micro",   "Micro instances",    2,  1.0),
        _db_candidate("db.t3.small",   "Micro instances",    2,  2.0),
        _db_candidate("db.m5.large",   "General purpose",    2,  8.0),
        _db_candidate("db.m5.xlarge",  "General purpose",    4, 16.0),
        _db_candidate("db.m5.2xlarge", "General purpose",    8, 32.0),
        _db_candidate("db.r5.large",   "Memory optimized",   2, 16.0),
        _db_candidate("db.r5.xlarge",  "Memory optimized",   4, 32.0),
        _db_candidate("db.r5.2xlarge", "Memory optimized",   8, 64.0),
    ]


def _base_state(**overrides):
    state = {
        "requirements": None,
        "latest_user_message": None,
        "next_question": None,
        "pending_field": None,
        "technical_needs": _needs(),
        "instance_candidates": None,
        "database_candidates": None,
        "recommendation": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _target_families
# ---------------------------------------------------------------------------

def test_target_families_general_for_balanced_high_concurrency():
    families = _target_families(_needs(
        resource_profile=ResourceProfile.BALANCED,
        estimated_concurrency=100,
    ))
    assert families == ["General purpose"]


def test_target_families_memory_optimized_for_memory_bound():
    families = _target_families(_needs(resource_profile=ResourceProfile.MEMORY_BOUND))
    assert "Memory optimized" in families
    # General purpose should be the fallback
    assert "General purpose" in families


def test_target_families_includes_micro_for_very_low_concurrency():
    families = _target_families(_needs(estimated_concurrency=5))
    assert "Micro instances" in families
    assert "General purpose" in families


def test_target_families_no_micro_above_threshold():
    families = _target_families(_needs(estimated_concurrency=50))
    assert "Micro instances" not in families


# ---------------------------------------------------------------------------
# _filter_by_family
# ---------------------------------------------------------------------------

def test_filter_memory_bound_returns_memory_optimized_candidates():
    pool = _make_pool()
    filtered = _filter_by_family(
        pool,
        _needs(resource_profile=ResourceProfile.MEMORY_BOUND),
    )
    families = {c.family for c in filtered}
    assert "Memory optimized" in families
    # General purpose is accepted as fallback but micro should not appear alone
    assert "Micro instances" not in families or "Memory optimized" in families


def test_filter_general_excludes_memory_optimized_for_balanced_high_load():
    pool = _make_pool()
    filtered = _filter_by_family(
        pool,
        _needs(resource_profile=ResourceProfile.BALANCED, estimated_concurrency=100),
    )
    families = {c.family for c in filtered}
    assert families == {"General purpose"}


def test_filter_low_concurrency_includes_micro_and_general():
    pool = _make_pool()
    filtered = _filter_by_family(
        pool,
        _needs(estimated_concurrency=5),
    )
    families = {c.family for c in filtered}
    assert "Micro instances" in families
    assert "General purpose" in families


def test_filter_excludes_zero_vcpu_candidates():
    pool = [
        _db_candidate("db.m5.large", "General purpose", 0, 8.0),  # invalid
        _db_candidate("db.m5.xlarge", "General purpose", 4, 16.0),
    ]
    filtered = _filter_by_family(pool, _needs())
    assert all(c.vcpu > 0 for c in filtered)
    assert len(filtered) == 1


# ---------------------------------------------------------------------------
# research_database — skip path
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.database_researcher.fetch_rds_instance_data")
def test_needs_database_false_skips_fetch(mock_fetch):
    state = _base_state(technical_needs=_needs(needs_database=False))
    result = research_database(state)
    mock_fetch.assert_not_called()
    assert result["database_candidates"] == []


# ---------------------------------------------------------------------------
# research_database — happy paths
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.database_researcher.fetch_rds_instance_data")
def test_memory_bound_routes_to_memory_optimized(mock_fetch):
    mock_fetch.return_value = _make_pool()
    state = _base_state(
        technical_needs=_needs(resource_profile=ResourceProfile.MEMORY_BOUND),
    )
    result = research_database(state)
    candidates = result["database_candidates"]
    assert candidates
    families = {c.family for c in candidates}
    assert "Memory optimized" in families


@patch("app.agent.nodes.database_researcher.fetch_rds_instance_data")
def test_general_purpose_is_default_family(mock_fetch):
    mock_fetch.return_value = _make_pool()
    state = _base_state(
        technical_needs=_needs(
            resource_profile=ResourceProfile.BALANCED,
            estimated_concurrency=100,
        ),
    )
    result = research_database(state)
    candidates = result["database_candidates"]
    assert candidates
    assert all(c.family == "General purpose" for c in candidates)


@patch("app.agent.nodes.database_researcher.fetch_rds_instance_data")
def test_result_is_capped_and_diverse(mock_fetch):
    # Build a large synthetic pool to exercise the diversity sampler.
    pool = [
        _db_candidate(f"db.m5.{i}xlarge", "General purpose", max(i, 1) * 2, max(i, 1) * 8.0)
        for i in range(1, 60)
    ]
    mock_fetch.return_value = pool
    state = _base_state()
    result = research_database(state)
    candidates = result["database_candidates"]
    assert len(candidates) <= 48
    types = [c.instance_type for c in candidates]
    assert len(types) == len(set(types)), "Duplicate instance types in results"


# ---------------------------------------------------------------------------
# research_database — error path
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.database_researcher.fetch_rds_instance_data")
def test_no_matching_candidates_raises_database_research_error(mock_fetch):
    # Pool with no matching family for a balanced/high-concurrency workload.
    mock_fetch.return_value = [
        _db_candidate("db.t3.micro", "Micro instances", 2, 1.0),
    ]
    # Override _filter_by_family so only "General purpose" is targeted but none exist.
    with patch(
        "app.agent.nodes.database_researcher._filter_by_family",
        return_value=[],
    ):
        state = _base_state()
        with pytest.raises(DatabaseResearchError):
            research_database(state)


@patch("app.agent.nodes.database_researcher.fetch_rds_instance_data")
def test_all_zero_vcpu_raises_database_research_error(mock_fetch):
    mock_fetch.return_value = [
        _db_candidate("db.m5.large", "General purpose", 0, 8.0),
        _db_candidate("db.m5.xlarge", "General purpose", 0, 16.0),
    ]
    state = _base_state()
    with pytest.raises(DatabaseResearchError):
        research_database(state)
