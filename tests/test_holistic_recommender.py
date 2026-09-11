"""
Tests for the holistic recommender node.

All LLM calls are mocked — no live API calls.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agent.nodes.holistic_recommender import (
    HolisticRecommendationError,
    recommend_system_design,
)
from app.models.schemas import (
    CacheCandidate,
    CacheEngine,
    CacheRecommendation,
    DatabaseCandidate,
    DatabaseRecommendation,
    EstimatedCost,
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


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _needs(**kw) -> TechnicalNeeds:
    base = dict(
        estimated_concurrency=100,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical / fixed size",
        needs_database=True,
        needs_cache=True,
        min_instances=1,
        max_instances=1,
        load_balancer_needed=False,
        reasoning="test",
    )
    base.update(kw)
    return TechnicalNeeds(**base)


def _compute_candidate(instance_type: str, vcpu: int = 2, mem: float = 8.0, price: float | None = 0.096) -> InstanceCandidate:
    return InstanceCandidate(
        instance_type=instance_type, vcpu=vcpu, memory_gib=mem,
        hourly_price_usd=price,
    )


def _db_candidate(instance_type: str, vcpu: int = 2, mem: float = 8.0, price: float | None = 0.23) -> DatabaseCandidate:
    return DatabaseCandidate(
        instance_type=instance_type, family="General purpose", vcpu=vcpu, memory_gib=mem,
        hourly_price_usd=price,
    )


def _cache_candidate(
    instance_type: str,
    engine: CacheEngine,
    vcpu: int = 2,
    mem: float = 6.0,
    price: float | None = 0.173,
) -> CacheCandidate:
    return CacheCandidate(
        instance_type=instance_type, family="Standard", engine=engine, vcpu=vcpu, memory_gib=mem,
        hourly_price_usd=price,
    )


def _full_result(
    compute_type: str = "m5.large",
    db_type: str = "db.t3.medium",
    cache_type: str = "cache.r5.large",
    cache_engine: CacheEngine = CacheEngine.REDIS,
    lb_needed: bool = False,
    alt_cache_type: str | None = None,
    alt_cache_engine: CacheEngine | None = None,
) -> SystemDesignRecommendation:
    return SystemDesignRecommendation(
        compute=InstanceRecommendation(
            recommended_instance=compute_type,
            why="fits steady load",
            assumptions=[],
            confidence="medium",
        ),
        database=DatabaseRecommendation(
            needed=True,
            recommended_instance=db_type,
            engine_suggestion="PostgreSQL",
            why="standard web app db",
            assumptions=[],
            confidence="medium",
        ),
        cache=CacheRecommendation(
            needed=True,
            recommended_instance=cache_type,
            engine=cache_engine,
            why="read-heavy",
            assumptions=[],
            confidence="medium",
            alternative_instance=alt_cache_type,
            alternative_engine=alt_cache_engine,
        ),
        load_balancer=LoadBalancerRecommendation(
            needed=lb_needed,
            load_balancer_type="Application Load Balancer" if lb_needed else None,
            why="no lb needed" if not lb_needed else "distributes traffic",
        ),
        architecture_summary="Single instance, Postgres RDS, Redis cache.",
    )


def _base_state(**overrides) -> dict:
    state = {
        "requirements": UserRequirements(
            workload_type=WorkloadType.WEB_APP,
            registered_users=5000,
            traffic_pattern=TrafficPattern.STEADY,
        ),
        "latest_user_message": None,
        "next_question": None,
        "pending_field": None,
        "technical_needs": _needs(),
        "instance_candidates": [
            _compute_candidate("m5.large"),
            _compute_candidate("m5.xlarge", vcpu=4, mem=16.0),
        ],
        "database_candidates": [
            _db_candidate("db.t3.medium"),
            _db_candidate("db.t3.large", vcpu=4, mem=16.0),
        ],
        "cache_candidates": [
            _cache_candidate("cache.r5.large", CacheEngine.REDIS),
            _cache_candidate("cache.r5.large", CacheEngine.MEMCACHED),
            _cache_candidate("cache.r5.xlarge", CacheEngine.REDIS, vcpu=4, mem=26.0),
        ],
        "recommendation": None,
        "system_design_recommendation": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# needs_database=False and needs_cache=False short-circuit paths
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_needs_database_false_sets_needed_false_without_llm_db(mock_invoke):
    """When needs_database=False, database sub-rec must be needed=False regardless of LLM."""
    # LLM returns needed=True for database — but the node must override it.
    result = _full_result()
    result = result.model_copy(
        update={"database": DatabaseRecommendation(
            needed=True, recommended_instance="db.t3.medium",
            why="llm suggested this", confidence="medium",
        )}
    )
    mock_invoke.return_value = result

    state = _base_state(technical_needs=_needs(needs_database=False))
    out = recommend_system_design(state)
    sdr = out["system_design_recommendation"]

    assert sdr.database.needed is False
    assert sdr.database.recommended_instance is None
    assert sdr.database.why == "Not required for this workload."


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_needs_cache_false_sets_needed_false_without_llm_cache(mock_invoke):
    """When needs_cache=False, cache sub-rec must be needed=False regardless of LLM."""
    result = _full_result()
    result = result.model_copy(
        update={"cache": CacheRecommendation(
            needed=True, recommended_instance="cache.r5.large",
            engine=CacheEngine.REDIS, why="llm suggested this", confidence="medium",
        )}
    )
    mock_invoke.return_value = result

    state = _base_state(technical_needs=_needs(needs_cache=False))
    out = recommend_system_design(state)
    sdr = out["system_design_recommendation"]

    assert sdr.cache.needed is False
    assert sdr.cache.recommended_instance is None
    assert sdr.cache.engine is None
    assert sdr.cache.why == "Not required for this workload."


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_skipped_tier_uses_relevant_reasoning_excerpt(mock_invoke):
    mock_invoke.return_value = _full_result()
    reasoning = (
        "The workload is steady and does not need a persistent database because "
        "the application has no durable relational data."
    )
    state = _base_state(
        technical_needs=_needs(needs_database=False, reasoning=reasoning),
        database_candidates=[],
    )

    sdr = recommend_system_design(state)["system_design_recommendation"]

    assert sdr.database.why == reasoning


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_skipped_tier_preserves_long_reasoning(mock_invoke):
    mock_invoke.return_value = _full_result()
    reasoning = (
        "The batch workload does not need a cache because it has one worker slot, "
        "steady execution, no repeated hot-read pattern, and no shared transient "
        "state between concurrent workers. This full rationale must remain intact "
        "for the report and Markdown export."
    )
    state = _base_state(
        technical_needs=_needs(needs_cache=False, reasoning=reasoning),
        cache_candidates=[],
    )

    sdr = recommend_system_design(state)["system_design_recommendation"]

    assert sdr.cache.why == reasoning


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_both_skipped_tiers_still_produce_valid_compute(mock_invoke):
    mock_invoke.return_value = _full_result()
    state = _base_state(
        technical_needs=_needs(needs_database=False, needs_cache=False),
        database_candidates=[],
        cache_candidates=[],
    )
    out = recommend_system_design(state)
    sdr = out["system_design_recommendation"]
    assert sdr.compute.recommended_instance == "m5.large"
    assert sdr.database.needed is False
    assert sdr.cache.needed is False
    mock_invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Load-balancer invariant
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_load_balancer_needed_always_matches_technical_needs(mock_invoke):
    """LLM says lb not needed, but technical_needs.load_balancer_needed=True — must be corrected."""
    result = _full_result(lb_needed=False)  # LLM got it wrong
    mock_invoke.return_value = result

    state = _base_state(technical_needs=_needs(load_balancer_needed=True))
    out = recommend_system_design(state)
    sdr = out["system_design_recommendation"]

    assert sdr.load_balancer.needed is True


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_load_balancer_not_needed_when_single_instance(mock_invoke):
    result = _full_result(lb_needed=False)
    mock_invoke.return_value = result

    state = _base_state(technical_needs=_needs(load_balancer_needed=False))
    out = recommend_system_design(state)
    assert out["system_design_recommendation"].load_balancer.needed is False


# ---------------------------------------------------------------------------
# Compute hallucination detection and retry
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_invented_compute_is_caught_and_retried(mock_invoke):
    bad = _full_result(compute_type="m5.fantasy")
    good = _full_result(compute_type="m5.large")
    mock_invoke.side_effect = [bad, good]

    state = _base_state()
    out = recommend_system_design(state)
    assert out["system_design_recommendation"].compute.recommended_instance == "m5.large"
    assert mock_invoke.call_count == 2


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_invented_compute_after_retry_raises(mock_invoke):
    bad = _full_result(compute_type="m5.fantasy")
    mock_invoke.return_value = bad

    with pytest.raises(HolisticRecommendationError, match="not in the live EC2 candidate set"):
        recommend_system_design(_base_state())
    assert mock_invoke.call_count == 2


# ---------------------------------------------------------------------------
# Database hallucination detection and retry
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_invented_database_instance_is_caught_and_retried(mock_invoke):
    bad = _full_result(db_type="db.r99.fantasy")
    good = _full_result(db_type="db.t3.medium")
    mock_invoke.side_effect = [bad, good]

    state = _base_state()
    out = recommend_system_design(state)
    assert out["system_design_recommendation"].database.recommended_instance == "db.t3.medium"
    assert mock_invoke.call_count == 2


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_invented_database_after_retry_raises(mock_invoke):
    bad = _full_result(db_type="db.r99.fantasy")
    mock_invoke.return_value = bad

    with pytest.raises(HolisticRecommendationError, match="not in the live RDS candidate set"):
        recommend_system_design(_base_state())
    assert mock_invoke.call_count == 2


# ---------------------------------------------------------------------------
# Cache (instance_type, engine) PAIR validation — the critical test
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_cache_wrong_engine_for_valid_instance_type_is_caught(mock_invoke):
    """
    cache.r5.large exists in the pool tagged MEMCACHED and REDIS.
    But suppose the LLM assigns it engine=VALKEY — that pair doesn't exist.
    The node must catch this and retry.
    """
    # Pool: cache.r5.large exists only for REDIS and MEMCACHED, not VALKEY.
    cache_pool = [
        _cache_candidate("cache.r5.large", CacheEngine.REDIS),
        _cache_candidate("cache.r5.large", CacheEngine.MEMCACHED),
    ]
    bad = _full_result(cache_type="cache.r5.large", cache_engine=CacheEngine.VALKEY)
    good = _full_result(cache_type="cache.r5.large", cache_engine=CacheEngine.REDIS)
    mock_invoke.side_effect = [bad, good]

    state = _base_state(cache_candidates=cache_pool)
    out = recommend_system_design(state)
    sdr = out["system_design_recommendation"]
    assert sdr.cache.recommended_instance == "cache.r5.large"
    assert sdr.cache.engine == CacheEngine.REDIS
    assert mock_invoke.call_count == 2


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_cache_invalid_pair_after_retry_raises(mock_invoke):
    cache_pool = [
        _cache_candidate("cache.r5.large", CacheEngine.REDIS),
    ]
    bad = _full_result(cache_type="cache.r5.large", cache_engine=CacheEngine.VALKEY)
    mock_invoke.return_value = bad

    with pytest.raises(HolisticRecommendationError, match="not a valid"):
        recommend_system_design(_base_state(cache_candidates=cache_pool))
    assert mock_invoke.call_count == 2


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_cache_correct_pair_accepted_first_try(mock_invoke):
    cache_pool = [
        _cache_candidate("cache.r5.large", CacheEngine.REDIS),
        _cache_candidate("cache.r5.large", CacheEngine.MEMCACHED),
    ]
    good = _full_result(cache_type="cache.r5.large", cache_engine=CacheEngine.REDIS)
    mock_invoke.return_value = good

    out = recommend_system_design(_base_state(cache_candidates=cache_pool))
    assert out["system_design_recommendation"].cache.engine == CacheEngine.REDIS
    assert mock_invoke.call_count == 1


# ---------------------------------------------------------------------------
# Happy path — full recommendation
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_full_recommendation_populates_all_tiers(mock_invoke):
    mock_invoke.return_value = _full_result()
    out = recommend_system_design(_base_state())
    sdr = out["system_design_recommendation"]

    assert sdr.compute.recommended_instance == "m5.large"
    assert sdr.database.needed is True
    assert sdr.database.recommended_instance == "db.t3.medium"
    assert sdr.cache.needed is True
    assert sdr.cache.engine == CacheEngine.REDIS
    assert sdr.load_balancer.needed is False
    assert sdr.architecture_summary


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_missing_technical_needs_raises(mock_invoke):
    state = _base_state()
    state["technical_needs"] = None
    with pytest.raises(HolisticRecommendationError):
        recommend_system_design(state)
    mock_invoke.assert_not_called()


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_missing_compute_candidates_raises(mock_invoke):
    state = _base_state(instance_candidates=[])
    with pytest.raises(HolisticRecommendationError):
        recommend_system_design(state)
    mock_invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Estimated cost computation
# ---------------------------------------------------------------------------


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_estimated_cost_populated_when_all_pricing_available(mock_invoke):
    """All tiers have pricing → cost fields populated."""
    mock_invoke.return_value = _full_result()
    out = recommend_system_design(_base_state())
    cost = out["system_design_recommendation"].estimated_cost

    assert cost is not None
    # m5.large price = 0.096, min=1 max=1 → 0.096 × 730 = 70.08
    assert cost.compute_monthly_low == 70.08
    assert cost.total_monthly_low == round(70.08 + 167.9 + 126.29, 2)
    assert cost.total_monthly_high == round(70.08 + 167.9 + 126.29, 2)


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_estimated_cost_compute_range_with_multiple_instances(mock_invoke):
    """min=2, max=6 → compute cost spans a range."""
    mock_invoke.return_value = _full_result()
    state = _base_state()
    state["technical_needs"] = _needs(
        min_instances=2, max_instances=6, load_balancer_needed=True
    )
    out = recommend_system_design(state)
    cost = out["system_design_recommendation"].estimated_cost

    assert cost is not None
    # 0.096 × 2 × 730 = 140.16
    assert cost.compute_monthly_low == 140.16
    # 0.096 × 6 × 730 = 420.48
    assert cost.compute_monthly_high == 420.48
    assert cost.compute_monthly_low < cost.compute_monthly_high


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_estimated_cost_none_when_compute_pricing_unavailable(mock_invoke):
    """Compute candidate has no price → all costs None."""
    mock_invoke.return_value = _full_result(cache_type="cache.t3.medium", cache_engine=CacheEngine.REDIS)
    state = _base_state()
    state["instance_candidates"] = [_compute_candidate("m5.large", price=None)]
    state["database_candidates"] = [_db_candidate("db.t3.medium", price=None)]
    state["cache_candidates"] = [_cache_candidate("cache.t3.medium", CacheEngine.REDIS, price=None)]
    out = recommend_system_design(state)
    cost = out["system_design_recommendation"].estimated_cost

    assert cost is not None
    assert cost.compute_monthly_low is None
    assert cost.compute_monthly_high is None
    assert cost.database_monthly is None
    assert cost.cache_monthly is None
    assert cost.total_monthly_low is None
    assert cost.total_monthly_high is None


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_estimated_cost_skipped_tiers_not_priced(mock_invoke):
    """When DB/cache not needed, their costs are None."""
    result = _full_result()
    mock_invoke.return_value = result
    state = _base_state(
        technical_needs=_needs(needs_database=False, needs_cache=False),
        database_candidates=[],
        cache_candidates=[],
    )
    out = recommend_system_design(state)
    cost = out["system_design_recommendation"].estimated_cost

    assert cost is not None
    assert cost.compute_monthly_low is not None  # compute is priced
    assert cost.database_monthly is None
    assert cost.cache_monthly is None
    # Total includes only compute
    assert cost.total_monthly_low == cost.compute_monthly_low
    assert cost.total_monthly_high == cost.compute_monthly_high


@patch("app.agent.nodes.holistic_recommender.invoke_structured")
def test_estimated_cost_partial_pricing_partial_total(mock_invoke):
    """DB has price, cache candidates have None → cache cost None, total None."""
    result = _full_result(cache_type="cache.t3.medium", cache_engine=CacheEngine.REDIS)
    mock_invoke.return_value = result
    state = _base_state()
    state["cache_candidates"] = [_cache_candidate("cache.t3.medium", CacheEngine.REDIS, price=None)]
    out = recommend_system_design(state)
    cost = out["system_design_recommendation"].estimated_cost

    assert cost is not None
    assert cost.compute_monthly_low is not None
    assert cost.database_monthly is not None
    assert cost.cache_monthly is None
    assert cost.total_monthly_low is None  # can't sum with None
    assert cost.total_monthly_high is None
