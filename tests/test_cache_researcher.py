"""
Unit tests for the cache researcher node.

Uses synthetic CacheCandidate pools only — no live Vantage calls.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agent.nodes.cache_researcher import (
    CacheResearchError,
    _filter_by_family,
    _sample_per_engine,
    _target_family,
    research_cache,
)
from app.models.schemas import (
    CacheCandidate,
    CacheEngine,
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
        needs_cache=True,
        min_instances=1,
        max_instances=1,
        load_balancer_needed=False,
        reasoning="test",
    )
    base.update(kwargs)
    return TechnicalNeeds(**base)


def _cache_candidate(
    instance_type: str,
    family: str,
    engine: CacheEngine,
    vcpu: int,
    memory_gib: float,
    max_clients: int | None = None,
) -> CacheCandidate:
    return CacheCandidate(
        instance_type=instance_type,
        family=family,
        engine=engine,
        vcpu=vcpu,
        memory_gib=memory_gib,
        max_clients=max_clients,
    )


def _make_multi_engine_pool() -> list[CacheCandidate]:
    """
    A realistic synthetic pool: same node types exist under multiple engines,
    and the sizes span a range so the diversity sampler has something to work with.
    """
    engines = [CacheEngine.REDIS, CacheEngine.MEMCACHED, CacheEngine.VALKEY]
    pool: list[CacheCandidate] = []
    sizes = [
        ("cache.t3.small",   "Standard",        2,   1.37),
        ("cache.t3.medium",  "Standard",        2,   3.09),
        ("cache.m5.large",   "Standard",        2,   6.38),
        ("cache.m5.xlarge",  "Standard",        4,  12.93),
        ("cache.m5.2xlarge", "Standard",        8,  26.04),
        ("cache.r5.large",   "Memory optimized", 2,  13.07),
        ("cache.r5.xlarge",  "Memory optimized", 4,  26.32),
    ]
    for engine in engines:
        for instance_type, family, vcpu, mem in sizes:
            pool.append(_cache_candidate(instance_type, family, engine, vcpu, mem))
    return pool


def _make_single_engine_pool(engine: CacheEngine = CacheEngine.REDIS) -> list[CacheCandidate]:
    return [
        _cache_candidate("cache.m5.large",   "Standard", engine, 2,  6.38),
        _cache_candidate("cache.m5.xlarge",  "Standard", engine, 4, 12.93),
        _cache_candidate("cache.m5.2xlarge", "Standard", engine, 8, 26.04),
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
        "cache_candidates": None,
        "recommendation": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _target_family
# ---------------------------------------------------------------------------

def test_target_family_standard_for_balanced():
    assert _target_family(_needs(resource_profile=ResourceProfile.BALANCED)) == "Standard"


def test_target_family_standard_for_cpu_bound():
    assert _target_family(_needs(resource_profile=ResourceProfile.CPU_BOUND)) == "Standard"


def test_target_family_memory_optimized_for_memory_bound():
    assert _target_family(_needs(resource_profile=ResourceProfile.MEMORY_BOUND)) == "Memory optimized"


# ---------------------------------------------------------------------------
# _filter_by_family
# ---------------------------------------------------------------------------

def test_filter_returns_standard_for_balanced():
    pool = _make_multi_engine_pool()
    filtered = _filter_by_family(pool, _needs(resource_profile=ResourceProfile.BALANCED))
    assert filtered
    assert all(c.family == "Standard" for c in filtered)


def test_filter_returns_memory_optimized_for_memory_bound():
    pool = _make_multi_engine_pool()
    filtered = _filter_by_family(pool, _needs(resource_profile=ResourceProfile.MEMORY_BOUND))
    assert filtered
    assert all(c.family == "Memory optimized" for c in filtered)


def test_filter_excludes_zero_vcpu():
    pool = [
        _cache_candidate("cache.m5.large", "Standard", CacheEngine.REDIS, 0, 6.38),
        _cache_candidate("cache.m5.xlarge", "Standard", CacheEngine.REDIS, 4, 12.93),
    ]
    filtered = _filter_by_family(pool, _needs())
    assert all(c.vcpu > 0 for c in filtered)
    assert len(filtered) == 1


def test_filter_fallback_when_no_matching_family():
    # Pool has only Memory optimized; target is Standard → fallback returns all valid.
    pool = [
        _cache_candidate("cache.r5.large",  "Memory optimized", CacheEngine.REDIS, 2, 13.07),
        _cache_candidate("cache.r5.xlarge", "Memory optimized", CacheEngine.REDIS, 4, 26.32),
    ]
    filtered = _filter_by_family(pool, _needs(resource_profile=ResourceProfile.BALANCED))
    assert len(filtered) == 2  # fallback: all valid nodes returned


# ---------------------------------------------------------------------------
# _sample_per_engine — the critical multi-engine test
# ---------------------------------------------------------------------------

def test_sample_per_engine_preserves_all_engines():
    """
    Core invariant: when multiple engines are present in the filtered pool,
    every engine must appear in the final sample — none may be silently dropped.
    """
    pool = _make_multi_engine_pool()
    # Filter to Standard only (all three engines have Standard nodes in the pool).
    standard = [c for c in pool if c.family == "Standard"]
    result = _sample_per_engine(standard, _needs())
    engines_present = {c.engine for c in result}
    assert CacheEngine.REDIS in engines_present
    assert CacheEngine.MEMCACHED in engines_present
    assert CacheEngine.VALKEY in engines_present


def test_sample_per_engine_single_engine_pool_works():
    pool = _make_single_engine_pool(CacheEngine.REDIS)
    result = _sample_per_engine(pool, _needs())
    assert result
    assert all(c.engine == CacheEngine.REDIS for c in result)


def test_sample_per_engine_respects_per_engine_cap():
    # 30 Standard Redis nodes — should be capped to _MAX_PER_ENGINE (16) per engine.
    pool = [
        _cache_candidate(f"cache.m5.{i}xlarge", "Standard", CacheEngine.REDIS, max(i, 1) * 2, max(i, 1) * 6.0)
        for i in range(1, 31)
    ]
    result = _sample_per_engine(pool, _needs(), max_per_engine=16)
    redis_results = [c for c in result if c.engine == CacheEngine.REDIS]
    assert len(redis_results) <= 16


def test_flat_sample_would_lose_engine_but_per_engine_does_not():
    """
    Demonstrates why flat sampling is wrong: a pool with 10 Redis nodes and
    1 Memcached node, sampled flat to limit=5, may drop Memcached entirely.
    Per-engine sampling guarantees Memcached survives.
    """
    from app.agent.nodes._candidate_utils import select_diverse

    pool = [
        _cache_candidate(f"cache.r5.{i}xlarge", "Standard", CacheEngine.REDIS, i * 2, i * 8.0)
        for i in range(1, 11)
    ]
    lone_memcached = _cache_candidate("cache.m5.large", "Standard", CacheEngine.MEMCACHED, 2, 6.38)
    pool.append(lone_memcached)

    # Flat sampling — Memcached is likely to disappear.
    flat = select_diverse(pool, limit=5)
    flat_engines = {c.engine for c in flat}

    # Per-engine sampling — Memcached is guaranteed to survive.
    per_engine = _sample_per_engine(pool, _needs(), max_per_engine=5)
    per_engine_engines = {c.engine for c in per_engine}

    assert CacheEngine.MEMCACHED in per_engine_engines, (
        "Per-engine sampling must preserve Memcached even when Redis dominates the pool"
    )
    # Also document (not assert) that flat sampling may drop it.
    _ = flat_engines  # documented: may or may not contain MEMCACHED


# ---------------------------------------------------------------------------
# research_cache — skip path
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.cache_researcher.fetch_cache_instance_data")
def test_needs_cache_false_skips_fetch(mock_fetch):
    state = _base_state(technical_needs=_needs(needs_cache=False))
    result = research_cache(state)
    mock_fetch.assert_not_called()
    assert result["cache_candidates"] == []


# ---------------------------------------------------------------------------
# research_cache — happy paths
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.cache_researcher.fetch_cache_instance_data")
def test_memory_bound_routes_to_memory_optimized(mock_fetch):
    mock_fetch.return_value = _make_multi_engine_pool()
    state = _base_state(technical_needs=_needs(resource_profile=ResourceProfile.MEMORY_BOUND))
    result = research_cache(state)
    candidates = result["cache_candidates"]
    assert candidates
    assert all(c.family == "Memory optimized" for c in candidates)


@patch("app.agent.nodes.cache_researcher.fetch_cache_instance_data")
def test_balanced_routes_to_standard(mock_fetch):
    mock_fetch.return_value = _make_multi_engine_pool()
    state = _base_state(technical_needs=_needs(resource_profile=ResourceProfile.BALANCED))
    result = research_cache(state)
    candidates = result["cache_candidates"]
    assert candidates
    assert all(c.family == "Standard" for c in candidates)


@patch("app.agent.nodes.cache_researcher.fetch_cache_instance_data")
def test_all_engines_present_in_result(mock_fetch):
    """
    End-to-end check through research_cache: the final state must contain
    candidates for more than one engine when the live pool has multiple engines.
    This is the correctness invariant described in the module docstring.
    """
    mock_fetch.return_value = _make_multi_engine_pool()
    state = _base_state()
    result = research_cache(state)
    engines = {c.engine for c in result["cache_candidates"]}
    assert len(engines) > 1, (
        f"Expected multiple engines in cache_candidates, got: {engines}"
    )
    assert CacheEngine.REDIS in engines
    assert CacheEngine.MEMCACHED in engines


@patch("app.agent.nodes.cache_researcher.fetch_cache_instance_data")
def test_result_has_no_duplicate_engine_instance_pairs(mock_fetch):
    mock_fetch.return_value = _make_multi_engine_pool()
    state = _base_state()
    result = research_cache(state)
    candidates = result["cache_candidates"]
    seen = set()
    for c in candidates:
        key = (c.instance_type, c.engine)
        assert key not in seen, f"Duplicate (instance_type, engine) pair: {key}"
        seen.add(key)


# ---------------------------------------------------------------------------
# research_cache — error path
# ---------------------------------------------------------------------------

@patch("app.agent.nodes.cache_researcher.fetch_cache_instance_data")
def test_no_matching_candidates_raises_cache_research_error(mock_fetch):
    mock_fetch.return_value = _make_multi_engine_pool()
    with patch(
        "app.agent.nodes.cache_researcher._filter_by_family",
        return_value=[],
    ):
        state = _base_state()
        with pytest.raises(CacheResearchError):
            research_cache(state)


@patch("app.agent.nodes.cache_researcher.fetch_cache_instance_data")
def test_all_zero_vcpu_raises_cache_research_error(mock_fetch):
    mock_fetch.return_value = [
        _cache_candidate("cache.m5.large",  "Standard", CacheEngine.REDIS, 0, 6.38),
        _cache_candidate("cache.m5.xlarge", "Standard", CacheEngine.REDIS, 0, 12.93),
    ]
    state = _base_state()
    with pytest.raises(CacheResearchError):
        research_cache(state)
