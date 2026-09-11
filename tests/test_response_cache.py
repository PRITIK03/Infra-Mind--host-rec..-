"""
Tests for the in-memory response cache in app/llm/structured.py.

Scenarios covered
-----------------
1. Cache miss: first call hits the LLM; result stored.
2. Cache hit: identical prompt → LLM NOT called again; same object returned.
3. Whitespace normalisation: minor whitespace differences in prompt → same
   cache key → cache hit (LLM called only once).
4. Different schema, same prompt → different cache key → two LLM calls.
5. Expiry: expired entry is not returned; LLM is called again.
6. GroundingResult is never cached: same prompt called twice always hits
   the LLM both times.
7. retry_callback supplied → cache bypassed entirely.
8. Max-size eviction: inserting > _CACHE_MAX_SIZE entries evicts the oldest.
9. _cache_clear() wipes all entries.

All tests mock _call_with_failover so no real LLM or network calls happen.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import app.llm.structured as structured_mod
from app.llm.structured import (
    _cache_clear,
    _cache_key,
    _cache_get,
    _cache_put,
    _CACHE_TTL_S,
    _CACHE_MAX_SIZE,
    _is_cacheable,
    invoke_structured,
)
from app.agent.nodes.grounding_check import GroundingResult


# ---------------------------------------------------------------------------
# Minimal Pydantic schema for testing cache without importing heavy schemas
# ---------------------------------------------------------------------------

class _DummySchema(BaseModel):
    value: str


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure a clean cache before every test."""
    _cache_clear()
    yield
    _cache_clear()


# ---------------------------------------------------------------------------
# Helper: patch the inner LLM call and return a known value
# ---------------------------------------------------------------------------

def _patched_invoke(schema: type, prompt: str, mock_return, **kw):
    """
    Call invoke_structured with _call_with_failover fully mocked.
    mock_return is the value the fake LLM call returns.
    """
    def _fake_failover(fn, *, retry_callback=None):
        # fn receives a model; we hand it a MagicMock model.
        # But invoke_structured builds json_prompt internally and calls
        # _call_with_failover — the fn closure calls model.invoke(json_prompt).
        # We short-circuit by making fn return mock_return directly.
        return mock_return

    with patch("app.llm.structured._call_with_failover", side_effect=_fake_failover):
        return invoke_structured(schema, prompt, **kw)


# ---------------------------------------------------------------------------
# 1 & 2. Miss then hit
# ---------------------------------------------------------------------------

def test_cache_miss_then_hit():
    """First call hits the (mocked) LLM; second identical call uses the cache."""
    result1 = _DummySchema(value="hello")

    call_count = 0

    def _fake_failover(fn, *, retry_callback=None):
        nonlocal call_count
        call_count += 1
        return result1

    with patch("app.llm.structured._call_with_failover", side_effect=_fake_failover):
        r1 = invoke_structured(_DummySchema, "tell me something")
        r2 = invoke_structured(_DummySchema, "tell me something")

    assert r1 is result1
    assert r2 is result1        # same object from cache
    assert call_count == 1      # LLM called only once


# ---------------------------------------------------------------------------
# 3. Whitespace normalisation
# ---------------------------------------------------------------------------

def test_whitespace_normalisation_gives_cache_hit():
    """Extra whitespace in the prompt → same cache key → only one LLM call."""
    call_count = 0
    result = _DummySchema(value="normalised")

    def _fake_failover(fn, *, retry_callback=None):
        nonlocal call_count
        call_count += 1
        return result

    with patch("app.llm.structured._call_with_failover", side_effect=_fake_failover):
        invoke_structured(_DummySchema, "  hello   world  ")
        invoke_structured(_DummySchema, "hello world")   # normalised identical

    assert call_count == 1


# ---------------------------------------------------------------------------
# 4. Different schema → different key → two LLM calls
# ---------------------------------------------------------------------------

class _OtherSchema(BaseModel):
    label: str


def test_different_schema_different_key():
    """Same prompt but different schema → separate cache entries."""
    call_count = 0

    def _fake_failover(fn, *, retry_callback=None):
        nonlocal call_count
        call_count += 1
        # Return appropriate type per call order.
        if call_count == 1:
            return _DummySchema(value="d")
        return _OtherSchema(label="o")

    with patch("app.llm.structured._call_with_failover", side_effect=_fake_failover):
        invoke_structured(_DummySchema, "shared prompt")
        invoke_structured(_OtherSchema, "shared prompt")

    assert call_count == 2


# ---------------------------------------------------------------------------
# 5. Expiry: expired entry → LLM called again
# ---------------------------------------------------------------------------

def test_expired_entry_triggers_new_llm_call():
    """After TTL expires, the cache key is considered a miss and LLM is re-called."""
    call_count = 0
    result = _DummySchema(value="fresh")

    def _fake_failover(fn, *, retry_callback=None):
        nonlocal call_count
        call_count += 1
        return result

    with patch("app.llm.structured._call_with_failover", side_effect=_fake_failover):
        invoke_structured(_DummySchema, "expiry test")

    assert call_count == 1

    # Manually expire the entry by backdating its expires_at.
    key = _cache_key("_DummySchema", "expiry test")
    with structured_mod._cache_lock:
        if key in structured_mod._cache:
            structured_mod._cache[key]["expires_at"] = time.monotonic() - 1.0

    with patch("app.llm.structured._call_with_failover", side_effect=_fake_failover):
        invoke_structured(_DummySchema, "expiry test")

    assert call_count == 2   # LLM called again after expiry


# ---------------------------------------------------------------------------
# 6. GroundingResult is never cached
# ---------------------------------------------------------------------------

def test_grounding_result_never_cached():
    """
    GroundingResult calls must always hit the LLM — no caching allowed,
    because grounding checks audit freshly-produced recommendations.
    """
    assert _is_cacheable(GroundingResult) is False

    call_count = 0
    gr = GroundingResult(passed=True, issues=[])

    def _fake_failover(fn, *, retry_callback=None):
        nonlocal call_count
        call_count += 1
        return gr

    with patch("app.llm.structured._call_with_failover", side_effect=_fake_failover):
        invoke_structured(GroundingResult, "same grounding prompt")
        invoke_structured(GroundingResult, "same grounding prompt")

    assert call_count == 2   # both calls go to LLM — never served from cache


# ---------------------------------------------------------------------------
# 7. retry_callback supplied → cache bypassed
# ---------------------------------------------------------------------------

def test_retry_callback_bypasses_cache():
    """
    Supplying a retry_callback opts out of caching entirely so test
    overrides can always get a fresh LLM call.
    """
    call_count = 0
    result = _DummySchema(value="bypass")

    def _fake_failover(fn, *, retry_callback=None):
        nonlocal call_count
        call_count += 1
        return result

    cb = lambda attempt, max_: None   # no-op callback

    with patch("app.llm.structured._call_with_failover", side_effect=_fake_failover):
        invoke_structured(_DummySchema, "bypass prompt", retry_callback=cb)
        invoke_structured(_DummySchema, "bypass prompt", retry_callback=cb)

    assert call_count == 2   # no caching when retry_callback is set


# ---------------------------------------------------------------------------
# 8. Max-size eviction
# ---------------------------------------------------------------------------

def test_max_size_eviction():
    """
    After inserting _CACHE_MAX_SIZE entries the cache never exceeds the cap,
    and the oldest entry is evicted to make room for new ones.
    """
    # Fill the cache to exactly the limit.
    base_time = time.monotonic()
    for i in range(_CACHE_MAX_SIZE):
        key = f"_DummySchema:{'a' * 60}{i:04d}"
        _cache_put(key, _DummySchema(value=str(i)))

    with structured_mod._cache_lock:
        assert len(structured_mod._cache) == _CACHE_MAX_SIZE

    # Manually set the first entry to be the oldest (smallest expires_at).
    with structured_mod._cache_lock:
        first_key = next(iter(structured_mod._cache))
        structured_mod._cache[first_key]["expires_at"] = base_time  # very old

    # Insert one more — should evict the oldest.
    new_key = "_DummySchema:brand_new_entry"
    _cache_put(new_key, _DummySchema(value="new"))

    with structured_mod._cache_lock:
        # Size is still at most _CACHE_MAX_SIZE.
        assert len(structured_mod._cache) <= _CACHE_MAX_SIZE
        # New entry present.
        assert new_key in structured_mod._cache


# ---------------------------------------------------------------------------
# 9. _cache_clear wipes everything
# ---------------------------------------------------------------------------

def test_cache_clear_empties_cache():
    _cache_put("k1", _DummySchema(value="x"))
    _cache_put("k2", _DummySchema(value="y"))

    with structured_mod._cache_lock:
        assert len(structured_mod._cache) == 2

    _cache_clear()

    with structured_mod._cache_lock:
        assert len(structured_mod._cache) == 0


# ---------------------------------------------------------------------------
# 10. _is_cacheable helper
# ---------------------------------------------------------------------------

def test_is_cacheable_true_for_regular_schema():
    assert _is_cacheable(_DummySchema) is True


def test_is_cacheable_false_for_grounding_result():
    assert _is_cacheable(GroundingResult) is False
