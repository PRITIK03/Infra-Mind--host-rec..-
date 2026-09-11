"""
Shared pytest fixtures for the AWS Instance Advisor test suite.

_clear_llm_cache
    Clears the in-memory structured-output response cache before and after
    every test.  This prevents successful LLM mock calls in one test from
    returning cached results in a later test that intentionally exercises
    error paths.  The autouse=True means it applies to all tests in the
    suite without requiring explicit import.

_reset_obs_state (not here — stays in test_observability.py because it
    also manipulates DATABASE_URL, which is observability-specific).
"""

from __future__ import annotations

import pytest

from app.llm.structured import _cache_clear


@pytest.fixture(autouse=True)
def _clear_llm_cache():
    """Clear the LLM response cache before and after every test."""
    _cache_clear()
    yield
    _cache_clear()
