"""
Tests for:
  1. LLM_FALLBACK_MODELS config parsing and OpenRouter 'models' array wiring
  2. retry_info updates and clears via the _retry_context ContextVar
  3. Job.retry_info / JobStore.update_retry_info
  4. GET /api/recommend/{id} exposes retry_info when set, omits it when None

All LLM and live-data calls are mocked — no real external calls.
"""

from __future__ import annotations

import os
import time
from contextvars import copy_context
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

os.environ.setdefault("API_KEY", "primary-test-key")
os.environ.setdefault("BASE_URL", "https://example.com/v1")
os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("VANTAGE_API_KEY", "vantage-test-key")
os.environ.setdefault("CORS_ALLOWED_ORIGIN", "http://localhost:3000")

from app.llm.client import (
    _retry_context,
    _call_with_failover,
    get_chat_model,
    RateLimitExhaustedError,
)
from app.api import jobs as jobs_module
from app.models.schemas import UserRequirements


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rate_limit_error():
    from openai import RateLimitError
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    response.json.return_value = {"error": {"message": "rate limited"}}
    body = {"error": {"message": "rate limited", "type": "requests", "code": "rate_limit_exceeded"}}
    return RateLimitError(message="rate limited", response=response, body=body)


def _make_job(job_id: str) -> jobs_module.Job:
    return jobs_module.Job(
        job_id=job_id,
        status="collecting",
        current_stage="Initializing",
        state={
            "requirements": UserRequirements(),
            "latest_user_message": "test",
            "next_question": None,
            "pending_field": None,
            "technical_needs": None,
            "instance_candidates": None,
            "database_candidates": None,
            "cache_candidates": None,
            "recommendation": None,
            "system_design_recommendation": None,
            "terraform_files": None,
        },
    )


# ---------------------------------------------------------------------------
# 1. fallback_models config parsing
# ---------------------------------------------------------------------------


def test_fallback_models_empty_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("LLM_FALLBACK_MODELS", raising=False)
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("BASE_URL", "https://x.com/v1")
    monkeypatch.setenv("MODEL_NAME", "m")

    from app.config import get_llm_settings
    settings = get_llm_settings()
    assert settings.fallback_models == ()


def test_fallback_models_empty_when_env_var_blank(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "")
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("BASE_URL", "https://x.com/v1")
    monkeypatch.setenv("MODEL_NAME", "m")

    from app.config import get_llm_settings
    settings = get_llm_settings()
    assert settings.fallback_models == ()


def test_fallback_models_parsed_from_comma_separated(monkeypatch):
    monkeypatch.setenv(
        "LLM_FALLBACK_MODELS",
        "meta-llama/llama-3.1-70b-instruct:free, mistralai/mistral-7b-instruct:free",
    )
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("BASE_URL", "https://x.com/v1")
    monkeypatch.setenv("MODEL_NAME", "m")

    from app.config import get_llm_settings
    settings = get_llm_settings()
    assert settings.fallback_models == (
        "meta-llama/llama-3.1-70b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
    )


# ---------------------------------------------------------------------------
# 2. get_chat_model: 'models' array in extra_body
# ---------------------------------------------------------------------------


def test_get_chat_model_no_models_array_when_fallback_empty(monkeypatch):
    """Backward-compatible: no 'models' key in extra_body when fallback list is empty."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("MODEL_NAME", "primary-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    monkeypatch.setenv("LLM_REASONING_MAX_TOKENS", "2048")
    monkeypatch.delenv("LLM_FALLBACK_MODELS", raising=False)

    get_chat_model.cache_clear()
    model = get_chat_model()

    payload = model._get_request_payload([HumanMessage(content="ping")], stop=None)
    extra = payload.get("extra_body", {})

    assert "models" not in extra, (
        f"'models' key must not appear in extra_body when fallback_models is empty; got {extra}"
    )
    # reasoning key must still be present
    assert "reasoning" in extra


def test_get_chat_model_adds_models_array_when_fallback_set(monkeypatch):
    """When fallback_models is configured, extra_body['models'] = [primary, *fallbacks]."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("MODEL_NAME", "primary-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    monkeypatch.setenv("LLM_REASONING_MAX_TOKENS", "2048")
    monkeypatch.setenv(
        "LLM_FALLBACK_MODELS",
        "meta-llama/llama-3.1-70b-instruct:free,google/gemma-2-9b-it:free",
    )

    get_chat_model.cache_clear()
    model = get_chat_model()

    payload = model._get_request_payload([HumanMessage(content="ping")], stop=None)
    extra = payload.get("extra_body", {})

    assert "models" in extra, f"Expected 'models' in extra_body, got: {extra}"
    assert extra["models"] == [
        "primary-model",
        "meta-llama/llama-3.1-70b-instruct:free",
        "google/gemma-2-9b-it:free",
    ]
    # reasoning must still be present alongside models
    assert "reasoning" in extra


def test_get_chat_model_models_array_starts_with_primary(monkeypatch):
    """Primary model is always first in the array."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("MODEL_NAME", "my/primary")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "other/model-a,other/model-b")

    get_chat_model.cache_clear()
    model = get_chat_model()

    payload = model._get_request_payload([HumanMessage(content="ping")], stop=None)
    models = payload["extra_body"]["models"]
    assert models[0] == "my/primary"
    assert models[1:] == ["other/model-a", "other/model-b"]


# ---------------------------------------------------------------------------
# 3. _retry_context ContextVar: retry_info updates and clears
# ---------------------------------------------------------------------------


def test_retry_context_callback_called_on_rate_limit(monkeypatch):
    """_call_with_failover reads _retry_context and calls it on each retry."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    callback_calls: list[tuple[int, int]] = []

    def _cb(attempt: int, max_attempts: int) -> None:
        callback_calls.append((attempt, max_attempts))

    attempt_count = 0

    def _fail_twice_then_succeed(model):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise _make_rate_limit_error()
        return "ok"

    token = _retry_context.set(_cb)
    try:
        with patch("app.llm.client.time.sleep"):
            result = _call_with_failover(_fail_twice_then_succeed)
    finally:
        _retry_context.reset(token)

    assert result == "ok"
    # Two retries fired (attempts 1 and 2), so callback called twice
    assert len(callback_calls) == 2
    assert callback_calls[0] == (2, 4)  # attempt 2 of 4
    assert callback_calls[1] == (3, 4)  # attempt 3 of 4


def test_retry_context_not_called_when_first_attempt_succeeds(monkeypatch):
    """Callback is never called when the first attempt succeeds."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    callback_calls: list = []

    token = _retry_context.set(lambda a, m: callback_calls.append((a, m)))
    try:
        with patch("app.llm.client.time.sleep"):
            result = _call_with_failover(lambda _: "immediate-success")
    finally:
        _retry_context.reset(token)

    assert result == "immediate-success"
    assert callback_calls == []


def test_retry_context_default_none_no_error(monkeypatch):
    """With no context var set, _call_with_failover succeeds silently (no AttributeError)."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    # _retry_context defaults to None — must not cause any error
    with patch("app.llm.client.time.sleep"):
        result = _call_with_failover(lambda _: "no-context-ok")

    assert result == "no-context-ok"


def test_retry_context_explicit_callback_overrides_context_var(monkeypatch):
    """Explicit retry_callback= arg takes precedence over _retry_context."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    context_calls: list = []
    explicit_calls: list = []

    attempt_n = 0

    def _fail_once(model):
        nonlocal attempt_n
        attempt_n += 1
        if attempt_n == 1:
            raise _make_rate_limit_error()
        return "ok"

    token = _retry_context.set(lambda a, m: context_calls.append((a, m)))
    try:
        with patch("app.llm.client.time.sleep"):
            _call_with_failover(
                _fail_once,
                retry_callback=lambda a, m: explicit_calls.append((a, m)),
            )
    finally:
        _retry_context.reset(token)

    # Explicit callback fired, context var callback did NOT fire
    assert len(explicit_calls) == 1
    assert len(context_calls) == 0


# ---------------------------------------------------------------------------
# 4. JobStore.update_retry_info
# ---------------------------------------------------------------------------


def test_job_store_update_retry_info_sets_message():
    store = jobs_module.JobStore()
    job = _make_job("ri-test-001")
    store.put(job)

    store.update_retry_info("ri-test-001", "Retrying after rate limit (attempt 2 of 4)")
    fetched = store.get("ri-test-001")
    assert fetched is not None
    assert fetched.retry_info == "Retrying after rate limit (attempt 2 of 4)"


def test_job_store_update_retry_info_clears_to_none():
    store = jobs_module.JobStore()
    job = _make_job("ri-test-002")
    store.put(job)

    store.update_retry_info("ri-test-002", "Retrying after rate limit (attempt 2 of 4)")
    store.update_retry_info("ri-test-002", None)
    fetched = store.get("ri-test-002")
    assert fetched is not None
    assert fetched.retry_info is None


def test_job_store_update_retry_info_noop_for_missing_job():
    store = jobs_module.JobStore()
    # Must not raise for a job_id that doesn't exist
    store.update_retry_info("nonexistent-job", "some info")


def test_job_retry_info_starts_as_none():
    job = _make_job("ri-test-003")
    assert job.retry_info is None


# ---------------------------------------------------------------------------
# 5. GET /api/recommend/{id} response includes retry_info when set
# ---------------------------------------------------------------------------


def test_api_response_includes_retry_info_when_set():
    """_job_response includes retry_info key only when it is not None."""
    import asyncio
    import httpx
    from app.api.main import app, jobs as api_jobs

    transport = httpx.ASGITransport(app=app)

    async def _do():
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get("/api/health")

    # Register a job with retry_info set
    job_id = "retry-info-api-test-001"
    job = _make_job(job_id)
    api_jobs.put(job)
    api_jobs.update_retry_info(job_id, "Retrying after rate limit (attempt 2 of 4)")

    async def _poll():
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get(f"/api/recommend/{job_id}")

    resp = asyncio.run(_poll())
    assert resp.status_code == 200
    body = resp.json()
    assert "retry_info" in body
    assert body["retry_info"] == "Retrying after rate limit (attempt 2 of 4)"


def test_api_response_omits_retry_info_when_none():
    """retry_info must NOT appear in the response when it is None."""
    import asyncio
    import httpx
    from app.api.main import app, jobs as api_jobs

    transport = httpx.ASGITransport(app=app)
    job_id = "retry-info-api-test-002"
    job = _make_job(job_id)
    api_jobs.put(job)
    # retry_info starts as None — must be omitted from response

    async def _poll():
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get(f"/api/recommend/{job_id}")

    resp = asyncio.run(_poll())
    assert resp.status_code == 200
    body = resp.json()
    assert "retry_info" not in body


def test_api_response_retry_info_cleared_after_update():
    """retry_info disappears from response after being cleared back to None."""
    import asyncio
    import httpx
    from app.api.main import app, jobs as api_jobs

    transport = httpx.ASGITransport(app=app)
    job_id = "retry-info-api-test-003"
    job = _make_job(job_id)
    api_jobs.put(job)

    # Set then clear
    api_jobs.update_retry_info(job_id, "Retrying after rate limit (attempt 2 of 4)")
    api_jobs.update_retry_info(job_id, None)

    async def _poll():
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get(f"/api/recommend/{job_id}")

    resp = asyncio.run(_poll())
    body = resp.json()
    assert "retry_info" not in body
