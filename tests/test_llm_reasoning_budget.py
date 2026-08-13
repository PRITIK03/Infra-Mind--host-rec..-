"""Regression tests for LLM reasoning-token budget wiring."""

from __future__ import annotations

import json

from app.config import get_llm_settings
from app.llm.client import _reasoning_effort_for_budget, get_chat_model
from langchain_core.messages import HumanMessage


def test_llm_settings_includes_reasoning_max_tokens(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    monkeypatch.setenv("LLM_REASONING_MAX_TOKENS", "2048")

    settings = get_llm_settings()
    assert settings.max_tokens == 8192
    assert settings.reasoning_max_tokens == 2048
    assert settings.reasoning_max_tokens < settings.max_tokens


def test_reasoning_budget_maps_to_openrouter_effort():
    # 2048/8192 ≈ 25% → low (~20% OpenRouter allocation)
    assert _reasoning_effort_for_budget(2048, 8192) == "low"
    assert _reasoning_effort_for_budget(4096, 8192) == "medium"
    assert _reasoning_effort_for_budget(7000, 8192) == "high"
    assert _reasoning_effort_for_budget(512, 8192) == "minimal"
    assert _reasoning_effort_for_budget(0, 8192) == "none"


def test_get_chat_model_includes_reasoning_cap_in_request_payload(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    monkeypatch.setenv("LLM_REASONING_MAX_TOKENS", "2048")

    get_chat_model.cache_clear()
    model = get_chat_model()

    payload = model._get_request_payload([HumanMessage(content="ping")], stop=None)
    # OpenRouter forbids setting both effort and max_tokens; effort is the
    # reliable control for openrouter/free routes.
    assert payload["extra_body"] == {"reasoning": {"effort": "low"}}

    merged = dict(payload)
    extra = merged.pop("extra_body", {})
    if isinstance(extra, dict):
        merged.update(extra)
    assert merged["reasoning"] == {"effort": "low"}
    assert "reasoning" in json.dumps(merged)
