"""
LLM client model factory for the AWS Instance Advisor agent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import get_llm_settings
from app.llm.retry import (  # re-exported for backward compatibility
    BACKOFF_BASE_S,
    BACKOFF_MAX_S,
    MAX_RATE_LIMIT_ATTEMPTS,
    RateLimitExhaustedError,
    _call_with_failover,
    _retry_context,
)
from app.llm.structured import (  # re-exported for backward compatibility
    StructuredOutputError,
    invoke_structured,
)


def _reasoning_effort_for_budget(reasoning_max_tokens: int, max_tokens: int) -> str:
    """
    Map a reasoning token budget onto OpenRouter's effort enum.

    OpenRouter allows either ``reasoning.max_tokens`` OR ``reasoning.effort``,
    not both. Several models reached via ``openrouter/free`` (notably some
    Cohere free routes) ignore ``reasoning.max_tokens`` and can still consume
    the entire completion budget on hidden reasoning. Effort is honored more
    broadly, so we derive it from ``reasoning_max_tokens / max_tokens``.
    """
    if max_tokens <= 0 or reasoning_max_tokens <= 0:
        return "none"
    ratio = reasoning_max_tokens / max_tokens
    # OpenRouter approximate allocations: high~80%, medium~50%, low~20%,
    # minimal~10%, none=off.
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.5:
        return "medium"
    if ratio >= 0.2:
        return "low"
    if ratio >= 0.05:
        return "minimal"
    return "none"


@lru_cache(maxsize=2)
def get_chat_model(use_secondary: bool = False) -> ChatOpenAI:
    """
    Returns a cached ChatOpenAI instance.

    Parameters
    ----------
    use_secondary:
        When True, build the client with `api_key_secondary`. Raises
        RuntimeError if no secondary key is configured.

    Notes
    -----
    ``max_retries=0`` disables the OpenAI SDK's internal retry loop.
    All retry logic (with hard bounds and explicit backoff) is handled
    by ``_call_with_failover`` so the SDK never silently sleeps and pins
    a worker thread on sustained 429s.

    ``extra_body["models"]`` is populated when LLM_FALLBACK_MODELS is
    configured. This passes OpenRouter's server-side model fallback
    array, which reroutes a request to the next model in the list when
    the primary is rate-limited or unavailable — a different mechanism
    from the dual-key failover (which handles per-account 429s). The
    two are additive: keys handle account quotas, this handles per-model
    congestion.
    """
    settings = get_llm_settings()
    api_key = settings.api_key

    if use_secondary:
        if not settings.api_key_secondary:
            raise RuntimeError(
                "Secondary API key requested but API_KEY_2 is not configured."
            )
        api_key = settings.api_key_secondary

    effort = _reasoning_effort_for_budget(
        settings.reasoning_max_tokens, settings.max_tokens
    )

    extra: dict[str, Any] = {"reasoning": {"effort": effort}}

    # OpenRouter server-side model fallback: if the primary model is
    # rate-limited or unavailable, OpenRouter tries each model in order.
    # Only added when LLM_FALLBACK_MODELS is configured — zero change to
    # request payload when the list is empty (backward compatible).
    if settings.fallback_models:
        extra["models"] = [settings.model_name, *settings.fallback_models]

    return ChatOpenAI(
        api_key=api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
        max_tokens=settings.max_tokens,
        # 0 = disabled: we own all retry logic in _call_with_failover.
        # The SDK's internal loop uses tenacity/httpx-retry and can sleep
        # for minutes under sustained 429s without ever raising to our code.
        max_retries=0,
        timeout=30,
        extra_body=extra,
    )
