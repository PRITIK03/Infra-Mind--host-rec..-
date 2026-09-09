"""
LLM client wrapper for the AWS Instance Advisor agent.

Wraps a single OpenAI-compatible chat model (configured via API_KEY,
BASE_URL, MODEL_NAME) as a LangChain ChatOpenAI instance, and provides
provider-agnostic structured extraction that does not depend on OpenAI's
native json_schema structured-output API (unreliable on many OpenRouter
/ free models that return null `choices`).

Rate-limit retry strategy
--------------------------
We set ``max_retries=0`` on ChatOpenAI to disable the OpenAI SDK's own
internal retry loop entirely.  That loop uses tenacity/httpx-retry with
exponential backoff *inside* the SDK, meaning it never surfaces a
``RateLimitError`` to our code — it just sleeps, potentially for
minutes, pinning the worker thread indefinitely under sustained 429s.

Instead, ``_call_with_failover`` owns all retry logic with explicit
hard bounds:

  - Up to MAX_RATE_LIMIT_ATTEMPTS total attempts (primary and secondary
    keys combined, interleaved so both keys get a chance on each round).
  - Exponential backoff between attempts: BACKOFF_BASE_S * 2^attempt,
    capped at BACKOFF_MAX_S.  This is wall-clock sleep that the caller
    can reason about.
  - After all attempts are exhausted, raises RateLimitExhaustedError
    with a user-facing message suitable for surfacing in the API response.

This means the TOTAL worst-case wall-clock time for one logical LLM call
under sustained rate-limiting is bounded and predictable:
  sum(min(BACKOFF_BASE_S * 2^i, BACKOFF_MAX_S) for i in range(attempts))
"""

from __future__ import annotations

import json
import re
import time
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Callable, TypeVar

from langchain_openai import ChatOpenAI
from openai import RateLimitError
from pydantic import BaseModel, ValidationError

from app.config import get_llm_settings

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Rate-limit retry constants — all in one place for easy tuning.
# ---------------------------------------------------------------------------

# Total attempts across BOTH keys combined.  With 2 keys and 4 attempts,
# each key gets at most 2 tries.  With 1 key, it gets all 4 tries.
MAX_RATE_LIMIT_ATTEMPTS: int = 4

# Initial sleep before the second attempt (doubles each round, capped below).
BACKOFF_BASE_S: float = 2.0

# Hard ceiling on any single inter-attempt sleep to avoid very long waits.
BACKOFF_MAX_S: float = 16.0

# ---------------------------------------------------------------------------
# Thread-local retry context
# ---------------------------------------------------------------------------

# A ContextVar holding an optional callback (attempt, max_attempts) -> None.
# Set by api/main.py's _run_graph_with_streaming for the duration of a graph
# pass so that every invoke_structured call in that thread automatically
# updates job.retry_info without needing changes to node function signatures.
# Defaults to None (no-op) — safe to call from CLI or tests without any setup.
_retry_context: ContextVar[Callable[[int, int], None] | None] = ContextVar(
    "_retry_context", default=None
)


class StructuredOutputError(RuntimeError):
    """Raised when the LLM fails to produce valid structured output."""


class RateLimitExhaustedError(RuntimeError):
    """
    Raised when all rate-limit retry attempts across all configured keys
    are exhausted without a successful response.

    The message is intentionally user-facing and safe to surface directly
    in an API error response.
    """


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
        When True, build the client with `api_key_secondary`.  Raises
        RuntimeError if no secondary key is configured.

    Notes
    -----
    ``max_retries=0`` disables the OpenAI SDK's internal retry loop.
    All retry logic (with hard bounds and explicit backoff) is handled
    by ``_call_with_failover`` so the SDK never silently sleeps and pins
    a worker thread on sustained 429s.

    ``extra_body["models"]`` is populated when LLM_FALLBACK_MODELS is
    configured.  This passes OpenRouter's server-side model fallback
    array, which reroutes a request to the next model in the list when
    the primary is rate-limited or unavailable — a different mechanism
    from the dual-key failover (which handles per-account 429s).  The
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


def _call_with_failover(
    fn: Callable[[ChatOpenAI], T],
    *,
    retry_callback: Callable[[int, int], None] | None = None,
) -> T:
    """
    Call *fn* with bounded retry and dual-key failover on RateLimitError.

    Attempt schedule (MAX_RATE_LIMIT_ATTEMPTS=4, two keys configured):
      attempt 0 — primary key   (no sleep before)
      attempt 1 — secondary key (sleep BACKOFF_BASE_S * 2^0 = 2s)
      attempt 2 — primary key   (sleep BACKOFF_BASE_S * 2^1 = 4s)
      attempt 3 — secondary key (sleep BACKOFF_BASE_S * 2^2 = 8s)
      → RateLimitExhaustedError

    With only one key configured every attempt uses the primary key.

    Non-RateLimitError exceptions propagate immediately without retrying —
    errors are never swallowed.

    Parameters
    ----------
    fn:
        Callable that receives a ChatOpenAI model and returns a result.
    retry_callback:
        Optional explicit callback invoked at the start of each retry
        attempt (attempt > 0) with (current_attempt, max_attempts).
        If None, falls back to the thread-local _retry_context value so
        callers don't need to thread the callback manually — it's set
        once per graph pass by api/main.py.
    """
    settings = get_llm_settings()
    has_secondary = bool(settings.api_key_secondary)
    last_exc: RateLimitError | None = None

    # Resolve callback: explicit arg wins; fall back to context var.
    cb = retry_callback if retry_callback is not None else _retry_context.get()

    for attempt in range(MAX_RATE_LIMIT_ATTEMPTS):
        # Interleave keys: even attempts → primary, odd → secondary (if available).
        use_secondary = has_secondary and (attempt % 2 == 1)
        if attempt > 0:
            sleep_s = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)
            key_label = "secondary" if use_secondary else "primary"
            print(
                f"[rate-limit] attempt {attempt + 1}/{MAX_RATE_LIMIT_ATTEMPTS} "
                f"(key={key_label}, backoff={sleep_s:.0f}s) …"
            )
            if cb is not None:
                cb(attempt + 1, MAX_RATE_LIMIT_ATTEMPTS)
            time.sleep(sleep_s)
        try:
            return fn(get_chat_model(use_secondary=use_secondary))
        except RateLimitError as exc:
            last_exc = exc
            continue  # try next attempt

    raise RateLimitExhaustedError(
        "Rate limited after multiple attempts — please try again in a few minutes."
    ) from last_exc


def _message_text(response: object) -> str:
    content = getattr(response, "content", response)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _parse_json_object(text: str) -> object:
    """
    Extract a JSON object from model text. Allows optional markdown fences
    or leading/trailing prose; final acceptance is always Pydantic validation.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("LLM returned empty content")

    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("LLM response did not contain a JSON object") from None
        return json.loads(cleaned[start : end + 1])


def _normalize_instance_recommendation_payload(
    schema: type[T], data: object
) -> object:
    """
    Apply narrow, schema-specific compatibility fixes for known provider drift.

    This is intentionally limited to InstanceRecommendation so other schemas
    keep their own validation and failure behavior.
    """
    if schema.__name__ != "InstanceRecommendation" or not isinstance(data, dict):
        return data

    normalized: dict[str, Any] = dict(data)
    schema_fields = set(schema.model_fields)

    if "confidence" not in normalized and "confidence_level" in normalized:
        normalized["confidence"] = normalized["confidence_level"]
        normalized.pop("confidence_level", None)

    if (
        "why" not in normalized
        and "reasoning" in normalized
        and "why" in schema_fields
        and "reasoning" not in schema_fields
    ):
        normalized["why"] = normalized["reasoning"]
        normalized.pop("reasoning", None)

    assumptions = normalized.get("assumptions")
    if isinstance(assumptions, str) and assumptions.strip():
        normalized["assumptions"] = [assumptions]

    return normalized


def _validate_structured_payload(schema: type[T], data: object) -> T:
    """
    Validate parsed JSON against *schema*, with narrow TechnicalNeeds unwrap.
    """
    try:
        normalized = _normalize_instance_recommendation_payload(schema, data)
        return schema.model_validate(normalized)
    except ValidationError:
        # Narrow contract-mismatch handling for known wrapper shape:
        # { "reasoning": "...", "technical_needs": { ...TechnicalNeeds... } }
        if schema.__name__ == "TechnicalNeeds" and isinstance(data, dict):
            inner = data.get("technical_needs")
            if isinstance(inner, dict):
                candidate = dict(inner)
                if "reasoning" in data and "reasoning" not in candidate:
                    candidate["reasoning"] = data["reasoning"]
                return schema.model_validate(candidate)
        raise


_JSON_RETRY_NOTE = (
    "Your previous response did not contain valid JSON matching the required "
    "schema — it may have included non-JSON content or been incomplete. "
    "Respond with ONLY a valid JSON object matching the schema. No other text, "
    "labels, or commentary."
)


def invoke_structured(
    schema: type[T],
    prompt: str,
    *,
    retry_callback: Callable[[int, int], None] | None = None,
) -> T:
    """
    Provider-agnostic structured extraction into a Pydantic schema.

    Primary path: plain chat completion + JSON object + Pydantic validation.
    Secondary path: LangChain ``with_structured_output(..., method="json_mode")``
    if the primary path fails.

    The primary + fallback pair is treated as one attempt and retried
    exactly once on parse/validation failure.  Rate-limit errors
    propagate immediately after failover exhaustion.

    Both paths call ``_call_with_failover``, which reads the thread-local
    ``_retry_context`` ContextVar automatically — no explicit wiring
    needed from callers.  The optional ``retry_callback`` kwarg overrides
    the context var and is mainly useful in tests.
    """
    schema_name = schema.__name__
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    def _run_attempt(active_prompt: str) -> T:
        json_prompt = (
            f"{active_prompt}\n\n"
            f"Return a single JSON object only (no markdown fences, no commentary) "
            f"that conforms to this JSON Schema for {schema_name}:\n"
            f"{schema_json}\n"
            f"Omit fields that are unknown or not supported by the evidence; "
            f"do not invent values."
        )

        primary_error: Exception | None = None
        try:
            def _primary(model: ChatOpenAI):
                response = model.invoke(json_prompt)
                text = _message_text(response)
                data = _parse_json_object(text)
                return _validate_structured_payload(schema, data)

            return _call_with_failover(_primary, retry_callback=retry_callback)
        except RateLimitExhaustedError:
            raise
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            primary_error = exc
        except Exception as exc:
            # Provider/SDK parse failures (e.g. choices=None) land here.
            primary_error = exc

        try:
            def _fallback(model: ChatOpenAI):
                structured = model.with_structured_output(schema, method="json_mode")
                result = structured.invoke(
                    f"{active_prompt}\n\n"
                    f"Respond with a JSON object matching the {schema_name} schema."
                )
                if result is None:
                    raise StructuredOutputError(
                        f"LLM returned no {schema_name} object (empty structured response)."
                    )
                if isinstance(result, schema):
                    return result
                normalized = _normalize_instance_recommendation_payload(schema, result)
                return schema.model_validate(normalized)

            return _call_with_failover(_fallback, retry_callback=retry_callback)
        except RateLimitExhaustedError:
            raise
        except StructuredOutputError:
            raise
        except Exception as fallback_exc:
            raise StructuredOutputError(
                f"Failed to obtain valid {schema_name} from the LLM. "
                f"JSON extraction error: {primary_error}. "
                f"Structured-output fallback error: {fallback_exc}"
            ) from fallback_exc

    try:
        return _run_attempt(prompt)
    except RateLimitExhaustedError:
        raise
    except Exception:
        # One bounded retry of the entire primary+fallback unit — do not loop.
        try:
            return _run_attempt(f"{prompt}\n\n{_JSON_RETRY_NOTE}")
        except RateLimitExhaustedError:
            raise
        except StructuredOutputError:
            raise
        except Exception as retry_exc:
            raise StructuredOutputError(
                f"Failed to obtain valid {schema_name} from the LLM after retry. "
                f"Last error: {retry_exc}"
            ) from retry_exc
