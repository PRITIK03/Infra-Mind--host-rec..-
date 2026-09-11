"""
Provider-agnostic structured output extraction and fallback logic.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from typing import Any, Callable, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from app.llm.retry import RateLimitExhaustedError, _call_with_failover

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when the LLM fails to produce valid structured output."""


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
# Keyed on (schema_name, sha256(normalized_prompt)).  "Normalized" means
# collapsed whitespace so minor formatting changes don't defeat the cache.
#
# TTL: 1 hour — long enough to absorb repeated identical requests during
# free-tier-constrained testing sessions; short enough that stale results
# don't persist across meaningful context changes.
#
# Max size: 200 entries.  At ~50 KB/entry (a realistic structured response)
# that's ~10 MB worst-case, well within a small cloud instance's headroom.
#
# Which calls are NOT cached
# --------------------------
# GroundingResult calls are deliberately excluded from caching even though
# they share this code path.  Grounding checks are cheap single-call audits
# whose correctness depends on the exact (recommendation, technical_needs)
# pair produced moments earlier — caching them could mask a freshly
# introduced inconsistency in the recommendation they're meant to catch.
# The exclusion is enforced by checking schema.__name__ == "GroundingResult"
# before the cache lookup.
#
# All other schemas (TechnicalNeeds, SystemDesignRecommendation, etc.) are
# cached: the same prompt content always warrants the same structured
# response within the TTL window.

_CACHE_TTL_S: float = 3600.0   # 1 hour
_CACHE_MAX_SIZE: int = 200

# Each entry: {"value": T, "expires_at": float}
_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()


def _cache_key(schema_name: str, prompt: str) -> str:
    """Stable cache key: schema name + SHA-256 of whitespace-normalised prompt."""
    normalised = re.sub(r"\s+", " ", prompt).strip()
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    return f"{schema_name}:{digest}"


def _cache_get(key: str) -> Any | None:
    """Return cached value if present and not expired, else None."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry["expires_at"]:
            del _cache[key]
            return None
        return entry["value"]


def _cache_put(key: str, value: Any) -> None:
    """Insert value; evict oldest entries when over capacity."""
    with _cache_lock:
        # Evict expired entries first (cheap scan, keeps the dict tidy).
        now = time.monotonic()
        expired = [k for k, v in _cache.items() if now > v["expires_at"]]
        for k in expired:
            del _cache[k]
        # If still over capacity, evict the entry with the smallest expires_at.
        while len(_cache) >= _CACHE_MAX_SIZE:
            oldest = min(_cache, key=lambda k: _cache[k]["expires_at"])
            del _cache[oldest]
        _cache[key] = {"value": value, "expires_at": now + _CACHE_TTL_S}


def _cache_clear() -> None:
    """Clear all cache entries.  Intended for tests only."""
    with _cache_lock:
        _cache.clear()


def _is_cacheable(schema: type) -> bool:
    """
    Return True when responses for *schema* should be cached.

    GroundingResult is excluded: each check audits a freshly produced
    recommendation and must never reuse a stale verdict.  All other
    structured schemas benefit from caching identical prompts.
    """
    return schema.__name__ != "GroundingResult"


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
    exactly once on parse/validation failure. Rate-limit errors
    propagate immediately after failover exhaustion.

    Both paths call ``_call_with_failover``, which reads the thread-local
    ``_retry_context`` ContextVar automatically — no explicit wiring
    needed from callers. The optional ``retry_callback`` kwarg overrides
    the context var and is mainly useful in tests.

    Response caching
    ----------------
    Identical (schema, normalised-prompt) pairs are served from an
    in-memory cache (TTL 1h, max 200 entries) to reduce redundant LLM
    calls during free-tier-constrained testing.  GroundingResult calls
    are intentionally excluded — see _is_cacheable() for rationale.
    Cache bypassing is automatic when retry_callback is supplied (tests
    that need fresh responses can pass a no-op lambda).
    """
    schema_name = schema.__name__
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    # ── Cache lookup (skipped for GroundingResult and explicit test overrides) ──
    use_cache = _is_cacheable(schema) and retry_callback is None
    cache_key: str | None = None
    if use_cache:
        cache_key = _cache_key(schema_name, prompt)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

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
        result = _run_attempt(prompt)
    except RateLimitExhaustedError:
        raise
    except Exception:
        # One bounded retry of the entire primary+fallback unit — do not loop.
        try:
            result = _run_attempt(f"{prompt}\n\n{_JSON_RETRY_NOTE}")
        except RateLimitExhaustedError:
            raise
        except StructuredOutputError:
            raise
        except Exception as retry_exc:
            raise StructuredOutputError(
                f"Failed to obtain valid {schema_name} from the LLM after retry. "
                f"Last error: {retry_exc}"
            ) from retry_exc

    # ── Cache store ────────────────────────────────────────────────────────
    if use_cache and cache_key is not None:
        _cache_put(cache_key, result)

    return result
