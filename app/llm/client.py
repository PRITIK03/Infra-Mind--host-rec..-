"""
LLM client wrapper for the AWS Instance Advisor agent.

Wraps a single OpenAI-compatible chat model (configured via API_KEY,
BASE_URL, MODEL_NAME) as a LangChain ChatOpenAI instance, and provides
provider-agnostic structured extraction that does not depend on OpenAI's
native json_schema structured-output API (unreliable on many OpenRouter
/ free models that return null `choices`).

Supports optional dual-key rate-limit failover: if API_KEY_2 is
configured the client will retry once using the secondary key when
OpenAI returns a 429 RateLimitError.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Callable, TypeVar

from langchain_openai import ChatOpenAI
from openai import RateLimitError
from pydantic import BaseModel, ValidationError

from app.config import get_llm_settings

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when the LLM fails to produce valid structured output."""


@lru_cache(maxsize=2)
def get_chat_model(use_secondary: bool = False) -> ChatOpenAI:
    """
    Returns a cached ChatOpenAI instance.

    Parameters
    ----------
    use_secondary:
        When True, build the client with `api_key_secondary`.  Raises
        RuntimeError if no secondary key is configured.
    """
    settings = get_llm_settings()
    api_key = settings.api_key

    if use_secondary:
        if not settings.api_key_secondary:
            raise RuntimeError(
                "Secondary API key requested but API_KEY_2 is not configured."
            )
        api_key = settings.api_key_secondary

    return ChatOpenAI(
        api_key=api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
        max_tokens=settings.max_tokens,
        max_retries=3,
        timeout=30,
    )


def _call_with_failover(fn: Callable[[ChatOpenAI], T]) -> T:
    """
    Tries *fn* with the primary-key model first.  On
    ``openai.RateLimitError`` specifically (not other exceptions),
    retries **once** using the secondary-key model IF one is configured.
    Any other exception, or exhausting both keys, propagates
    immediately — errors are never swallowed.
    """
    try:
        return fn(get_chat_model(use_secondary=False))
    except RateLimitError:
        settings = get_llm_settings()
        if not settings.api_key_secondary:
            raise  # no secondary key configured — propagate the 429
        print("[failover] Primary key rate-limited — retrying with secondary key …")
        return fn(get_chat_model(use_secondary=True))


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


def invoke_structured(schema: type[T], prompt: str) -> T:
    """
    Provider-agnostic structured extraction into a Pydantic schema.

    Primary path: plain chat completion + JSON object + Pydantic validation.
    This avoids OpenAI `json_schema` structured-output / tool-call APIs that
    many OpenAI-compatible providers (including OpenRouter free models) handle
    poorly, often returning `choices=None` and crashing LangChain parsers.

    Secondary path: LangChain `with_structured_output(..., method="json_mode")`
    if the primary path fails — still validated as the schema type when possible.

    Both paths are routed through _call_with_failover() for automatic
    rate-limit failover when a secondary API key is configured.
    """
    schema_name = schema.__name__
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    json_prompt = (
        f"{prompt}\n\n"
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
            try:
                normalized = _normalize_instance_recommendation_payload(schema, data)
                return schema.model_validate(normalized)
            except ValidationError:
                # Narrow contract-mismatch handling for known wrapper shape:
                # { "reasoning": "...", "technical_needs": { ...TechnicalNeeds... } }
                # Some models do this despite the schema request. We only
                # unwrap TechnicalNeeds and only when the unwrapped payload
                # validates; otherwise we re-raise the original validation error.
                if schema.__name__ == "TechnicalNeeds" and isinstance(data, dict):
                    inner = data.get("technical_needs")
                    if isinstance(inner, dict):
                        candidate = dict(inner)
                        if "reasoning" in data and "reasoning" not in candidate:
                            candidate["reasoning"] = data["reasoning"]
                        return schema.model_validate(candidate)
                raise

        return _call_with_failover(_primary)
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        primary_error = exc
    except RateLimitError:
        raise  # already exhausted both keys in _call_with_failover
    except Exception as exc:
        # Provider/SDK parse failures (e.g. choices=None) land here.
        primary_error = exc

    # Fallback: json_mode structured output (more compatible than json_schema).
    try:
        def _fallback(model: ChatOpenAI):
            structured = model.with_structured_output(schema, method="json_mode")
            result = structured.invoke(
                f"{prompt}\n\nRespond with a JSON object matching the {schema_name} schema."
            )
            if result is None:
                raise StructuredOutputError(
                    f"LLM returned no {schema_name} object (empty structured response)."
                )
            if isinstance(result, schema):
                return result
            normalized = _normalize_instance_recommendation_payload(schema, result)
            return schema.model_validate(normalized)

        return _call_with_failover(_fallback)
    except StructuredOutputError:
        raise
    except Exception as fallback_exc:
        raise StructuredOutputError(
            f"Failed to obtain valid {schema_name} from the LLM. "
            f"JSON extraction error: {primary_error}. "
            f"Structured-output fallback error: {fallback_exc}"
        ) from fallback_exc
