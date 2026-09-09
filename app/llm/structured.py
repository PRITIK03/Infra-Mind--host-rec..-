"""
Provider-agnostic structured output extraction and fallback logic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from app.llm.retry import RateLimitExhaustedError, _call_with_failover

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when the LLM fails to produce valid structured output."""


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
