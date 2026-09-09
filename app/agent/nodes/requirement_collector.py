"""
Requirement collector node.

Extracts structured UserRequirements fields from the user's latest
message using the LLM, merging newly found values into requirements
already gathered so far — never overwriting a known field with a
blank extraction.
"""

from __future__ import annotations

from app.agent.state import AgentState
from app.llm.client import StructuredOutputError, invoke_structured
from app.models.schemas import UserRequirements, WorkloadType


class RequirementExtractionError(RuntimeError):
    """Raised when the LLM fails to extract structured requirements."""


def _should_merge_field(current_value: object, extracted_value: object, default_value: object) -> bool:
    if extracted_value is None:
        return False
    if current_value is None:
        return True
    if current_value == default_value and extracted_value != default_value:
        return True
    return False


def collect_requirements(state: AgentState) -> AgentState:
    message = state.get("latest_user_message")
    if not message:
        return state

    pending_field = state.get("pending_field")

    workload_type = state["requirements"].workload_type

    guidance = ""
    if pending_field:
        guidance = f"\n\nThe user is answering a follow-up question about: {pending_field}."
        if pending_field == "expected_scale":
            if workload_type == WorkloadType.BATCH_PROCESSING:
                guidance += (
                    "\nInterpret a bare number as concurrent job/worker slots (batch scale). "
                    "Store it in estimated_concurrent_users."
                )
            else:
                guidance += (
                    "\nInterpret a bare number as the expected scale. "
                    "Store it in one of: registered_users, estimated_concurrent_users, or requests_per_second."
                )
        elif pending_field == "traffic_pattern":
            guidance += (
                "\nInterpret the answer as traffic pattern: steady or bursty."
            )

    prompt = (
        "Extract AWS workload requirements from this user message. "
        "Only fill fields that are explicitly stated or clearly implied; "
        "leave everything else unset.\n\n"
        "Interpretation rules:\n"
        "- If the user provides daily visitors, unique visitors, or total users (e.g. '50 visitors a day'), store the count in registered_users.\n"
        "- If the user provides concurrent jobs/workers/worker slots (common for "
        "batch or backend processing), store the number in estimated_concurrent_users.\n"
        "- If the user describes a batch job with no live users, store 0 in registered_users.\n"
        "- If the user provides peak RPS (requests per second), store it in requests_per_second.\n"
        f"User message: {message}"
        + guidance
    )
    try:
        extracted = invoke_structured(UserRequirements, prompt)
    except StructuredOutputError as exc:
        raise RequirementExtractionError(str(exc)) from exc

    current = state["requirements"]
    merged = current.model_dump()
    defaults = current.__class__.model_fields
    for field, value in extracted.model_dump().items():
        if _should_merge_field(merged.get(field), value, defaults[field].default):
            merged[field] = value

    state["requirements"] = UserRequirements(**merged)
    state["latest_user_message"] = None
    state["pending_field"] = None
    return state
