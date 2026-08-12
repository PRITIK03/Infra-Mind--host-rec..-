"""
Requirement validator node.

Checks whether critical requirement fields are still missing and, if
so, prepares a follow-up question. If nothing critical is missing,
clears next_question so the graph can proceed to reasoning.
"""

from __future__ import annotations

from app.agent.state import AgentState
from app.models.schemas import WorkloadType

_FIELD_PROMPTS = {
    "workload_type": "What kind of application is this — a web app, an API service, batch processing, or ML inference?",
    "traffic_pattern": "Is traffic fairly steady throughout the day, or bursty with specific peak periods?",
}

_EXPECTED_SCALE_COMMON = (
    "For this workload, what is the expected scale? "
    "Provide one of: registered users, concurrent users, or peak RPS (requests per second)."
)

_EXPECTED_SCALE_BATCH = (
    "For this batch/backend workload, what is the expected scale? "
    "Provide one of: concurrent job/worker slots, or jobs processed per unit time."
)


def validate_requirements(state: AgentState) -> AgentState:
    missing = state["requirements"].missing_critical_fields()

    if not missing:
        state["next_question"] = None
        state["pending_field"] = None
        return state

    field = missing[0]
    if field == "expected_scale":
        workload_type = state["requirements"].workload_type
        if workload_type == WorkloadType.BATCH_PROCESSING:
            question = _EXPECTED_SCALE_BATCH
        else:
            question = _EXPECTED_SCALE_COMMON
    else:
        question = _FIELD_PROMPTS.get(
            field, f"Could you tell me more about {field.replace('_', ' ')}?"
        )

    state["pending_field"] = field
    state["next_question"] = question
    return state