"""
Shared LangGraph state for the AWS Instance Advisor agent.
"""

from __future__ import annotations

from typing import TypedDict

from app.models.schemas import (
    InstanceCandidate,
    InstanceRecommendation,
    TechnicalNeeds,
    UserRequirements,
)


class AgentState(TypedDict):
    requirements: UserRequirements
    latest_user_message: str | None
    next_question: str | None
    pending_field: str | None
    technical_needs: TechnicalNeeds | None
    instance_candidates: list[InstanceCandidate] | None
    recommendation: InstanceRecommendation | None