"""
Shared LangGraph state for the AWS Instance Advisor agent.
"""

from __future__ import annotations

from typing import TypedDict

from app.models.schemas import (
    CacheCandidate,
    DatabaseCandidate,
    InstanceCandidate,
    InstanceRecommendation,
    SystemDesignRecommendation,
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
    database_candidates: list[DatabaseCandidate] | None
    cache_candidates: list[CacheCandidate] | None
    recommendation: InstanceRecommendation | None
    system_design_recommendation: SystemDesignRecommendation | None