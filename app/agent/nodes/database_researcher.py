"""
Database researcher node.

Fetches live RDS instance-class data and filters it down to a focused
candidate set appropriate for the workload's technical needs.

RDS exposes a clean ``family`` field with only three values:
  - "General purpose"
  - "Memory optimized"
  - "Micro instances"

Filtering is therefore entirely data-driven from that field — no
hardcoded prefix guessing required, unlike the EC2 researcher.

Size narrowing and diversity sampling reuse the generic helpers from
_candidate_utils, which operate on any type that has ``vcpu``,
``memory_gib``, and ``instance_type`` attributes.
"""

from __future__ import annotations

from app.agent.nodes._candidate_utils import (
    narrow_by_size as _narrow_by_size,
    select_diverse as _select_diverse,
)
from app.agent.state import AgentState
from app.models.schemas import DatabaseCandidate, ResourceProfile, TechnicalNeeds
from app.tools.aws_instance_data import fetch_rds_instance_data

# Low-concurrency threshold: at or below this, micro instances are worth
# including (mirrors the single-digit threshold used in instance_researcher).
_MICRO_CONCURRENCY_THRESHOLD = 10

_FAMILY_GENERAL = "General purpose"
_FAMILY_MEMORY = "Memory optimized"
_FAMILY_MICRO = "Micro instances"


class DatabaseResearchError(RuntimeError):
    """Raised when no plausible RDS instance candidates can be found."""


def _target_families(needs: TechnicalNeeds) -> list[str]:
    """
    Return the ordered list of RDS family labels to include, based on
    the workload's resource profile and estimated concurrency.

    - MEMORY_BOUND  → prefer Memory optimized; General purpose as fallback
    - very low concurrency → include Micro instances (+ General purpose)
    - otherwise → General purpose only
    """
    if needs.resource_profile == ResourceProfile.MEMORY_BOUND:
        return [_FAMILY_MEMORY, _FAMILY_GENERAL]
    if needs.estimated_concurrency <= _MICRO_CONCURRENCY_THRESHOLD:
        return [_FAMILY_MICRO, _FAMILY_GENERAL]
    return [_FAMILY_GENERAL]


def _filter_by_family(
    all_instances: list[DatabaseCandidate],
    needs: TechnicalNeeds,
) -> list[DatabaseCandidate]:
    """Filter to the target RDS families; fall back to all if nothing matched."""
    families = set(_target_families(needs))
    matched = [c for c in all_instances if c.vcpu > 0 and c.family in families]
    return matched or [c for c in all_instances if c.vcpu > 0]


def research_database(state: AgentState) -> AgentState:
    needs: TechnicalNeeds | None = state["technical_needs"]
    if needs is None:
        raise DatabaseResearchError(
            "technical_needs must be set before researching database instances."
        )

    # Skip entirely when the workload doesn't need a relational database.
    if not needs.needs_database:
        state["database_candidates"] = []
        return state

    all_instances = fetch_rds_instance_data()
    family_matched = _filter_by_family(all_instances, needs)
    if not family_matched:
        raise DatabaseResearchError(
            "No RDS instance candidates found for the derived technical needs."
        )

    sized = _narrow_by_size(family_matched, needs)
    state["database_candidates"] = _select_diverse(sized)
    return state
