"""
Cache researcher node.

Fetches live ElastiCache node-type data and filters it down to a focused
candidate set appropriate for the workload's technical needs.

Key design decisions baked in here:
- Engine choice (Redis vs Memcached vs Valkey) is NOT decided at this
  stage.  Engine belongs to the final recommender's reasoning, exactly
  the way alternative_instance is decided at recommendation time rather
  than being baked in early.  This node provides real options for each
  engine so the recommender can reason between them.
- Because node types are engine-specific (a given type may only exist
  under one engine), diversity sampling MUST run per-engine group, then
  the results are combined.  Running a flat sample on the mixed list
  risks silently dropping an entire engine if another engine has more
  entries in the relevant size band.
- Family filtering is data-driven from the ``family`` field: two values
  are relevant here — "Standard" and "Memory optimized".
"""

from __future__ import annotations

from collections import defaultdict

from app.agent.nodes._candidate_utils import (
    narrow_by_size as _narrow_by_size,
    select_diverse as _select_diverse,
)
from app.agent.state import AgentState
from app.models.schemas import CacheCandidate, CacheEngine, ResourceProfile, TechnicalNeeds
from app.tools.aws_instance_data import fetch_cache_instance_data

_FAMILY_STANDARD = "Standard"
_FAMILY_MEMORY = "Memory optimized"

# Per-engine candidate cap — keeps the recommender prompt manageable while
# guaranteeing representation for each engine.
_MAX_PER_ENGINE = 16


class CacheResearchError(RuntimeError):
    """Raised when no plausible ElastiCache node candidates can be found."""


def _target_family(needs: TechnicalNeeds) -> str:
    """
    Single target family based on resource profile.
    MEMORY_BOUND → Memory optimized; everything else → Standard.
    """
    if needs.resource_profile == ResourceProfile.MEMORY_BOUND:
        return _FAMILY_MEMORY
    return _FAMILY_STANDARD


def _filter_by_family(
    all_nodes: list[CacheCandidate],
    needs: TechnicalNeeds,
) -> list[CacheCandidate]:
    """Filter to the target cache family; fall back to all valid nodes if nothing matched."""
    target = _target_family(needs)
    matched = [c for c in all_nodes if c.vcpu > 0 and c.family == target]
    return matched or [c for c in all_nodes if c.vcpu > 0]


def _sample_per_engine(
    candidates: list[CacheCandidate],
    needs: TechnicalNeeds,
    max_per_engine: int = _MAX_PER_ENGINE,
) -> list[CacheCandidate]:
    """
    Group candidates by engine, run narrow_by_size + select_diverse on each
    group independently, then concatenate.  This guarantees every engine that
    survived family filtering is represented in the final result regardless of
    how uneven the per-engine counts are.
    """
    by_engine: dict[CacheEngine, list[CacheCandidate]] = defaultdict(list)
    for c in candidates:
        by_engine[c.engine].append(c)

    result: list[CacheCandidate] = []
    for engine_candidates in by_engine.values():
        sized = _narrow_by_size(engine_candidates, needs)
        result.extend(_select_diverse(sized, limit=max_per_engine))
    return result


def research_cache(state: AgentState) -> AgentState:
    needs: TechnicalNeeds | None = state["technical_needs"]
    if needs is None:
        raise CacheResearchError(
            "technical_needs must be set before researching cache nodes."
        )

    # Skip entirely when the workload doesn't benefit from a caching layer.
    if not needs.needs_cache:
        state["cache_candidates"] = []
        return state

    all_nodes = fetch_cache_instance_data()
    family_matched = _filter_by_family(all_nodes, needs)
    if not family_matched:
        raise CacheResearchError(
            "No ElastiCache node candidates found for the derived technical needs."
        )

    state["cache_candidates"] = _sample_per_engine(family_matched, needs)
    return state
