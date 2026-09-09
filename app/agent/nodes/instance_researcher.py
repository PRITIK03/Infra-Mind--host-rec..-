"""
Instance researcher node.

Fetches live EC2 instance data and filters it down to families that
plausibly match the technical needs from the reasoner — by resource
profile (CPU/memory-bound), GPU requirement (from live GPU fields),
and rough load size — rather than handing the recommender every
instance type that exists.
"""

from __future__ import annotations

from app.agent.nodes._candidate_utils import (
    MAX_CANDIDATES as _MAX_CANDIDATES,
    narrow_by_size as _narrow_by_size,
    select_diverse as _select_diverse,
)
from app.agent.state import AgentState
from app.models.schemas import InstanceCandidate, ResourceProfile, TechnicalNeeds
from app.tools.aws_instance_data import fetch_ec2_instance_data


class InstanceResearchError(RuntimeError):
    """Raised when no plausible instance candidates can be found."""


# Prefixes, not exact family codes — live types use suffixes like g4dn, p4d, m6i.
_MEMORY_FAMILY_PREFIXES = ("r5", "r6", "r7", "r8", "x1", "x2", "u-")
_COMPUTE_FAMILY_PREFIXES = ("c5", "c6", "c7", "c8", "hpc")
_BALANCED_FAMILY_PREFIXES = ("m5", "m6", "m7", "m8")
_BURSTABLE_FAMILY_PREFIXES = ("t2", "t3", "t4")


def _family_of(instance_type: str) -> str:
    return instance_type.split(".", 1)[0].lower()


def _family_matches(instance_type: str, prefixes: tuple[str, ...]) -> bool:
    family = _family_of(instance_type)
    return any(family == prefix or family.startswith(prefix) for prefix in prefixes)


def _mem_per_vcpu(candidate: InstanceCandidate) -> float:
    if candidate.vcpu <= 0:
        return 0.0
    return float(candidate.memory_gib) / float(candidate.vcpu)


def _profile_prefixes(needs: TechnicalNeeds) -> tuple[str, ...] | None:
    """
    Family prefixes for non-GPU paths. GPU uses live gpu_count instead.
    Returns None when the GPU path should be used.
    """
    if needs.requires_gpu or needs.resource_profile == ResourceProfile.GPU_BOUND:
        return None
    if needs.resource_profile == ResourceProfile.MEMORY_BOUND:
        return _MEMORY_FAMILY_PREFIXES
    if needs.resource_profile == ResourceProfile.CPU_BOUND:
        return _COMPUTE_FAMILY_PREFIXES
    if needs.estimated_concurrency <= 50:
        return _BALANCED_FAMILY_PREFIXES + _BURSTABLE_FAMILY_PREFIXES
    return _BALANCED_FAMILY_PREFIXES


def _filter_by_profile(
    all_instances: list[InstanceCandidate],
    needs: TechnicalNeeds,
) -> list[InstanceCandidate]:
    prefixes = _profile_prefixes(needs)
    if prefixes is None:
        # Data-driven GPU path: trust live GPU fields, not hardcoded family names.
        return [c for c in all_instances if c.gpu_count > 0 and c.vcpu > 0]

    matched = [
        c
        for c in all_instances
        if c.vcpu > 0 and c.gpu_count == 0 and _family_matches(c.instance_type, prefixes)
    ]
    if matched:
        return matched

    # Fallback: ratio-based filter from live specs if family prefixes miss.
    if needs.resource_profile == ResourceProfile.MEMORY_BOUND:
        return [c for c in all_instances if c.vcpu > 0 and c.gpu_count == 0 and _mem_per_vcpu(c) >= 6.0]
    if needs.resource_profile == ResourceProfile.CPU_BOUND:
        return [c for c in all_instances if c.vcpu > 0 and c.gpu_count == 0 and _mem_per_vcpu(c) <= 3.0]
    return [c for c in all_instances if c.vcpu > 0 and c.gpu_count == 0 and 2.5 <= _mem_per_vcpu(c) <= 5.5]


def research_instances(state: AgentState) -> AgentState:
    needs = state["technical_needs"]
    if needs is None:
        raise InstanceResearchError("technical_needs must be set before researching instances.")

    all_instances = fetch_ec2_instance_data()
    profile_matched = _filter_by_profile(all_instances, needs)
    if not profile_matched:
        raise InstanceResearchError(
            "No EC2 instance candidates found for the derived technical needs."
        )

    sized = _narrow_by_size(profile_matched, needs)
    state["instance_candidates"] = _select_diverse(sized)
    return state
