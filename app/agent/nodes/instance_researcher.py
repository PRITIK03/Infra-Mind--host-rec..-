"""
Instance researcher node.

Fetches live EC2 instance data and filters it down to families that
plausibly match the technical needs from the reasoner — by resource
profile (CPU/memory-bound), GPU requirement (from live GPU fields),
and rough load size — rather than handing the recommender every
instance type that exists.
"""

from __future__ import annotations

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

# Keep the recommender prompt focused; live catalogs are large.
_MAX_CANDIDATES = 48


def _family_of(instance_type: str) -> str:
    return instance_type.split(".", 1)[0].lower()


def _family_matches(instance_type: str, prefixes: tuple[str, ...]) -> bool:
    family = _family_of(instance_type)
    return any(family == prefix or family.startswith(prefix) for prefix in prefixes)


def _mem_per_vcpu(candidate: InstanceCandidate) -> float:
    if candidate.vcpu <= 0:
        return 0.0
    return float(candidate.memory_gib) / float(candidate.vcpu)


def _target_vcpu_band(needs: TechnicalNeeds) -> tuple[int, int]:
    """
    Soft sizing band from concurrency + scaling intent.
    Used only to prefer relevant sizes, not as a hard architecture rule.
    """
    concurrency = max(int(needs.estimated_concurrency), 1)
    horizontal = "horizontal" in (needs.scaling_recommendation or "").lower()
    # Rough web/API heuristic: ~20–40 concurrent units per vCPU.
    units_per_vcpu = 30
    if horizontal:
        # Size a share of peak across a few replicas, not the whole fleet on one box.
        per_instance_load = max(concurrency / 4.0, 1.0)
        mid = max(int(per_instance_load / units_per_vcpu), 2)
    else:
        mid = max(int(concurrency / units_per_vcpu), 2)
    return max(mid // 4, 1), max(mid * 4, mid + 2)


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


def _narrow_by_size(
    candidates: list[InstanceCandidate],
    needs: TechnicalNeeds,
) -> list[InstanceCandidate]:
    lo, hi = _target_vcpu_band(needs)
    in_band = [c for c in candidates if lo <= c.vcpu <= hi]
    if len(in_band) >= 8:
        return in_band
    # Expand gradually so sparse catalogs still produce options.
    wider = [c for c in candidates if max(lo // 2, 1) <= c.vcpu <= hi * 2]
    return wider or candidates


def _select_diverse(candidates: list[InstanceCandidate], limit: int = _MAX_CANDIDATES) -> list[InstanceCandidate]:
    """Prefer a spread of sizes over dumping near-duplicates into the LLM."""
    if len(candidates) <= limit:
        return sorted(candidates, key=lambda c: (c.vcpu, c.memory_gib, c.instance_type))

    ordered = sorted(candidates, key=lambda c: (c.vcpu, c.memory_gib, c.instance_type))
    if limit == 1:
        return [ordered[len(ordered) // 2]]

    selected: list[InstanceCandidate] = []
    seen_types: set[str] = set()
    for index in range(limit):
        pos = round(index * (len(ordered) - 1) / (limit - 1))
        candidate = ordered[pos]
        if candidate.instance_type in seen_types:
            continue
        selected.append(candidate)
        seen_types.add(candidate.instance_type)

    # Fill remaining slots from unused mid-range types if sampling collided.
    if len(selected) < limit:
        for candidate in ordered:
            if candidate.instance_type in seen_types:
                continue
            selected.append(candidate)
            seen_types.add(candidate.instance_type)
            if len(selected) >= limit:
                break

    return sorted(selected, key=lambda c: (c.vcpu, c.memory_gib, c.instance_type))


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
