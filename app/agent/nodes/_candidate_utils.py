"""
Shared candidate-narrowing helpers for researcher nodes.

These helpers are generic over any candidate type that exposes
``vcpu`` (int) and ``memory_gib`` (float) attributes — currently
InstanceCandidate and DatabaseCandidate.  Keeping them in one place
ensures the two researchers apply identical sizing and diversity logic.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from app.models.schemas import TechnicalNeeds

# Keep the recommender prompt focused; live catalogs are large.
MAX_CANDIDATES = 48


class _HasSize(Protocol):
    vcpu: int
    memory_gib: float
    instance_type: str


C = TypeVar("C", bound=_HasSize)


def target_vcpu_band(needs: TechnicalNeeds) -> tuple[int, int]:
    """
    Soft vCPU sizing band derived from concurrency + scaling intent.
    Used only to prefer relevant sizes, not as a hard architecture rule.
    """
    concurrency = max(int(needs.estimated_concurrency), 1)
    horizontal = "horizontal" in (needs.scaling_recommendation or "").lower()
    # Rough web/API heuristic: ~20–40 concurrent units per vCPU.
    units_per_vcpu = 30
    if horizontal:
        per_instance_load = max(concurrency / 4.0, 1.0)
        mid = max(int(per_instance_load / units_per_vcpu), 2)
    else:
        mid = max(int(concurrency / units_per_vcpu), 2)
    return max(mid // 4, 1), max(mid * 4, mid + 2)


def narrow_by_size(candidates: list[C], needs: TechnicalNeeds) -> list[C]:
    """Prefer candidates whose vCPU count falls in the soft band for this load."""
    lo, hi = target_vcpu_band(needs)
    in_band = [c for c in candidates if lo <= c.vcpu <= hi]
    if len(in_band) >= 8:
        return in_band
    # Expand gradually so sparse catalogs still produce options.
    wider = [c for c in candidates if max(lo // 2, 1) <= c.vcpu <= hi * 2]
    return wider or candidates


def select_diverse(candidates: list[C], limit: int = MAX_CANDIDATES) -> list[C]:
    """Return at most *limit* candidates spread across the size range."""
    if len(candidates) <= limit:
        return sorted(candidates, key=lambda c: (c.vcpu, c.memory_gib, c.instance_type))

    ordered = sorted(candidates, key=lambda c: (c.vcpu, c.memory_gib, c.instance_type))
    if limit == 1:
        return [ordered[len(ordered) // 2]]

    selected: list[C] = []
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
