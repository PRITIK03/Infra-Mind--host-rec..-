"""
Shared recommendation-validation helpers.

Extracted from recommender.py so the same anti-hallucination pattern
(validate recommended identifiers against the live candidate set, then
retry once with a correction note) can be reused by both the existing
instance recommender and the new holistic recommender.

Two validators are provided:

  validate_instance_recommendation
    Checks that recommended_instance and alternative_instance are in
    the set of allowed EC2 instance type strings.

  validate_cache_recommendation
    Checks the (instance_type, engine) PAIR against the live cache
    candidate set — not instance_type alone — because the same node-
    type name can exist under one engine but not another.

Both return the list of offending identifiers (empty = all good).
"""

from __future__ import annotations

from app.models.schemas import (
    CacheCandidate,
    CacheEngine,
    CacheRecommendation,
    InstanceRecommendation,
)


# ── InstanceRecommendation ─────────────────────────────────────────────────

def invalid_instance_types(
    result: InstanceRecommendation,
    allowed: set[str],
) -> list[str]:
    """Return invented EC2 instance types from the recommendation (if any)."""
    bad: list[str] = []
    if result.recommended_instance not in allowed:
        bad.append(result.recommended_instance)
    if result.alternative_instance is not None and result.alternative_instance not in allowed:
        bad.append(result.alternative_instance)
    return bad


def correction_prompt(
    base_prompt: str,
    *,
    invalid_instances: list[str],
    allowed_types: list[str],
) -> str:
    """Append a correction note to base_prompt listing the offending identifiers."""
    invalid_joined = ", ".join(repr(t) for t in invalid_instances)
    allowed_joined = ", ".join(allowed_types)
    return (
        f"{base_prompt}\n\n"
        f"Your previous answer suggested {invalid_joined}, which is not one "
        f"of the available options. You MUST choose recommended_instance and "
        f"alternative_instance only from this exact list: {allowed_joined}."
    )


# ── CacheRecommendation ────────────────────────────────────────────────────

def invalid_cache_pairs(
    result: CacheRecommendation,
    allowed_pairs: set[tuple[str, CacheEngine]],
) -> list[str]:
    """
    Return a description of any (instance_type, engine) pair in the
    recommendation that does not exist in the live candidate set.

    Checking the pair — not just the instance_type — is mandatory because
    the same node-type name can be valid under one engine and invalid under
    another (confirmed: cache.t3.medium only exists tagged Memcached in
    live data).
    """
    bad: list[str] = []
    if result.recommended_instance is not None and result.engine is not None:
        pair = (result.recommended_instance, result.engine)
        if pair not in allowed_pairs:
            bad.append(f"{result.recommended_instance!r} with engine {result.engine.value!r}")
    if (
        result.alternative_instance is not None
        and result.alternative_engine is not None
    ):
        alt_pair = (result.alternative_instance, result.alternative_engine)
        if alt_pair not in allowed_pairs:
            bad.append(
                f"{result.alternative_instance!r} with engine {result.alternative_engine.value!r}"
            )
    return bad


def cache_correction_prompt(
    base_prompt: str,
    *,
    invalid_descriptions: list[str],
    allowed_pairs: list[tuple[str, CacheEngine]],
) -> str:
    """Append a cache-specific correction note to base_prompt."""
    invalid_joined = "; ".join(invalid_descriptions)
    allowed_joined = ", ".join(
        f"{itype!r} (engine: {eng.value})" for itype, eng in sorted(allowed_pairs, key=lambda p: (p[0], p[1].value))
    )
    return (
        f"{base_prompt}\n\n"
        f"Your previous cache recommendation included {invalid_joined}, which "
        f"is not a valid (instance_type, engine) combination in the live data. "
        f"You MUST choose from these exact (instance_type, engine) pairs: "
        f"{allowed_joined}."
    )


def allowed_cache_pairs(candidates: list[CacheCandidate]) -> set[tuple[str, CacheEngine]]:
    """Build the set of valid (instance_type, engine) pairs from the live candidate list."""
    return {(c.instance_type, c.engine) for c in candidates}
