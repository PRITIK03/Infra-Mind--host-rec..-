"""
Holistic recommender node.

Produces a SystemDesignRecommendation in a single LLM call that reasons
about compute, database, and cache together — deliberately NOT split into
three isolated calls — because independent reasoning steps can produce
contradictions (e.g. the batch-job scaling mismatch already fixed in
the compute recommender).

Key design invariants:
- needs_database=False / needs_cache=False tiers are short-circuited
  entirely in code: the LLM is not asked to fill in tiers that aren't
  needed, and the resulting sub-recommendation is constructed directly
  with needed=False.
- LoadBalancerRecommendation.needed is always set directly from
  technical_needs.load_balancer_needed — already reasoned about and
  tested upstream — so the LLM cannot re-derive and contradict it.
- Cache validation checks the (instance_type, engine) PAIR, not just
  the instance_type, because the same node-type name can be valid for
  one engine and invalid for another.
- The anti-hallucination validation+retry pattern is shared with
  recommender.py via _recommendation_utils.py.
"""

from __future__ import annotations

import json
import re

from app.agent.nodes._recommendation_utils import (
    allowed_cache_pairs,
    cache_correction_prompt,
    correction_prompt as _instance_correction_prompt,
    invalid_cache_pairs,
    invalid_instance_types,
)
from app.agent.state import AgentState
from app.llm.structured import StructuredOutputError, invoke_structured
from app.models.schemas import (
    CacheCandidate,
    CacheRecommendation,
    DatabaseCandidate,
    DatabaseRecommendation,
    EstimatedCost,
    InstanceCandidate,
    InstanceRecommendation,
    LoadBalancerRecommendation,
    SystemDesignRecommendation,
    TechnicalNeeds,
)


class HolisticRecommendationError(RuntimeError):
    """Raised when the LLM fails to produce a valid holistic recommendation."""


# ── Prompt ────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are recommending a complete AWS architecture for a workload.
Reason about ALL tiers together — compute, database, and cache — as a
coherent system rather than independently, so the pieces are sized and
selected to work with each other.

Original user requirements:
{requirements}

Technical needs (derived from user requirements):
{technical_needs}

Available EC2 compute candidates (live data — authoritative):
{compute_candidates}

{database_section}

{cache_section}

Rules:
- Pick recommended_instance and alternative_instance for the compute
  tier ONLY from the EC2 candidates list above. Never invent a type.
- {database_rule}
- {cache_rule}
- Load balancer: needed={lb_needed}. {lb_instruction}
- Size compute for its share of load given scaling_recommendation and
  min_instances/max_instances from technical_needs.
- Prefer the smallest candidate that comfortably meets the need.
- Do NOT invent instance types, dollar amounts, or latency guarantees.
- Explain how the tiers work together in architecture_summary.

Output contract (very important):
- Return EXACTLY one JSON object matching the SystemDesignRecommendation
  schema directly. Field names must be exact.
- Top-level keys: compute, database, cache, load_balancer,
  architecture_summary.
- compute must match InstanceRecommendation: recommended_instance, why,
  assumptions (array), confidence, alternative_instance, trade_off.
- database must match DatabaseRecommendation: needed, recommended_instance,
  engine_suggestion, why, assumptions (array), confidence,
  alternative_instance.
- cache must match CacheRecommendation: needed, recommended_instance,
  engine (one of: "Memcached", "Redis", "Valkey" or null), why,
  assumptions (array), confidence, alternative_instance,
  alternative_engine.
- load_balancer must match LoadBalancerRecommendation: needed,
  load_balancer_type, why.
- assumptions in every sub-object MUST be a JSON array of strings.
- confidence in every sub-object must be "low", "medium", or "high".
- Do NOT wrap output inside any outer key. No wrapper object.
"""

_DB_CANDIDATES_SECTION = """\
Available RDS database candidates (live data — authoritative):
{db_candidates}"""

_DB_NO_CANDIDATES_SECTION = """\
Database tier: NOT NEEDED for this workload (needs_database=False).
Set database.needed=false in your output."""

_CACHE_CANDIDATES_SECTION = """\
Available ElastiCache cache candidates grouped by engine (live data — authoritative):
{cache_candidates}"""

_CACHE_NO_CANDIDATES_SECTION = """\
Cache tier: NOT NEEDED for this workload (needs_cache=False).
Set cache.needed=false in your output."""

_DB_RULE_PICK = (
    "Pick database.recommended_instance and alternative_instance ONLY "
    "from the RDS candidates list above."
)
_DB_RULE_SKIP = (
    "Set database.needed=false. Do not recommend a database instance."
)

_CACHE_RULE_PICK = (
    "Pick cache.recommended_instance + engine as an (instance_type, engine) "
    "pair that actually appears in the cache candidates above. The same "
    "node-type name may not be valid for all engines."
)
_CACHE_RULE_SKIP = (
    "Set cache.needed=false. Do not recommend a cache node."
)

_LB_INSTRUCTION_NEEDED = (
    "Fill in load_balancer_type (e.g. 'Application Load Balancer') and why."
)
_LB_INSTRUCTION_NOT_NEEDED = (
    "Set load_balancer.needed=false, load_balancer_type=null, "
    "and explain briefly why no load balancer is needed."
)


# ── Candidate formatters ──────────────────────────────────────────────────

def _format_compute(candidates: list[InstanceCandidate]) -> str:
    lines = []
    for c in candidates:
        line = f"- {c.instance_type}: {c.vcpu} vCPU, {c.memory_gib} GiB RAM"
        if c.gpu_count:
            line += f", {c.gpu_count}x {c.gpu_model} ({c.gpu_memory_gib} GiB GPU)"
        line += f", network: {c.network_performance}"
        lines.append(line)
    return "\n".join(lines)


def _format_database(candidates: list[DatabaseCandidate]) -> str:
    return "\n".join(
        f"- {c.instance_type}: {c.vcpu} vCPU, {c.memory_gib} GiB RAM, "
        f"family: {c.family}, network: {c.network_performance}"
        for c in candidates
    )


def _format_cache(candidates: list[CacheCandidate]) -> str:
    """Group by engine so the LLM sees engine affinity clearly."""
    from collections import defaultdict
    by_engine: dict[str, list[str]] = defaultdict(list)
    for c in candidates:
        entry = (
            f"  - {c.instance_type}: {c.vcpu} vCPU, {c.memory_gib} GiB RAM, "
            f"network: {c.network_performance}"
        )
        if c.max_clients is not None:
            entry += f", max_clients: {c.max_clients}"
        by_engine[c.engine.value].append(entry)
    lines = []
    for engine in sorted(by_engine):
        lines.append(f"{engine}:")
        lines.extend(by_engine[engine])
    return "\n".join(lines)


# ── Short-circuit constructors for skipped tiers ──────────────────────────

def _tier_reasoning(reasoning: str, tier: str) -> str:
    """Use model reasoning only when it explicitly discusses the skipped tier."""
    keywords = {
        "database": ("database", "db", "rds", "storage", "relational", "persistence"),
        "cache": ("cache", "caching", "redis", "memcached", "valkey"),
    }[tier]
    lowered = reasoning.lower()
    if not any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in keywords):
        return "Not required for this workload."
    excerpt = reasoning.strip()
    return excerpt or "Not required for this workload."


def _skipped_database(reasoning: str) -> DatabaseRecommendation:
    return DatabaseRecommendation(
        needed=False,
        why=_tier_reasoning(reasoning, "database"),
        confidence="high",
    )


def _skipped_cache(reasoning: str) -> CacheRecommendation:
    return CacheRecommendation(
        needed=False,
        why=_tier_reasoning(reasoning, "cache"),
        confidence="high",
    )


# ── Cost estimation ────────────────────────────────────────────────────────────

_HOURS_PER_MONTH = 730


def _find_price(
    candidates: list, instance_type: str | None
) -> float | None:
    """Look up the hourly on-demand price for an instance type from candidates."""
    if instance_type is None:
        return None
    for c in candidates:
        if c.instance_type == instance_type:
            return getattr(c, "hourly_price_usd", None)
    return None


def _compute_estimated_cost(
    sdr: SystemDesignRecommendation,
    tn: TechnicalNeeds,
    compute_candidates: list[InstanceCandidate],
    db_candidates: list[DatabaseCandidate],
    cache_candidates: list[CacheCandidate],
) -> EstimatedCost:
    """
    Compute an estimated monthly on-demand cost from live pricing data.

    Compute: min/max instances × recommended instance hourly price × 730h.
    Database & cache: single-instance hourly price × 730h.
    Any tier whose price is unavailable (None) surfaces as None — never fabricated.
    """
    compute_price = _find_price(compute_candidates, sdr.compute.recommended_instance)
    db_price = None
    cache_price = None

    if sdr.database.needed and sdr.database.recommended_instance:
        db_price = _find_price(db_candidates, sdr.database.recommended_instance)

    if sdr.cache.needed and sdr.cache.recommended_instance:
        cache_price = _find_price(cache_candidates, sdr.cache.recommended_instance)

    compute_low = None
    compute_high = None
    if compute_price is not None:
        compute_low = round(compute_price * tn.min_instances * _HOURS_PER_MONTH, 2)
        compute_high = round(compute_price * tn.max_instances * _HOURS_PER_MONTH, 2)

    db_monthly = round(db_price * _HOURS_PER_MONTH, 2) if db_price is not None else None
    cache_monthly = round(cache_price * _HOURS_PER_MONTH, 2) if cache_price is not None else None

    total_low = None
    total_high = None
    parts_low = [compute_low, db_monthly, cache_monthly]
    parts_high = [compute_high, db_monthly, cache_monthly]
    # A tier that is not needed contributes 0 to the total; a tier that is
    # needed but whose price is unavailable makes the total uncomputable.
    db_needed = sdr.database.needed and sdr.database.recommended_instance
    cache_needed = sdr.cache.needed and sdr.cache.recommended_instance
    unresolved = (
        (compute_low is None) or
        (db_needed and db_monthly is None) or
        (cache_needed and cache_monthly is None)
    )
    if not unresolved:
        total_low = round(sum(p or 0 for p in parts_low), 2)
        total_high = round(sum(p or 0 for p in parts_high), 2)

    return EstimatedCost(
        compute_monthly_low=compute_low,
        compute_monthly_high=compute_high,
        database_monthly=db_monthly,
        cache_monthly=cache_monthly,
        total_monthly_low=total_low,
        total_monthly_high=total_high,
    )


# ── Validation helpers ────────────────────────────────────────────────────

def _validate_compute(
    result: SystemDesignRecommendation,
    allowed: set[str],
) -> list[str]:
    return invalid_instance_types(result.compute, allowed)


def _validate_database(
    result: SystemDesignRecommendation,
    allowed: set[str],
) -> list[str]:
    bad: list[str] = []
    db = result.database
    if not db.needed:
        return bad
    if db.recommended_instance is not None and db.recommended_instance not in allowed:
        bad.append(db.recommended_instance)
    if db.alternative_instance is not None and db.alternative_instance not in allowed:
        bad.append(db.alternative_instance)
    return bad


def _validate_cache(
    result: SystemDesignRecommendation,
    pairs: set[tuple],
) -> list[str]:
    if not result.cache.needed:
        return []
    return invalid_cache_pairs(result.cache, pairs)


# ── Retry correction prompts ──────────────────────────────────────────────

def _compute_correction(
    base_prompt: str,
    bad: list[str],
    allowed_list: list[str],
) -> str:
    return _instance_correction_prompt(
        base_prompt,
        invalid_instances=bad,
        allowed_types=allowed_list,
    )


def _database_correction(
    base_prompt: str,
    bad: list[str],
    allowed_list: list[str],
) -> str:
    invalid_joined = ", ".join(repr(t) for t in bad)
    allowed_joined = ", ".join(allowed_list)
    return (
        f"{base_prompt}\n\n"
        f"Your previous database recommendation included {invalid_joined}, which "
        f"is not in the live RDS candidate set. You MUST choose "
        f"database.recommended_instance and alternative_instance only from: "
        f"{allowed_joined}."
    )


def _cache_correction(
    base_prompt: str,
    bad: list[str],
    cache_candidates: list[CacheCandidate],
) -> str:
    pairs = [(c.instance_type, c.engine) for c in cache_candidates]
    return cache_correction_prompt(
        base_prompt,
        invalid_descriptions=bad,
        allowed_pairs=pairs,
    )


# ── Main node ─────────────────────────────────────────────────────────────

def _build_prompt(
    state: AgentState,
    needs_db: bool,
    needs_cache: bool,
    compute_text: str,
    db_text: str | None,
    cache_text: str | None,
    lb_needed: bool,
) -> str:
    requirements = state["requirements"]
    needs = state["technical_needs"]

    database_section = (
        _DB_CANDIDATES_SECTION.format(db_candidates=db_text)
        if needs_db and db_text
        else _DB_NO_CANDIDATES_SECTION
    )
    cache_section = (
        _CACHE_CANDIDATES_SECTION.format(cache_candidates=cache_text)
        if needs_cache and cache_text
        else _CACHE_NO_CANDIDATES_SECTION
    )
    db_rule = _DB_RULE_PICK if needs_db else _DB_RULE_SKIP
    cache_rule = _CACHE_RULE_PICK if needs_cache else _CACHE_RULE_SKIP
    lb_instruction = _LB_INSTRUCTION_NEEDED if lb_needed else _LB_INSTRUCTION_NOT_NEEDED

    return _PROMPT_TEMPLATE.format(
        requirements=requirements.model_dump_json(indent=2),
        technical_needs=needs.model_dump_json(indent=2),
        compute_candidates=compute_text,
        database_section=database_section,
        cache_section=cache_section,
        database_rule=db_rule,
        cache_rule=cache_rule,
        lb_needed=lb_needed,
        lb_instruction=lb_instruction,
    )


def recommend_system_design(state: AgentState) -> AgentState:
    needs: TechnicalNeeds | None = state["technical_needs"]
    requirements = state["requirements"]
    compute_candidates = state.get("instance_candidates") or []
    db_candidates = state.get("database_candidates") or []
    cache_candidates = state.get("cache_candidates") or []

    if needs is None or not compute_candidates:
        raise HolisticRecommendationError(
            "technical_needs and instance_candidates must be set before "
            "calling the holistic recommender."
        )

    # Determine which tiers are active.
    needs_db = needs.needs_database and bool(db_candidates)
    needs_cache = needs.needs_cache and bool(cache_candidates)
    lb_needed = needs.load_balancer_needed

    # Pre-build allowed sets for validation.
    compute_allowed = {c.instance_type for c in compute_candidates}
    compute_allowed_list = [c.instance_type for c in compute_candidates]
    db_allowed = {c.instance_type for c in db_candidates} if needs_db else set()
    db_allowed_list = [c.instance_type for c in db_candidates] if needs_db else []
    cache_pairs = allowed_cache_pairs(cache_candidates) if needs_cache else set()

    # Format candidate texts.
    compute_text = _format_compute(compute_candidates)
    db_text = _format_database(db_candidates) if needs_db else None
    cache_text = _format_cache(cache_candidates) if needs_cache else None

    prompt = _build_prompt(
        state, needs_db, needs_cache,
        compute_text, db_text, cache_text, lb_needed,
    )

    # ── LLM call ──────────────────────────────────────────────────────────
    try:
        result: SystemDesignRecommendation = invoke_structured(
            SystemDesignRecommendation, prompt
        )
    except StructuredOutputError as exc:
        raise HolisticRecommendationError(str(exc)) from exc

    # ── Validate all three tiers; retry once if any are invalid ───────────
    bad_compute = _validate_compute(result, compute_allowed)
    bad_db = _validate_database(result, db_allowed) if needs_db else []
    bad_cache = _validate_cache(result, cache_pairs) if needs_cache else []

    if bad_compute or bad_db or bad_cache:
        # Build a single correction prompt addressing all violations at once.
        retry_prompt = prompt
        if bad_compute:
            retry_prompt = _compute_correction(retry_prompt, bad_compute, compute_allowed_list)
        if bad_db:
            retry_prompt = _database_correction(retry_prompt, bad_db, db_allowed_list)
        if bad_cache:
            retry_prompt = _cache_correction(retry_prompt, bad_cache, cache_candidates)

        try:
            result = invoke_structured(SystemDesignRecommendation, retry_prompt)
        except StructuredOutputError as exc:
            raise HolisticRecommendationError(str(exc)) from exc

        # Final check — one retry only.
        bad_compute = _validate_compute(result, compute_allowed)
        bad_db = _validate_database(result, db_allowed) if needs_db else []
        bad_cache = _validate_cache(result, cache_pairs) if needs_cache else []

        if bad_compute:
            raise HolisticRecommendationError(
                f"Model recommended compute {bad_compute[0]!r}, "
                "which is not in the live EC2 candidate set."
            )
        if bad_db:
            raise HolisticRecommendationError(
                f"Model recommended database {bad_db[0]!r}, "
                "which is not in the live RDS candidate set."
            )
        if bad_cache:
            raise HolisticRecommendationError(
                f"Model recommended cache {bad_cache[0]}, "
                "which is not a valid (instance_type, engine) pair."
            )

    # ── Enforce load-balancer invariant in code (not LLM-derived) ─────────
    # The LLM fills in load_balancer_type and why, but needed must always
    # match what the system design reasoner already decided.
    lb = result.load_balancer
    if lb.needed != lb_needed:
        result = result.model_copy(
            update={"load_balancer": lb.model_copy(update={"needed": lb_needed})}
        )

    # ── Short-circuit skipped tiers ────────────────────────────────────────
    # If the workload doesn't need a tier, replace whatever the LLM said
    # with the canonical needed=False value constructed in code.
    if not needs.needs_database:
        result = result.model_copy(update={"database": _skipped_database(needs.reasoning)})
    if not needs.needs_cache:
        result = result.model_copy(update={"cache": _skipped_cache(needs.reasoning)})

    # ── Compute estimated monthly cost from live pricing ──────────────────────
    # None when pricing data is unavailable for any needed tier.
    result = result.model_copy(
        update={
            "estimated_cost": _compute_estimated_cost(
                result, needs, compute_candidates, db_candidates, cache_candidates
            )
        }
    )

    state["system_design_recommendation"] = result
    return state


holistic_recommend = recommend_system_design
