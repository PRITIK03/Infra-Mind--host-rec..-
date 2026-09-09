"""
Recommender node.

Given technical needs, original user requirements, and a filtered set of
live EC2 candidates, asks the LLM to pick the best fit and produce a
structured InstanceRecommendation with reasoning, assumptions,
confidence, and an alternative.
"""

from __future__ import annotations

from app.agent.nodes._recommendation_utils import (
    correction_prompt as _correction_prompt,
    invalid_instance_types as _invalid_instance_types,
)
from app.agent.state import AgentState
from app.llm.client import StructuredOutputError, invoke_structured
from app.models.schemas import InstanceRecommendation


class RecommendationError(RuntimeError):
    """Raised when the LLM fails to produce a recommendation."""


_PROMPT_TEMPLATE = """\
You are recommending an AWS EC2 instance type for a workload.

Original user requirements (may include budget/latency constraints):
{requirements}

Technical needs (derived from the user's stated requirements):
{technical_needs}

Available EC2 instance candidates (live data — vCPU, memory, and GPU
fields here are authoritative; do not invent specs or instance types
not listed):
{candidates}

Rules:
- Pick recommended_instance and alternative_instance ONLY from the
  candidates list above. Never invent an instance type.
- Pay attention to scaling_recommendation above. If it recommends
  horizontal scaling (multiple instances behind a load balancer / auto
  scaling group), size your pick for a reasonable share of the peak
  load per instance — not the full estimated_concurrency alone — and
  say so explicitly in your reasoning. If a single instance is more
  appropriate for this workload, size for the full estimated_concurrency
  instead and explain why horizontal scaling wasn't preferred.
- When horizontal scaling applies, state explicitly (as one of your
  assumptions) what per-instance capacity you're assuming and how you
  arrived at it. All else being roughly equal, prefer more, smaller
  instances over fewer, larger ones for a horizontally-scaled,
  auto-scaled workload — this gives better elasticity and fault
  tolerance, and more closely matches actual bursty demand. Only prefer
  fewer, larger instances if you have a concrete reason (e.g. high
  per-request memory footprint, work that doesn't parallelize well
  across instances, or a stated cost/simplicity preference) — and if
  so, say that reason explicitly.
- If budget_constraint_usd_monthly is set, prefer cost-efficient right-
  sizing and call out budget tension in trade_off / assumptions when
  relevant. Candidate hourly prices are NOT available in the live data
  provided here (hourly_price_usd is unset). Do NOT invent dollar
  amounts, hourly rates, or monthly cost figures. State clearly that
  cost comparison is qualitative only.
- If latency_requirement_ms is set, note it as a design constraint and
  prefer candidates whose size/network profile can plausibly support
  responsive workloads. Do NOT claim an exact latency guarantee or
  measured p95/p99 from EC2 specs alone — instance specs do not prove
  application latency.
- Prefer the smallest candidate that comfortably meets the need; only
  recommend a larger one if you have a specific reason, and state that
  reason explicitly rather than defaulting to a bigger instance for
  safety margin alone.
- For GPU workloads, prefer candidates whose GPU count/model/memory fit
  the stated need; do not recommend a non-GPU type when requires_gpu is
  true.

Pick the single best-fit instance_type from the candidates above.
Explain why it fits. State any assumptions you're making, since the
concurrency/resource profile above is itself an estimate. Give a
confidence level (low, medium, or high) based on how much was assumed
versus explicitly stated. Suggest one alternative instance_type from
the candidates and the trade-off versus your main recommendation.

Output contract (very important):
- Return EXACTLY one JSON object.
- The top-level JSON object MUST match the InstanceRecommendation schema
  directly and use its exact field names.
- Required top-level fields: recommended_instance, why, assumptions,
  confidence.
- assumptions MUST be a JSON array/list of strings, even if there is
  only one assumption.
- Use confidence, NOT confidence_level.
- Do not add reasoning unless it actually exists in the schema.
- Do not wrap the recommendation inside any outer key like
  "recommendation" or "InstanceRecommendation".
- No wrapper object.
- Do not add extra top-level keys.
- Do not return anything except the single JSON object.
"""


def recommend_instance(state: AgentState) -> AgentState:
    needs = state["technical_needs"]
    requirements = state["requirements"]
    candidates = state.get("instance_candidates")

    if needs is None or not candidates:
        raise RecommendationError("technical_needs and instance_candidates must be set first.")

    candidates_text = "\n".join(
        f"- {c.instance_type}: {c.vcpu} vCPU, {c.memory_gib} GiB RAM"
        + (
            f", {c.gpu_count}x {c.gpu_model} ({c.gpu_memory_gib} GiB GPU memory)"
            if c.gpu_count
            else ""
        )
        + f", network: {c.network_performance}"
        for c in candidates
    )

    prompt = _PROMPT_TEMPLATE.format(
        requirements=requirements.model_dump_json(indent=2),
        technical_needs=needs.model_dump_json(indent=2),
        candidates=candidates_text,
    )
    try:
        result = invoke_structured(InstanceRecommendation, prompt)
    except StructuredOutputError as exc:
        raise RecommendationError(str(exc)) from exc

    allowed = {c.instance_type for c in candidates}
    allowed_list = [c.instance_type for c in candidates]
    invalid = _invalid_instance_types(result, allowed)

    if invalid:
        # One bounded retry — do not loop indefinitely.
        retry_prompt = _correction_prompt(
            prompt,
            invalid_instances=invalid,
            allowed_types=allowed_list,
        )
        try:
            result = invoke_structured(InstanceRecommendation, retry_prompt)
        except StructuredOutputError as exc:
            raise RecommendationError(str(exc)) from exc

        invalid = _invalid_instance_types(result, allowed)
        if invalid:
            raise RecommendationError(
                f"Model recommended {invalid[0]!r}, which is not in the live candidate set."
            )

    state["recommendation"] = result
    return state
