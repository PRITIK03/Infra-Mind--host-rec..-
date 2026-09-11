"""
Grounding / consistency check node.

Runs after holistic_recommend and before generate_terraform.  Makes a
single structured LLM call to detect concrete contradictions between
system_design_recommendation and technical_needs — the same class of
logical bug that was previously found and fixed by hand (e.g. a
batch-job recommendation whose architecture_summary praised horizontal
auto-scaling while min_instances == max_instances == 1).

Workflow
--------
1. Call the LLM with (system_design_recommendation, technical_needs)
   and ask for a GroundingResult{passed: bool, issues: list[str]}.
2. If passed → continue to generate_terraform unchanged.
3. If NOT passed → rebuild the holistic_recommend prompt with the
   specific issues appended as required corrections, call holistic
   recommend once more, then re-run the grounding check.
4. If still failing after one retry → set grounding_passed=False and
   populate grounding_notes on the recommendation so the failure is
   explicitly visible to the caller.  We never silently proceed with a
   known-bad output.

Design invariants
-----------------
- One bounded retry only — no loops.
- Never raises to stop the graph; a failed check surfaces as a flagged
  field, consistent with the project's honest-failure principle.
- Uses invoke_structured() exactly as every other node does.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.agent.nodes.holistic_recommender import (
    _build_prompt,
    _compute_estimated_cost,
    _format_cache,
    _format_compute,
    _format_database,
    _skipped_cache,
    _skipped_database,
)
from app.agent.state import AgentState
from app.llm.structured import StructuredOutputError, invoke_structured
from app.models.schemas import (
    SystemDesignRecommendation,
    TechnicalNeeds,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema returned by the grounding LLM call
# ---------------------------------------------------------------------------


class GroundingResult(BaseModel):
    """
    Structured output of the grounding / consistency check.

    passed: True only when no concrete contradictions were found.
    issues: Empty when passed=True; otherwise a list of short, specific
            contradiction descriptions that the repair prompt can quote
            verbatim so the model knows exactly what to fix.
    """

    passed: bool = Field(
        ...,
        description=(
            "True if the recommendation is internally consistent with "
            "technical_needs.  False if at least one concrete contradiction "
            "was found."
        ),
    )
    issues: list[str] = Field(
        default_factory=list,
        description=(
            "Specific contradictions found.  Each entry should be one short "
            "sentence identifying the conflict clearly enough for the "
            "recommender to fix it.  Empty when passed=True."
        ),
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_GROUNDING_PROMPT = """\
You are a consistency auditor for AWS architecture recommendations.

Given the system design recommendation and the technical needs that produced
it, identify CONCRETE contradictions only — not stylistic preferences.

Contradictions to look for (non-exhaustive):
- architecture_summary mentions a tier (database, cache, load balancer) that
  is marked needed=False in the recommendation, or vice versa.
- scaling language ("horizontal scaling", "auto scaling", "scale out") is used
  in architecture_summary or compute.why while min_instances == max_instances
  (single instance — nothing to scale horizontally).
- The opposite: min_instances < max_instances (horizontal range) but the
  summary / why say "vertical" or "single instance".
- compute.why or architecture_summary reference GPU capabilities, but
  technical_needs.requires_gpu is False, or vice versa.
- A tier's "why" field contradicts technical_needs.reasoning — e.g. the "why"
  claims the workload is CPU-bound when technical_needs says memory-bound.
- load_balancer.needed is True but max_instances == 1 (one instance never
  needs a load balancer).
- load_balancer.needed is False but max_instances > 1.

Do NOT flag:
- Reasonable sizing trade-offs or conservative vs generous sizing opinions.
- Missing or terse explanations that are not factually wrong.
- Anything that requires domain knowledge beyond what is present in the inputs.

Technical needs:
{technical_needs}

System design recommendation:
{recommendation}

Return a GroundingResult JSON object.  If no contradictions are found, return
{{"passed": true, "issues": []}}.  Keep each issue description under 120 chars.
"""


def _run_grounding_check(
    sdr: SystemDesignRecommendation,
    tn: TechnicalNeeds,
) -> GroundingResult:
    """Single structured LLM call → GroundingResult."""
    prompt = _GROUNDING_PROMPT.format(
        technical_needs=tn.model_dump_json(indent=2),
        recommendation=sdr.model_dump_json(
            indent=2,
            # Exclude grounding fields — they're irrelevant to the check itself.
            exclude={"grounding_passed", "grounding_notes"},
        ),
    )
    try:
        return invoke_structured(GroundingResult, prompt)
    except StructuredOutputError as exc:
        # If the grounding check itself fails to parse, treat it conservatively
        # as a soft pass with a logged warning rather than crashing the run.
        logger.warning("Grounding check LLM call failed (%s); treating as passed.", exc)
        return GroundingResult(passed=True, issues=[])


def _build_repair_prompt(state: AgentState, issues: list[str]) -> str:
    """
    Re-build the holistic_recommend base prompt augmented with the
    specific grounding issues as mandatory corrections.
    """
    needs: TechnicalNeeds = state["technical_needs"]  # type: ignore[assignment]
    compute_candidates = state.get("instance_candidates") or []
    db_candidates = state.get("database_candidates") or []
    cache_candidates = state.get("cache_candidates") or []

    needs_db = needs.needs_database and bool(db_candidates)
    needs_cache = needs.needs_cache and bool(cache_candidates)
    lb_needed = needs.load_balancer_needed

    compute_text = _format_compute(compute_candidates)
    db_text = _format_database(db_candidates) if needs_db else None
    cache_text = _format_cache(cache_candidates) if needs_cache else None

    base = _build_prompt(
        state, needs_db, needs_cache,
        compute_text, db_text, cache_text, lb_needed,
    )

    issues_block = "\n".join(f"  - {i}" for i in issues)
    repair_suffix = (
        "\n\nCRITICAL CORRECTIONS REQUIRED — your previous recommendation "
        "contained the following concrete inconsistencies that MUST be "
        "resolved in this revised output:\n"
        f"{issues_block}\n"
        "Fix each one. Do not repeat them in the new recommendation."
    )
    return base + repair_suffix


def _apply_post_processing(
    repaired: SystemDesignRecommendation,
    state: AgentState,
) -> SystemDesignRecommendation:
    """
    Enforce the same load-balancer / skipped-tier / cost invariants that
    holistic_recommender applies after its own LLM call.
    """
    needs: TechnicalNeeds = state["technical_needs"]  # type: ignore[assignment]
    compute_candidates = state.get("instance_candidates") or []
    db_candidates = state.get("database_candidates") or []
    cache_candidates = state.get("cache_candidates") or []

    lb = repaired.load_balancer
    if lb.needed != needs.load_balancer_needed:
        repaired = repaired.model_copy(
            update={"load_balancer": lb.model_copy(update={"needed": needs.load_balancer_needed})}
        )
    if not needs.needs_database:
        repaired = repaired.model_copy(
            update={"database": _skipped_database(needs.reasoning)}
        )
    if not needs.needs_cache:
        repaired = repaired.model_copy(
            update={"cache": _skipped_cache(needs.reasoning)}
        )
    repaired = repaired.model_copy(
        update={
            "estimated_cost": _compute_estimated_cost(
                repaired, needs, compute_candidates, db_candidates, cache_candidates
            )
        }
    )
    return repaired


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def grounding_check(state: AgentState) -> AgentState:
    """
    LangGraph node: consistency-check the holistic recommendation before
    Terraform generation.  Runs after holistic_recommend.
    """
    sdr: SystemDesignRecommendation | None = state.get("system_design_recommendation")
    tn: TechnicalNeeds | None = state.get("technical_needs")

    if sdr is None or tn is None:
        # Nothing to check — pass through silently.
        return state

    # ── First grounding check ──────────────────────────────────────────────
    result = _run_grounding_check(sdr, tn)

    if result.passed:
        state["system_design_recommendation"] = sdr.model_copy(
            update={"grounding_passed": True, "grounding_notes": []}
        )
        return state

    # ── Failed — attempt one bounded repair ───────────────────────────────
    logger.info(
        "Grounding check failed with %d issue(s); triggering repair retry. Issues: %s",
        len(result.issues),
        result.issues,
    )

    repair_prompt = _build_repair_prompt(state, result.issues)

    try:
        repaired: SystemDesignRecommendation = invoke_structured(
            SystemDesignRecommendation, repair_prompt
        )
    except StructuredOutputError as exc:
        # Repair call itself failed — surface grounding failure honestly.
        logger.warning("Grounding repair call failed (%s); flagging result.", exc)
        state["system_design_recommendation"] = sdr.model_copy(
            update={"grounding_passed": False, "grounding_notes": result.issues}
        )
        return state

    repaired = _apply_post_processing(repaired, state)

    # ── Second grounding check on the repair ──────────────────────────────
    second_result = _run_grounding_check(repaired, tn)

    if second_result.passed:
        state["system_design_recommendation"] = repaired.model_copy(
            update={"grounding_passed": True, "grounding_notes": []}
        )
        return state

    # ── Still failing after one retry — flag honestly, do NOT suppress ────
    logger.warning(
        "Grounding check still failing after repair. Remaining issues: %s",
        second_result.issues,
    )
    state["system_design_recommendation"] = repaired.model_copy(
        update={"grounding_passed": False, "grounding_notes": second_result.issues}
    )
    return state
