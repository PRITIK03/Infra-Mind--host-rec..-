"""
System design reasoner node.

Converts validated UserRequirements into TechnicalNeeds — translating
raw workload information (registered users, RPS, traffic pattern, etc.)
into a concurrency estimate, resource profile, and scaling
recommendation, with reasoning attached.

Optionally runs one Tavily search round when configured and the model
decides research would help. Search failures never abort reasoning.

Only runs once requirement_validator has confirmed no critical fields
are missing.
"""

from __future__ import annotations

import json
from typing import Any

from app.agent.state import AgentState
from app.llm.client import StructuredOutputError, get_chat_model, invoke_structured
from app.models.schemas import ResourceProfile, TechnicalNeeds, TrafficPattern
from app.tools.web_search import try_get_search_tool


class ReasoningError(RuntimeError):
    """Raised when the LLM fails to produce technical needs."""


_PROMPT_TEMPLATE = """\
You are reasoning about system design for an AWS deployment. Given the
following user-stated requirements, determine the technical needs of
the system.

Distinguish registered/total users from actual concurrent load — do not
assume registered users equals concurrent users. If concurrency or RPS
isn't explicitly stated, estimate a reasonable concurrency figure from
the traffic pattern and user count, and explain that estimate in your
reasoning.

resource_profile may be "gpu_bound" when the workload is primarily
constrained by GPU capacity (typically when requires_gpu is true).
Reserve CPU-bound and memory-bound labels for workloads where GPU is
not the primary constraint.

When determining scaling_recommendation, base it on the traffic
pattern of CURRENT expected load, not on possible future growth.
Steady, predictable traffic generally favors vertical scaling or a
fixed-size deployment sized for that steady load — horizontal scaling
with auto-scaling is earned by traffic that actually varies or spikes
(bursty patterns), since its main benefit is elastically absorbing
peaks without over-provisioning during quiet periods. Do not
recommend horizontal scaling for steady traffic solely because the
user base might grow later — that is a separate, future capacity
planning concern, not today's scaling strategy.

scaling_recommendation must be consistent with estimated_concurrency.
Horizontal scaling only makes sense when there is enough concurrent
load to meaningfully distribute across multiple instances. If
estimated_concurrency is very low (roughly single digits, e.g. 1-3
concurrent users or job/worker slots), prefer vertical scaling or a
single right-sized instance even if traffic_pattern is bursty — note
in your reasoning that horizontal scaling isn't justified at this
concurrency level. Reserve horizontal/auto-scaling recommendations
for cases where concurrency is high enough that spreading load across
multiple instances is actually meaningful.

Also, if workload_type is batch_processing, frame estimated_concurrency
as concurrent job/worker slots rather than concurrent human users,
and say so explicitly in your reasoning, since the two are conceptually
different.

If latency_requirement_ms or budget_constraint_usd_monthly are present,
reflect their implications in scaling_recommendation and reasoning
(e.g. tighter latency may favor fewer hops / warmer capacity; tighter
budget may favor right-sizing and horizontal elasticity over oversized
single nodes).

Ground estimates in the stated requirements. Prefer conservative,
explicit assumptions over invented peak loads. If the user already
stated traffic_pattern or resource_profile (not unknown), preserve
those classifications unless an explicit GPU requirement overrides the
resource profile. If optional research findings are provided below,
use only what is directly relevant; do not let generic marketing
content override the user's concrete numbers.

Also determine: needs_database (does this workload need a persistent
relational database — most web apps and API services do, default to
true unless the workload is clearly stateless or explicitly uses
another storage approach), needs_cache (would a caching layer
meaningfully help — this should NOT default to true; only recommend
it when there is a clear read-heavy or repeated-access pattern, do
not add it reflexively), and min_instances/max_instances (concrete
numbers for the compute tier, consistent with your
scaling_recommendation and estimated_concurrency — if
scaling_recommendation is vertical/single-instance, min_instances
should equal max_instances; if horizontal, max_instances should
exceed min_instances to reflect real elasticity). Set
load_balancer_needed to true whenever max_instances > 1.

Output contract (very important):
- Return EXACTLY one JSON object.
- The top-level JSON object MUST match the TechnicalNeeds schema directly.
- Do NOT wrap TechnicalNeeds fields inside any outer key like
  "technical_needs", "TechnicalNeeds", or similar.
- Do NOT add extra top-level keys.

Requirements:
{requirements}
"""


_SEARCH_DECISION_PROMPT_TEMPLATE = """\
You are planning a system-design reasoning step for an AWS workload.

Decide if a quick web search would materially improve confidence for
resource_profile or scaling_recommendation for this specific workload.

- If your own knowledge is sufficient for a sound estimate from the
  stated requirements, do not call any tool and answer briefly that no
  search is needed.
- If web research would help, call the search tool exactly once with one
  specific query focused on AWS architecture / EC2 sizing / scaling
  guidance for a similar workload (workload type, traffic shape, GPU
  needs). Prefer concrete technical guidance over product marketing.

Requirements:
{requirements}
"""


def _extract_search_query(tool_call: dict[str, Any]) -> str | None:
    args = tool_call.get("args")
    if isinstance(args, dict):
        query = args.get("query")
        return str(query).strip() if query else None
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict) and parsed.get("query"):
                return str(parsed["query"]).strip()
        except ValueError:
            text = args.strip()
            return text or None
    return None


def _format_research_findings(query: str, results: Any) -> str:
    if isinstance(results, dict):
        records = results.get("results")
        if isinstance(records, list) and records:
            lines = [f"Query: {query}"]
            for index, item in enumerate(records, start=1):
                if not isinstance(item, dict):
                    lines.append(f"{index}. {item}")
                    continue
                title = item.get("title") or "Untitled"
                url = item.get("url") or ""
                content = (item.get("content") or "").strip()
                snippet = content[:280] + ("..." if len(content) > 280 else "")
                lines.append(f"{index}. {title} ({url})")
                if snippet:
                    lines.append(f"   - {snippet}")
            return "\n".join(lines)

    return "\n".join(
        [
            f"Query: {query}",
            "Raw results:",
            json.dumps(results, indent=2, default=str),
        ]
    )


def _maybe_collect_research(requirements) -> str | None:
    """
    Bounded optional research: at most one search round.
    Returns None when search is disabled, declined by the model, or fails.
    """
    search_tool = try_get_search_tool(max_results=3)
    if search_tool is None:
        return None

    try:
        decision_model = get_chat_model().bind_tools([search_tool])
        decision = decision_model.invoke(
            _SEARCH_DECISION_PROMPT_TEMPLATE.format(
                requirements=requirements.model_dump_json(indent=2)
            )
        )
    except Exception:
        return None

    tool_calls = getattr(decision, "tool_calls", None) or []
    if not tool_calls:
        return None

    query = _extract_search_query(tool_calls[0])
    if not query:
        return None

    try:
        results = search_tool.invoke({"query": query})
    except Exception:
        return None

    return _format_research_findings(query, results)


def _apply_deterministic_requirement_bridges(
    requirements,
    needs: TechnicalNeeds,
) -> TechnicalNeeds:
    """
    Honor explicit user-stated constraints that must not depend on the LLM
    remembering them. GPU requirement wins over a non-GPU resource profile.
    """
    updates: dict[str, Any] = {}

    if requirements.traffic_pattern != TrafficPattern.UNKNOWN:
        if needs.traffic_pattern != requirements.traffic_pattern:
            updates["traffic_pattern"] = requirements.traffic_pattern

    if requirements.gpu_required is True:
        if not needs.requires_gpu:
            updates["requires_gpu"] = True
        if needs.resource_profile != ResourceProfile.GPU_BOUND:
            updates["resource_profile"] = ResourceProfile.GPU_BOUND
    else:
        if requirements.gpu_required is False and needs.requires_gpu:
            updates["requires_gpu"] = False

        user_profile = requirements.resource_profile
        if user_profile not in (ResourceProfile.UNKNOWN, ResourceProfile.GPU_BOUND):
            if needs.resource_profile != user_profile:
                updates["resource_profile"] = user_profile
        elif (
            requirements.gpu_required is False
            and needs.resource_profile == ResourceProfile.GPU_BOUND
        ):
            updates["resource_profile"] = ResourceProfile.BALANCED

    if not updates:
        return needs
    return needs.model_copy(update=updates)


def reason_system_design(state: AgentState) -> AgentState:
    requirements = state["requirements"]

    research_findings = _maybe_collect_research(requirements)

    prompt = _PROMPT_TEMPLATE.format(requirements=requirements.model_dump_json(indent=2))
    if research_findings:
        prompt = f"{prompt}\n\nRelevant research findings:\n{research_findings}"

    try:
        result = invoke_structured(TechnicalNeeds, prompt)
    except StructuredOutputError as exc:
        raise ReasoningError(str(exc)) from exc

    state["technical_needs"] = _apply_deterministic_requirement_bridges(
        requirements, result
    )
    return state
