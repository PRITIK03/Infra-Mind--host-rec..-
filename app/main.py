"""
CLI entrypoint for the AWS Instance Advisor agent.

Collects requirements turn by turn (asking follow-up questions where
needed), then reasons about system design, researches live EC2
candidates, and prints a final recommendation.
"""

from __future__ import annotations

import sys

from app.agent.graph import build_graph
from app.models.schemas import UserRequirements

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> None:
    graph = build_graph()

    state = {
        "requirements": UserRequirements(),
        "latest_user_message": None,
        "next_question": None,
        "pending_field": None,
        "technical_needs": None,
        "instance_candidates": None,
        "recommendation": None,
    }

    print("Describe your application and expected workload:")
    state["latest_user_message"] = input("> ")

    while True:
        try:
            state = graph.invoke(state)
        except RuntimeError as exc:
            print(f"\nError: {exc}")
            break

        if state.get("recommendation"):
            rec = state["recommendation"]
            tn = state["technical_needs"]
            if tn is not None:
                print("\n--- System Design Reasoning ---")
                print(f"Estimated concurrency: {tn.estimated_concurrency}")
                print(f"Resource profile: {tn.resource_profile.value}")
                print(f"Traffic pattern: {tn.traffic_pattern.value}")
                print(f"Requires GPU: {tn.requires_gpu}")
                print(f"Scaling recommendation: {tn.scaling_recommendation}")
                print(f"Reasoning: {tn.reasoning}")
            print("\n--- Recommendation ---")
            print(f"Instance: {rec.recommended_instance}")
            print(f"Why: {rec.why}")
            print(f"Assumptions: {rec.assumptions}")
            print(f"Confidence: {rec.confidence}")
            print(f"Alternative: {rec.alternative_instance}")
            print(f"Trade-off: {rec.trade_off}")
            break

        if state.get("next_question"):
            print(f"\n{state['next_question']}")
            state["latest_user_message"] = input("> ")
            continue

        print("\nNo recommendation or follow-up question was produced. Stopping.")
        break


if __name__ == "__main__":
    main()
