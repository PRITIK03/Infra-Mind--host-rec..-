"""
CLI entrypoint for the AWS Instance Advisor agent.

Collects requirements turn by turn (asking follow-up questions where
needed), then reasons about system design, researches live EC2
candidates, prints a final recommendation, and writes deployable
Terraform files to ./terraform_output/.
"""

from __future__ import annotations

import os
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
        "database_candidates": None,
        "cache_candidates": None,
        "recommendation": None,
        "system_design_recommendation": None,
        "terraform_files": None,
    }

    print("Describe your application and expected workload:")
    try:
        state["latest_user_message"] = input("> ")
    except EOFError:
        return

    while True:
        try:
            state = graph.invoke(state)
        except RuntimeError as exc:
            print(f"\nError: {exc}")
            break

        if state.get("system_design_recommendation"):
            rec = state["system_design_recommendation"]
            tn = state["technical_needs"]
            if tn is not None:
                print("\n--- System Design Reasoning ---")
                print(f"Estimated concurrency: {tn.estimated_concurrency}")
                print(f"Resource profile: {tn.resource_profile.value}")
                print(f"Traffic pattern: {tn.traffic_pattern.value}")
                print(f"Requires GPU: {tn.requires_gpu}")
                print(f"Scaling recommendation: {tn.scaling_recommendation}")
                print(f"Needs database: {tn.needs_database}")
                print(f"Needs cache: {tn.needs_cache}")
                print(f"Load balancer needed: {tn.load_balancer_needed}")
                print(f"Reasoning: {tn.reasoning}")

            print("\n--- Architecture Summary ---")
            print(rec.architecture_summary)

            print("\n--- Component Recommendations ---")

            # Compute Tier
            c = rec.compute
            print("\n[Compute]")
            print(f"  Recommended instance: {c.recommended_instance}")
            print(f"  Why: {c.why}")
            if c.assumptions:
                assumptions_str = ", ".join(c.assumptions) if isinstance(c.assumptions, list) else str(c.assumptions)
                print(f"  Assumptions: {assumptions_str}")
            print(f"  Confidence: {c.confidence}")
            if c.alternative_instance:
                print(f"  Alternative: {c.alternative_instance}")
            if c.trade_off:
                print(f"  Trade-off: {c.trade_off}")

            # Database Tier
            db = rec.database
            print("\n[Database]")
            if db.needed:
                print("  Needed: Yes")
                print(f"  Recommended instance: {db.recommended_instance}")
                if db.engine_suggestion:
                    print(f"  Engine: {db.engine_suggestion}")
                print(f"  Why: {db.why}")
                if db.assumptions:
                    assumptions_str = ", ".join(db.assumptions) if isinstance(db.assumptions, list) else str(db.assumptions)
                    print(f"  Assumptions: {assumptions_str}")
                print(f"  Confidence: {db.confidence}")
                if db.alternative_instance:
                    print(f"  Alternative: {db.alternative_instance}")
            else:
                print("  Needed: No")
                print(f"  Why: {db.why}")

            # Cache Tier
            cache = rec.cache
            print("\n[Cache]")
            if cache.needed:
                print("  Needed: Yes")
                print(f"  Recommended instance: {cache.recommended_instance}")
                engine_str = cache.engine.value if hasattr(cache.engine, "value") else str(cache.engine) if cache.engine else "None"
                print(f"  Engine: {engine_str}")
                print(f"  Why: {cache.why}")
                if cache.assumptions:
                    assumptions_str = ", ".join(cache.assumptions) if isinstance(cache.assumptions, list) else str(cache.assumptions)
                    print(f"  Assumptions: {assumptions_str}")
                print(f"  Confidence: {cache.confidence}")
                if cache.alternative_instance:
                    alt_eng = f" ({cache.alternative_engine.value if hasattr(cache.alternative_engine, 'value') else cache.alternative_engine})" if cache.alternative_engine else ""
                    print(f"  Alternative: {cache.alternative_instance}{alt_eng}")
            else:
                print("  Needed: No")
                print(f"  Why: {cache.why}")

            # Load Balancer
            lb = rec.load_balancer
            print("\n[Load Balancer]")
            if lb.needed:
                print("  Needed: Yes")
                print(f"  Type: {lb.load_balancer_type or 'Application Load Balancer'}")
                print(f"  Why: {lb.why}")
            else:
                print("  Needed: No")
                print(f"  Why: {lb.why}")

            # Terraform output
            tf_files = state.get("terraform_files")
            if tf_files:
                out_dir = os.path.join(os.getcwd(), "terraform_output")
                os.makedirs(out_dir, exist_ok=True)
                print(f"\n--- Terraform Output ({out_dir}) ---")
                for fname in sorted(tf_files):
                    fpath = os.path.join(out_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as fh:
                        fh.write(tf_files[fname])
                    print(f"  Written: {fpath}")

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
            try:
                state["latest_user_message"] = input("> ")
            except EOFError:
                break
            continue

        print("\nNo recommendation or follow-up question was produced. Stopping.")
        break


if __name__ == "__main__":
    main()
