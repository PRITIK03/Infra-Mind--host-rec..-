"""
FastAPI HTTP endpoint tests for the agent API layer.

All LLM calls and live-data fetches are mocked — no real external calls.
Uses FastAPI TestClient (starlette.TestClient under the hood) so the app
runs in-process. Background graph runs (asyncio.to_thread + event-loop
scheduling) are allowed to complete with a short timeout per check.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest

os.environ.setdefault("CORS_ALLOWED_ORIGIN", "http://localhost:3000")
os.environ.setdefault("PORT", "8000")

from app.api import jobs as jobs_module
from app.api.main import app, recommend_rate_limiter
from app.config import get_api_settings
from app.models.schemas import (
    CacheCandidate,
    CacheEngine,
    CacheRecommendation,
    DatabaseCandidate,
    DatabaseRecommendation,
    InstanceCandidate,
    InstanceRecommendation,
    LoadBalancerRecommendation,
    ResourceProfile,
    SystemDesignRecommendation,
    TechnicalNeeds,
    TrafficPattern,
    UserRequirements,
    WorkloadType,
)


_TRANSPORT = httpx.ASGITransport(app=app)


def _async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_TRANSPORT, base_url="http://testserver")


def _http(method: str, path: str, **kwargs) -> httpx.Response:
    """Synchronous wrapper: make an in-process HTTP request via ASGI transport."""
    async def _do() -> httpx.Response:
        async with _async_client() as c:
            return await c.request(method, path, **kwargs)
    return asyncio.run(_do())


def _get(path: str) -> httpx.Response:
    return _http("GET", path)


def _post(path: str, json: Any = None) -> httpx.Response:
    return _http("POST", path, json=json)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Keep process-local rate-limit state isolated between API tests."""
    recommend_rate_limiter.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_needs() -> TechnicalNeeds:
    return TechnicalNeeds(
        estimated_concurrency=20,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="Single instance",
        needs_database=True,
        needs_cache=True,
        load_balancer_needed=False,
        reasoning="Test system design reasoning",
    )


def _sample_compute_pool() -> list[InstanceCandidate]:
    return [
        InstanceCandidate(instance_type="t3.medium", vcpu=2, memory_gib=4.0),
        InstanceCandidate(instance_type="m5.large", vcpu=2, memory_gib=8.0),
    ]


def _sample_db_pool() -> list[DatabaseCandidate]:
    return [
        DatabaseCandidate(
            instance_type="db.t3.medium",
            family="General purpose",
            vcpu=2,
            memory_gib=4.0,
        ),
    ]


def _sample_cache_pool() -> list[CacheCandidate]:
    return [
        CacheCandidate(
            instance_type="cache.t3.medium",
            family="Standard",
            engine=CacheEngine.REDIS,
            vcpu=2,
            memory_gib=3.14,
        ),
    ]


def _sample_system_recommendation() -> SystemDesignRecommendation:
    return SystemDesignRecommendation(
        compute=InstanceRecommendation(
            recommended_instance="t3.medium",
            why="Good balanced general purpose instance",
            assumptions=["steady load"],
            confidence="high",
        ),
        database=DatabaseRecommendation(
            needed=True,
            recommended_instance="db.t3.medium",
            engine_suggestion="PostgreSQL",
            why="Standard relational storage",
            assumptions=["default"],
            confidence="high",
        ),
        cache=CacheRecommendation(
            needed=True,
            recommended_instance="cache.t3.medium",
            engine=CacheEngine.REDIS,
            why="Session cache",
            assumptions=["default"],
            confidence="high",
        ),
        load_balancer=LoadBalancerRecommendation(
            needed=False,
            why="Single instance workload",
        ),
        architecture_summary="Balanced architecture with compute, DB, and Redis cache.",
    )


def _sample_terraform() -> dict[str, str]:
    return {
        "main.tf": "terraform {\n  required_providers {\n    aws = {}\n  }\n}\n",
        "variables.tf": "variable \"aws_region\" {\n  default = \"us-east-1\"\n}\n",
        "outputs.tf": "output \"endpoint\" { value = aws_instance.app.public_ip }\n",
    }


def _wait_for(
    job_id: str,
    predicate,
    *,
    timeout: float = 8.0,
    interval: float = 0.1,
) -> dict[str, Any]:
    """Poll GET /api/recommend/{job_id} until predicate(resp) or timeout."""
    deadline = time.time() + timeout
    while True:
        resp = _get(f"/api/recommend/{job_id}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if predicate(data):
            return data
        if time.time() > deadline:
            raise AssertionError(
                f"Timed out waiting for predicate. Last response: {data}"
            )
        time.sleep(interval)


def _enter_full_mock_stack(exit_stack: contextlib.ExitStack) -> None:
    """Enter every patch into the given ExitStack.

    Patches are applied at the *graph-module* import level (the names
    build_graph() reads when wiring up nodes) plus at the node-function
    level for leaf helpers (invoke_structured, fetch_*). This double
    coverage ensures the mock takes effect both inside graph.stream()
    and when the node bodies reference these helpers at runtime, even
    from a background thread.
    """
    needs = _sample_needs()
    sys_rec = _sample_system_recommendation()
    compute_pool = _sample_compute_pool()
    db_pool = _sample_db_pool()
    cache_pool = _sample_cache_pool()
    tf_files = _sample_terraform()

    def _fake_collect(state):
        # Leave no-op: collector will typically parse user message and not ask follow-ups
        # when provided rich enough input. We still pass-through so requirement_validator
        # can mark requirements complete and proceed to downstream nodes.
        return state

    def _fake_validate(state):
        return {**state, "next_question": None, "pending_field": None}

    def _fake_reason(state):
        return {**state, "technical_needs": needs}

    def _fake_compute(state):
        return {**state, "instance_candidates": compute_pool}

    def _fake_db(state):
        return {**state, "database_candidates": db_pool}

    def _fake_cache(state):
        return {**state, "cache_candidates": cache_pool}

    def _fake_holistic(state):
        return {**state, "system_design_recommendation": sys_rec}

    def _fake_grounding(state):
        # Pass-through: mark grounding as passed on whatever SDR is present.
        sdr = state.get("system_design_recommendation")
        if sdr is not None:
            state = {**state, "system_design_recommendation": sdr.model_copy(
                update={"grounding_passed": True, "grounding_notes": []}
            )}
        return state

    def _fake_tf(state):
        return {**state, "terraform_files": tf_files}

    # Graph-level node function patches (what build_graph() wires)
    exit_stack.enter_context(
        patch("app.agent.graph.collect_requirements", side_effect=_fake_collect)
    )
    exit_stack.enter_context(
        patch("app.agent.graph.validate_requirements", side_effect=_fake_validate)
    )
    exit_stack.enter_context(
        patch("app.agent.graph.reason_system_design", side_effect=_fake_reason)
    )
    exit_stack.enter_context(
        patch("app.agent.graph.research_instances", side_effect=_fake_compute)
    )
    exit_stack.enter_context(
        patch("app.agent.graph.research_database", side_effect=_fake_db)
    )
    exit_stack.enter_context(
        patch("app.agent.graph.research_cache", side_effect=_fake_cache)
    )
    exit_stack.enter_context(
        patch("app.agent.graph.holistic_recommend", side_effect=_fake_holistic)
    )
    exit_stack.enter_context(
        patch("app.agent.graph.grounding_check", side_effect=_fake_grounding)
    )
    exit_stack.enter_context(
        patch("app.agent.graph.generate_terraform", side_effect=_fake_tf)
    )

    # Backstop: also patch node-level helper entry points so any other code path
    # that directly imports and calls them won't hit live APIs.
    exit_stack.enter_context(
        patch(
            "app.agent.nodes.system_design_reasoner.try_get_search_tool",
            return_value=None,
        )
    )
    exit_stack.enter_context(
        patch(
            "app.agent.nodes.system_design_reasoner.invoke_structured",
            return_value=needs,
        )
    )
    exit_stack.enter_context(
        patch(
            "app.agent.nodes.holistic_recommender.invoke_structured",
            return_value=sys_rec,
        )
    )
    exit_stack.enter_context(
        patch(
            "app.agent.nodes.instance_researcher.fetch_ec2_instance_data",
            return_value=compute_pool,
        )
    )
    exit_stack.enter_context(
        patch(
            "app.agent.nodes.database_researcher.fetch_rds_instance_data",
            return_value=db_pool,
        )
    )
    exit_stack.enter_context(
        patch(
            "app.agent.nodes.cache_researcher.fetch_cache_instance_data",
            return_value=cache_pool,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_endpoint():
    """GET /api/health returns 200 with ok status and no LLM/live-data calls."""
    resp = _get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "time" in body
    assert isinstance(body["time"], float)


def test_cors_origin_is_empty_and_warns_when_unset(monkeypatch, caplog):
    """Unset CORS config never falls back to a localhost origin."""
    monkeypatch.delenv("CORS_ALLOWED_ORIGIN", raising=False)
    with caplog.at_level("WARNING"):
        settings = get_api_settings()
    assert settings.cors_allowed_origin == ""
    assert "CORS_ALLOWED_ORIGIN is unset" in caplog.text


def test_recommend_rate_limit_returns_429_after_five_requests():
    """Expensive public recommendation creation is limited per client IP."""
    recommend_rate_limiter.clear()
    with patch("app.api.main._submit_with_timeout"):
        responses = [
            _post("/api/recommend", json={"message": "test workload"})
            for _ in range(5)
        ]
        blocked = _post("/api/recommend", json={"message": "test workload"})

    assert all(response.status_code == 200 for response in responses)
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"].isdigit()


@pytest.mark.parametrize(
    "message",
    ["", "   \n\t", "x" * 10_001],
)
def test_recommend_rejects_blank_or_oversized_messages(message):
    """Public request validation does not rely on the frontend."""
    response = _post("/api/recommend", json={"message": message})
    assert response.status_code == 422


def test_answer_rejects_blank_or_oversized_answers():
    """Follow-up input is bounded before it can reach the agent."""
    for answer in ("   ", "x" * 2_001):
        response = _post(
            "/api/recommend/not-a-real-job/answer",
            json={"answer": answer},
        )
        assert response.status_code == 422


def test_recommend_creates_job_and_returns_id():
    """POST /api/recommend returns a valid job_id immediately without blocking."""
    with contextlib.ExitStack() as xs:
        _enter_full_mock_stack(xs)
        start = time.time()
        resp = _post(
            "/api/recommend",
            json={"message": "I need a web app for 1000 registered users."},
        )
        elapsed = time.time() - start

    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert isinstance(body["job_id"], str) and len(body["job_id"]) > 10
    # Endpoint must be non-blocking: graph work is scheduled in a thread,
    # so HTTP response arrives well under a second even for a 3+min graph run.
    assert elapsed < 1.0, f"POST /api/recommend blocked for {elapsed:.2f}s"


def test_polling_before_completion_shows_collecting_or_running():
    """GET /api/recommend/{id} during the run reports a live status."""
    with contextlib.ExitStack() as xs:
        _enter_full_mock_stack(xs)
        create_resp = _post(
            "/api/recommend",
            json={"message": "Web app, steady load, 500 registered users."},
        )
        job_id = create_resp.json()["job_id"]

        poll_resp = _get(f"/api/recommend/{job_id}")
        assert poll_resp.status_code == 200
        poll_body = poll_resp.json()
        assert poll_body["job_id"] == job_id
        # The first poll should at a minimum see collecting/running or done;
        # we specifically check that the status is *one* of the valid ones and
        # that current_stage is populated.
        assert poll_body["status"] in {
            "collecting",
            "running",
            "awaiting_input",
            "done",
        }
        assert "current_stage" in poll_body
        assert isinstance(poll_body["current_stage"], str)
        assert poll_body["current_stage"]  # non-empty


def test_full_completion_surfaces_recommendation_and_terraform():
    """A complete mocked run surfaces SystemDesignRecommendation + terraform_files."""
    with contextlib.ExitStack() as xs:
        _enter_full_mock_stack(xs)
        create_resp = _post(
            "/api/recommend",
            json={
                "message": "Build a steady-state web application for 1000 registered users."
            },
        )
        job_id = create_resp.json()["job_id"]

        final = _wait_for(job_id, lambda d: d["status"] == "done")

    assert final["status"] == "done"
    assert "current_stage" in final
    assert "result" in final
    result = final["result"]
    assert "system_design_recommendation" in result

    rec = result["system_design_recommendation"]
    assert rec["compute"]["recommended_instance"] == "t3.medium"
    assert rec["database"]["needed"] is True
    assert rec["database"]["recommended_instance"] == "db.t3.medium"
    assert rec["cache"]["needed"] is True
    assert rec["cache"]["recommended_instance"] == "cache.t3.medium"
    assert rec["load_balancer"]["needed"] is False
    assert "architecture_summary" in rec

    assert "terraform_files" in result
    tf = result["terraform_files"]
    assert set(tf.keys()) >= {"main.tf", "variables.tf", "outputs.tf"}
    assert "terraform" in tf["main.tf"]


def test_awaiting_input_scenario_surfaces_question_and_accepts_answer():
    """When requirement_validator asks a follow-up, status -> awaiting_input,
    next_question is returned, and POST /answer resumes the run."""

    # First pass: validations sets next_question and returns from graph.
    # Second pass (after answer): collector fills the gap, validation passes,
    # remainder of graph runs and produces a recommendation.
    needs = _sample_needs()
    sys_rec = _sample_system_recommendation()
    tf_files = _sample_terraform()

    pass_calls = {"count": 0}

    def _fake_collector_node(state):
        if pass_calls["count"] == 0:
            reqs = UserRequirements(
                registered_users=500,
                traffic_pattern=TrafficPattern.STEADY,
            )
        else:
            reqs = UserRequirements(
                workload_type=WorkloadType.WEB_APP,
                registered_users=500,
                traffic_pattern=TrafficPattern.STEADY,
            )
        return {**state, "requirements": reqs, "latest_user_message": None}

    def _fake_validator_node(state):
        if pass_calls["count"] == 0:
            return {
                **state,
                "next_question": "What type of workload is this? (web_app, api_service, batch_processing, ml_inference, other)",
                "pending_field": "workload_type",
            }
        return {**state, "next_question": None, "pending_field": None}

    def _fake_reasoner_node(state):
        return {**state, "technical_needs": needs}

    def _fake_instance_node(state):
        return {**state, "instance_candidates": _sample_compute_pool()}

    def _fake_db_node(state):
        return {**state, "database_candidates": _sample_db_pool()}

    def _fake_cache_node(state):
        return {**state, "cache_candidates": _sample_cache_pool()}

    def _fake_holistic_node(state):
        return {**state, "system_design_recommendation": sys_rec}

    def _fake_grounding_node(state):
        sdr = state.get("system_design_recommendation")
        if sdr is not None:
            state = {**state, "system_design_recommendation": sdr.model_copy(
                update={"grounding_passed": True, "grounding_notes": []}
            )}
        return state

    def _fake_tf_node(state):
        return {**state, "terraform_files": tf_files}

    with (
        patch(
            "app.agent.graph.collect_requirements",
            side_effect=_fake_collector_node,
        ),
        patch(
            "app.agent.graph.validate_requirements",
            side_effect=_fake_validator_node,
        ),
        patch(
            "app.agent.graph.reason_system_design",
            side_effect=_fake_reasoner_node,
        ),
        patch(
            "app.agent.graph.research_instances",
            side_effect=_fake_instance_node,
        ),
        patch(
            "app.agent.graph.research_database",
            side_effect=_fake_db_node,
        ),
        patch(
            "app.agent.graph.research_cache",
            side_effect=_fake_cache_node,
        ),
        patch(
            "app.agent.graph.holistic_recommend",
            side_effect=_fake_holistic_node,
        ),
        patch(
            "app.agent.graph.grounding_check",
            side_effect=_fake_grounding_node,
        ),
        patch(
            "app.agent.graph.generate_terraform",
            side_effect=_fake_tf_node,
        ),
    ):
        create_resp = _post(
            "/api/recommend",
            json={"message": "I have 500 users on a steady traffic pattern."},
        )
        job_id = create_resp.json()["job_id"]

        awaiting = _wait_for(
            job_id,
            lambda d: d["status"] == "awaiting_input" or d["status"] == "done",
        )

        if awaiting["status"] == "awaiting_input":
            assert "next_question" in awaiting
            assert isinstance(awaiting["next_question"], str)
            assert len(awaiting["next_question"]) > 10
            # The follow-up question should be related to app type / workload
            lowered = awaiting["next_question"].lower()
            assert (
                "workload" in lowered
                or "application" in lowered
                or "web app" in lowered
                or "type" in lowered
            )

            pass_calls["count"] = 1
            answer_resp = _post(
                f"/api/recommend/{job_id}/answer",
                json={"answer": "It's a web application."},
            )
            assert answer_resp.status_code == 200
            assert answer_resp.json()["status"] == "running"

            final = _wait_for(job_id, lambda d: d["status"] == "done")
        else:
            final = awaiting

    assert final["status"] == "done"
    result = final["result"]
    assert "system_design_recommendation" in result
    assert "terraform_files" in result


def test_job_not_found_returns_404():
    """GET /api/recommend/<bogus> returns 404."""
    resp = _get("/api/recommend/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_answer_when_not_awaiting_returns_400():
    """POST /answer on a job that isn't in awaiting_input yields 400."""
    dummy_id = "00000000-0000-0000-0000-000000000099"
    state = {
        "requirements": UserRequirements(),
        "latest_user_message": "x",
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
    from app.api.main import jobs as api_jobs_store

    api_jobs_store.put(
        jobs_module.Job(
            job_id=dummy_id,
            status="collecting",
            current_stage="Initializing",
            state=state,
        )
    )
    resp = _post(f"/api/recommend/{dummy_id}/answer", json={"answer": "test"})
    assert resp.status_code == 400
    assert "not awaiting input" in resp.json()["detail"].lower()


def test_stage_label_mapping_covers_all_v2_nodes():
    """STAGE_LABELS should provide human-readable text for every V2 graph node."""
    v2_nodes = {
        "collect_requirements",
        "validate_requirements",
        "reason_system_design",
        "research_instances",
        "research_database",
        "research_cache",
        "holistic_recommend",
        "grounding_check",
        "generate_terraform",
    }
    for node in v2_nodes:
        label = jobs_module.label_for_node(node)
        assert isinstance(label, str)
        assert len(label) >= 5
