"""
FastAPI application exposing the LangGraph agent over HTTP.

Agent invocations can take 2-3+ minutes under free-tier LLM load, so all
work is dispatched to a background thread and clients poll via
GET /api/recommend/{job_id} for progress. The agent's multi-turn
requirement-collection loop is surfaced via the awaiting_input status +
POST /api/recommend/{job_id}/answer.

Thread-pool sizing
------------------
We use an explicit ThreadPoolExecutor rather than the default
asyncio.to_thread executor (which uses ThreadPoolExecutor(max_workers=None),
defaulting to min(32, os.cpu_count() + 4) — potentially 36 threads on a
4-core box, or just the OS default on a constrained host).

On a small Render/Railway/Fly instance (1–2 vCPUs, 512 MB – 1 GB RAM),
each agent thread holds a live HTTP connection + LangGraph state + LLM
response buffers. Running many concurrent threads on such a host causes
memory pressure and scheduler thrashing before the concurrency limit
matters. JOB_THREAD_POOL_SIZE=8 is deliberately conservative: it
allows meaningful concurrency (8 simultaneous agent runs) while leaving
headroom for the FastAPI worker, uvicorn I/O loop, and OS overhead.

If a job hangs (even after the LLM-layer fixes), JOB_TIMEOUT_SECONDS
ensures the slot is returned within a bounded time.  Adjust both
constants via environment variables for larger hosts.

Wall-clock job timeout
----------------------
Each background job is submitted via executor.submit() and tracked with
Future.result(timeout=JOB_TIMEOUT_SECONDS).  A concurrent.futures.TimeoutError
marks the job as "error" with a clear message — this is an independent
safety net that fires regardless of what's happening inside the graph,
protecting against any future hang scenario, not just rate-limit loops.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time
import uuid
from collections import deque
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.api.jobs import Job, JobStore, JobStatus, label_for_node
from app.config import get_api_settings
from app.models.schemas import SystemDesignRecommendation, UserRequirements

# ---------------------------------------------------------------------------
# Concurrency + timeout constants (overridable via env for larger hosts)
# ---------------------------------------------------------------------------

# Max simultaneous agent-run threads. Conservative for small cloud hosts
# (Render free/starter, Railway, Fly.io shared-cpu-1x).
JOB_THREAD_POOL_SIZE: int = int(os.getenv("JOB_THREAD_POOL_SIZE", "8"))

# Hard wall-clock limit for a single complete agent run, in seconds.
# A full run under free-tier rate limiting can legitimately take 3-4 min;
# 10 min is generous enough for paid keys while bounding any true hang.
JOB_TIMEOUT_SECONDS: float = float(os.getenv("JOB_TIMEOUT_SECONDS", "600"))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must contain non-whitespace text")
        return value


class AnswerRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=2_000)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer must contain non-whitespace text")
        return value


api_settings = get_api_settings()

app = FastAPI(title="AWS Instance Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        [api_settings.cors_allowed_origin]
        if api_settings.cors_allowed_origin
        else []
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = JobStore()


class InMemoryRateLimiter:
    """Small process-local sliding-window limiter for expensive public requests."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max(1, max_requests)
        self.window_seconds = max(1.0, window_seconds)
        self._requests: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client_id: str, now: float | None = None) -> tuple[bool, int]:
        current = time.monotonic() if now is None else now
        with self._lock:
            timestamps = self._requests.setdefault(client_id, deque())
            cutoff = current - self.window_seconds
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (current - timestamps[0])))
                return False, retry_after
            timestamps.append(current)
            return True, 0

    def clear(self) -> None:
        """Clear state for tests and controlled in-process maintenance."""
        with self._lock:
            self._requests.clear()


RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "5"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
recommend_rate_limiter = InMemoryRateLimiter(
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)

# Single shared executor for all background agent runs.
# Defined at module level so it is shared across requests and can be
# cleanly shut down on process exit.
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=JOB_THREAD_POOL_SIZE,
    thread_name_prefix="agent-job",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_state() -> AgentState:
    return {
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


def _run_graph_with_streaming(
    job_id: str,
    state: AgentState,
    collecting: bool = False,
) -> AgentState:
    """
    Run one graph pass using .stream() so we surface per-node stage
    labels to the job. Returns the final state after the full pass.

    Sets the thread-local retry context var so that invoke_structured
    calls made from any node in this pass automatically update
    job.retry_info without requiring any changes to node signatures.
    The context var is cleared after the pass completes.
    """
    from app.llm.retry import _retry_context

    def _retry_cb(attempt: int, max_attempts: int) -> None:
        jobs.update_retry_info(
            job_id,
            f"Retrying after rate limit (attempt {attempt} of {max_attempts})",
        )

    def _clear_retry_cb(attempt: int, max_attempts: int) -> None:  # noqa: ARG001
        # Sentinel: called with attempt=0 to signal "clear".
        jobs.update_retry_info(job_id, None)

    token = _retry_context.set(_retry_cb)
    try:
        graph = build_graph()
        final_state: AgentState = state

        for chunk in graph.stream(state):
            node_name = next(iter(chunk.keys()))
            stage_label = label_for_node(node_name)
            jobs.update_stage(job_id, stage_label)
            final_state = chunk[node_name]
            jobs.update_state(job_id, final_state)
            # Clear retry_info after each node completes successfully.
            jobs.update_retry_info(job_id, None)
    finally:
        _retry_context.reset(token)

    return final_state


def _graph_loop_sync(job_id: str, initial_state: AgentState) -> None:
    """
    Synchronous (thread-bound) driver that mirrors the CLI loop in
    app/main.py but writes progress into the shared JobStore.

    Runs passes of graph.stream(state) until either a final
    recommendation is produced or the agent asks a follow-up question.
    On any unhandled exception the job is marked errored.

    This function is submitted to _executor and monitored by
    _submit_with_timeout, which enforces JOB_TIMEOUT_SECONDS as an
    independent wall-clock safety net.
    """
    state = initial_state
    job_start = time.monotonic()
    try:
        while True:
            state = _run_graph_with_streaming(job_id, state, collecting=True)

            if (
                state.get("system_design_recommendation") is not None
                or state.get("recommendation") is not None
            ):
                final = _serialize_result(state)
                jobs.update_status(job_id, "done", result=final)
                _record_completed_run(job_id, state, job_start)
                return

            if state.get("next_question"):
                jobs.update_status(
                    job_id,
                    "awaiting_input",
                    next_question=state["next_question"],
                )
                return

            jobs.update_status(
                job_id,
                "error",
                error="No recommendation or follow-up question was produced.",
            )
            return

    except Exception as exc:
        jobs.update_status(job_id, "error", error=f"{type(exc).__name__}: {exc}")


def _resume_with_answer_sync(job_id: str, answer: str) -> None:
    """Resume a job in awaiting_input status after the user replies."""
    job = jobs.get(job_id)
    if job is None:
        return
    state = job.state
    state["latest_user_message"] = answer
    jobs.update_status(job_id, "running")
    job_start = time.monotonic()
    try:
        while True:
            state = _run_graph_with_streaming(job_id, state, collecting=True)

            if (
                state.get("system_design_recommendation") is not None
                or state.get("recommendation") is not None
            ):
                final = _serialize_result(state)
                jobs.update_status(job_id, "done", result=final)
                _record_completed_run(job_id, state, job_start)
                return

            if state.get("next_question"):
                jobs.update_status(
                    job_id,
                    "awaiting_input",
                    next_question=state["next_question"],
                )
                return

            jobs.update_status(
                job_id,
                "error",
                error="No recommendation or follow-up question was produced.",
            )
            return

    except Exception as exc:
        jobs.update_status(job_id, "error", error=f"{type(exc).__name__}: {exc}")


def _submit_with_timeout(fn, *args) -> None:
    """
    Submit *fn(*args)* to the shared executor and watch it with a
    daemon thread that enforces JOB_TIMEOUT_SECONDS.

    If the future does not complete in time, the job is marked as
    "error" with a clear timeout message.  The underlying thread
    continues running until it naturally exits (Python threads cannot
    be forcibly killed), but the job slot is freed from the caller's
    perspective and the executor queue is unblocked.

    The job_id is always the first positional argument by convention.
    """
    job_id: str = args[0]
    future = _executor.submit(fn, *args)

    def _watchdog() -> None:
        try:
            future.result(timeout=JOB_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            jobs.update_status(
                job_id,
                "error",
                error=(
                    f"Job timed out after {JOB_TIMEOUT_SECONDS:.0f}s — "
                    "the agent took too long to respond. Please try again."
                ),
            )
        except Exception:
            # The underlying fn already wrote its own error via jobs.update_status;
            # nothing to do here — exceptions from the future are already handled
            # inside _graph_loop_sync / _resume_with_answer_sync.
            pass

    import threading
    threading.Thread(target=_watchdog, daemon=True, name=f"watchdog-{job_id}").start()


def _record_completed_run(
    job_id: str,
    state: AgentState,
    job_start: float,
) -> None:
    """
    Fire-and-forget observability record for a successfully completed job.
    Extracts grounding status and cost from the final state and calls
    persist_run, which logs to stdout and optionally writes to the DB.
    """
    try:
        from app.observability import persist_run
        from app.config import get_llm_settings

        total_latency_s = time.monotonic() - job_start

        # Model name from config — this is the model that produced the run.
        try:
            model_used = get_llm_settings().model_name
        except Exception:
            model_used = None

        # Retry count: read from the job's current retry_info string.
        # retry_info is None (success) or a string like "Retrying … (attempt N of M)".
        # We track the highest attempt number seen; for a clean run it's 0.
        job = jobs.get(job_id)
        retry_count = 0
        if job is not None and job.retry_info:
            import re
            m = re.search(r"attempt (\d+)", job.retry_info)
            if m:
                retry_count = int(m.group(1))

        # Grounding result and cost from the recommendation.
        grounding_passed: bool | None = None
        cost_low: float | None = None
        cost_high: float | None = None

        sdr = state.get("system_design_recommendation")
        if sdr is not None:
            if hasattr(sdr, "grounding_passed"):
                grounding_passed = sdr.grounding_passed
            cost = getattr(sdr, "estimated_cost", None)
            if cost is not None:
                cost_low = getattr(cost, "total_monthly_low", None)
                cost_high = getattr(cost, "total_monthly_high", None)

        persist_run(
            job_id=job_id,
            total_latency_s=total_latency_s,
            model_used=model_used,
            retry_count=retry_count,
            grounding_passed=grounding_passed,
            estimated_cost_low=cost_low,
            estimated_cost_high=cost_high,
        )
    except Exception as exc:
        # Observability must never crash the response path.
        import logging
        logging.getLogger(__name__).warning(
            "Observability record failed for job %s: %s", job_id, exc
        )


def _serialize_result(state: AgentState) -> dict[str, Any]:
    rec = state.get("system_design_recommendation")
    v1_rec = state.get("recommendation")
    tf_files = state.get("terraform_files")
    tn = state.get("technical_needs")
    requirements = state.get("requirements")
    candidates = state.get("instance_candidates")
    result: dict[str, Any] = {}
    if rec is not None:
        if isinstance(rec, SystemDesignRecommendation):
            result["system_design_recommendation"] = rec.model_dump(mode="json")
        else:
            result["system_design_recommendation"] = rec
    if v1_rec is not None:
        if hasattr(v1_rec, "model_dump"):
            result["recommendation"] = v1_rec.model_dump(mode="json")
        else:
            result["recommendation"] = v1_rec
    if tf_files is not None:
        result["terraform_files"] = tf_files
    # Include technical_needs and instance_candidates so the frontend can
    # render ScalingRangeBar and CandidateLandscape from real data.
    if tn is not None:
        if hasattr(tn, "model_dump"):
            result["technical_needs"] = tn.model_dump(mode="json")
        else:
            result["technical_needs"] = tn
    if requirements is not None:
        if hasattr(requirements, "model_dump"):
            result["user_requirements"] = requirements.model_dump(mode="json")
        else:
            result["user_requirements"] = requirements
    if candidates:
        result["instance_candidates"] = [
            c.model_dump(mode="json") if hasattr(c, "model_dump") else c
            for c in candidates
        ]
    return result


def _job_response(job: Job) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "created_at": job.created_at,
    }
    if job.retry_info is not None:
        resp["retry_info"] = job.retry_info
    if job.status == "awaiting_input" and job.next_question is not None:
        resp["next_question"] = job.next_question
    if job.status == "done" and job.result is not None:
        resp["result"] = job.result
    if job.status == "error" and job.error is not None:
        resp["error"] = job.error
    return resp


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness probe. No LLM or live-data calls."""
    return {"status": "ok", "time": time.time()}


# Simple in-process cache so the landing page stat readout doesn't hammer
# Vantage on every page load.  TTL of 10 minutes is generous — instance
# counts don't change mid-session.
_stats_cache: dict[str, Any] = {}
_stats_cache_time: float = 0.0
_STATS_TTL_S: float = 600.0


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    """
    Returns live counts of tracked instance types.
    Used by the landing page readout — cached for _STATS_TTL_S seconds.
    """
    global _stats_cache, _stats_cache_time
    now = time.time()
    if _stats_cache and (now - _stats_cache_time) < _STATS_TTL_S:
        return _stats_cache

    from app.tools.aws_instance_data import (
        fetch_ec2_instance_data,
        fetch_rds_instance_data,
        fetch_cache_instance_data,
        InstanceDataUnavailableError,
    )
    counts: dict[str, int] = {}
    for key, fetcher in [
        ("ec2", fetch_ec2_instance_data),
        ("rds", fetch_rds_instance_data),
        ("cache", fetch_cache_instance_data),
    ]:
        try:
            counts[key] = len(fetcher())
        except InstanceDataUnavailableError:
            counts[key] = 0

    _stats_cache = counts
    _stats_cache_time = now
    return counts


@app.post("/api/recommend")
def create_recommend_job(request: Request, req: RecommendRequest) -> dict[str, Any]:
    """Kick off a new agent run. Returns immediately with a job_id to poll."""
    client_id = request.client.host if request.client else "unknown"
    allowed, retry_after = recommend_rate_limiter.allow(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many recommendation requests. Please try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )
    job_id = str(uuid.uuid4())
    state = _empty_state()
    state["latest_user_message"] = req.message

    job = Job(
        job_id=job_id,
        status="collecting",
        current_stage="Initializing",
        state=state,
    )
    jobs.put(job)
    _submit_with_timeout(_graph_loop_sync, job_id, state)

    return {"job_id": job_id}


@app.get("/api/recommend/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll the current state / progress / result of a job."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_response(job)


@app.post("/api/recommend/{job_id}/answer")
def answer_question(job_id: str, req: AnswerRequest) -> dict[str, Any]:
    """
    Provide the user's reply to a follow-up question. Only valid when
    the job is in 'awaiting_input' status. Resumes execution in the
    background.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != "awaiting_input":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not awaiting input (current status: {job.status})",
        )

    _submit_with_timeout(_resume_with_answer_sync, job_id, req.answer)

    return {"job_id": job_id, "status": "running"}


@app.get("/api/runs")
def get_runs(page: int = 1, page_size: int = 20) -> dict[str, Any]:
    """
    Return paginated run history from the observability store.

    When DATABASE_URL is not configured, returns an empty list with
    ``observability_configured: false`` — not an error.

    Query params:
      page      (int, default 1)      — 1-based page number
      page_size (int, default 20)     — rows per page, capped at 100
    """
    from app.observability import get_runs as _get_runs
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    return _get_runs(page=page, page_size=page_size)
