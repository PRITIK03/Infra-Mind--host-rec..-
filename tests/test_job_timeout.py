"""
Tests for the wall-clock job timeout in app/api/main.py.

Verifies that _submit_with_timeout marks a job as "error" with a clear
timeout message when the underlying graph run exceeds JOB_TIMEOUT_SECONDS,
and that normal fast jobs complete without interference from the watchdog.

All graph execution is mocked — no real LLM or live-data calls.
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest

os.environ.setdefault("CORS_ALLOWED_ORIGIN", "http://localhost:3000")
os.environ.setdefault("PORT", "8000")

from app.api import jobs as jobs_module
from app.api.main import (
    JOB_TIMEOUT_SECONDS,
    _submit_with_timeout,
    _graph_loop_sync,
    jobs,
)
from app.models.schemas import UserRequirements


def _make_state():
    return {
        "requirements": UserRequirements(),
        "latest_user_message": "test message",
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


def _register_job(job_id: str) -> None:
    jobs.put(
        jobs_module.Job(
            job_id=job_id,
            status="collecting",
            current_stage="Initializing",
            state=_make_state(),
        )
    )


def _wait_for_status(job_id: str, *, timeout: float = 5.0, interval: float = 0.05) -> str:
    """Poll the in-memory job store until status is terminal or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = jobs.get(job_id)
        if job and job.status in {"done", "error"}:
            return job.status
        time.sleep(interval)
    job = jobs.get(job_id)
    return job.status if job else "unknown"


# ---------------------------------------------------------------------------
# Timeout fires and marks job as error
# ---------------------------------------------------------------------------


def test_job_timeout_marks_job_as_error_when_exceeded():
    """
    When the underlying graph function hangs beyond JOB_TIMEOUT_SECONDS,
    _submit_with_timeout must mark the job as "error" with a descriptive message.

    We patch JOB_TIMEOUT_SECONDS to a tiny value (0.1s) and make the graph
    function sleep longer than that, then confirm the job reaches "error".
    """
    job_id = "timeout-test-001"
    _register_job(job_id)

    def _slow_graph(jid, state):
        # Sleeps longer than the patched timeout — simulates a true hang.
        time.sleep(2.0)
        # Should never reach here in the test, but mark done if it somehow does.
        jobs.update_status(jid, "done")

    with patch("app.api.main.JOB_TIMEOUT_SECONDS", 0.1):
        _submit_with_timeout(_slow_graph, job_id, _make_state())

    # Give the watchdog thread time to fire (timeout=0.1s + margin)
    final_status = _wait_for_status(job_id, timeout=3.0)

    assert final_status == "error", f"Expected 'error', got '{final_status}'"

    job = jobs.get(job_id)
    assert job is not None
    assert job.error is not None
    assert "timed out" in job.error.lower(), (
        f"Expected 'timed out' in error message, got: {job.error!r}"
    )


def test_job_timeout_does_not_interfere_with_fast_job():
    """
    A job that completes quickly must reach 'done' without the watchdog
    incorrectly overwriting it with 'error'.
    """
    job_id = "timeout-test-002"
    _register_job(job_id)

    def _fast_graph(jid, state):
        jobs.update_status(jid, "done")

    # Use a generous timeout — fast job should finish well before it.
    with patch("app.api.main.JOB_TIMEOUT_SECONDS", 5.0):
        _submit_with_timeout(_fast_graph, job_id, _make_state())

    final_status = _wait_for_status(job_id, timeout=3.0)

    assert final_status == "done", f"Expected 'done', got '{final_status}'"
    job = jobs.get(job_id)
    assert job.error is None


# ---------------------------------------------------------------------------
# Timeout message content
# ---------------------------------------------------------------------------


def test_job_timeout_error_message_is_user_facing():
    """
    The timeout error message must be human-readable and include the
    timeout duration — suitable for surfacing in an API response.
    """
    job_id = "timeout-test-003"
    _register_job(job_id)

    def _hangs(jid, state):
        time.sleep(10.0)

    with patch("app.api.main.JOB_TIMEOUT_SECONDS", 0.1):
        _submit_with_timeout(_hangs, job_id, _make_state())

    _wait_for_status(job_id, timeout=3.0)

    job = jobs.get(job_id)
    assert job is not None and job.error is not None
    msg = job.error.lower()
    assert "timed out" in msg
    assert "try again" in msg


# ---------------------------------------------------------------------------
# Error from graph function still surfaces correctly (not swallowed by watchdog)
# ---------------------------------------------------------------------------


def test_job_timeout_does_not_swallow_graph_errors():
    """
    If the graph function raises an exception and marks the job as error
    on its own, the watchdog must not overwrite it with a timeout message.
    """
    job_id = "timeout-test-004"
    _register_job(job_id)

    def _raises(jid, state):
        jobs.update_status(jid, "error", error="GraphError: something bad happened")

    with patch("app.api.main.JOB_TIMEOUT_SECONDS", 5.0):
        _submit_with_timeout(_raises, job_id, _make_state())

    final_status = _wait_for_status(job_id, timeout=3.0)

    assert final_status == "error"
    job = jobs.get(job_id)
    # Original error message must be preserved, not replaced by timeout message
    assert "GraphError" in (job.error or ""), (
        f"Expected original error preserved, got: {job.error!r}"
    )
    assert "timed out" not in (job.error or "").lower(), (
        "Watchdog must not overwrite an already-set error with a timeout message"
    )
