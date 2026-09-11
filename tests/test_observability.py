"""
Tests for the observability layer (app/observability.py).

Strategy
--------
- "Unconfigured" path: DATABASE_URL unset → persist_run is a no-op (stdout
  log only), get_runs returns the empty sentinel response with
  observability_configured=False.
- "Configured" path: we point DATABASE_URL at an in-process SQLite DB so
  we can test real persistence without a network dependency.  SQLite is
  the same code path as Postgres via SQLAlchemy Core, so it gives us real
  coverage without mocking the engine internals.

reset_engine_for_testing() is called before every test that changes the
DATABASE_URL environment variable, so lazy-init state never bleeds between
tests.
"""

from __future__ import annotations

import os
import uuid

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_obs_state():
    """Reset the observability module's lazy-init state before every test."""
    import app.observability as obs
    obs.reset_engine_for_testing()
    # Also clear any DATABASE_URL left over from previous tests.
    os.environ.pop("DATABASE_URL", None)
    yield
    obs.reset_engine_for_testing()
    os.environ.pop("DATABASE_URL", None)


@pytest.fixture()
def sqlite_db(tmp_path):
    """
    Point DATABASE_URL at a fresh SQLite file in a temp dir and return the URL.
    The observability module is reset so it initialises against this DB.
    """
    import app.observability as obs
    db_path = tmp_path / "test_runs.db"
    url = f"sqlite:///{db_path}"
    os.environ["DATABASE_URL"] = url
    obs.reset_engine_for_testing()
    return url


# ---------------------------------------------------------------------------
# Unconfigured path
# ---------------------------------------------------------------------------

def test_get_runs_returns_empty_when_unconfigured():
    """
    No DATABASE_URL → get_runs must return an empty list with
    observability_configured=False, not an error.
    """
    from app.observability import get_runs
    result = get_runs()
    assert result["runs"] == []
    assert result["total"] == 0
    assert result["observability_configured"] is False
    assert "error" not in result


def test_get_runs_pagination_fields_present_when_unconfigured():
    """page/page_size fields are echoed back even when unconfigured."""
    from app.observability import get_runs
    result = get_runs(page=3, page_size=5)
    assert result["page"] == 3
    assert result["page_size"] == 5
    assert result["observability_configured"] is False


def test_persist_run_does_not_raise_when_unconfigured(caplog):
    """
    persist_run must complete silently (stdout log only) when the DB is
    not configured — no exception, no crash.
    """
    import logging
    from app.observability import persist_run

    with caplog.at_level(logging.INFO, logger="app.observability"):
        persist_run(
            job_id="test-unconfigured-123",
            total_latency_s=5.0,
            model_used="test-model",
            retry_count=0,
            grounding_passed=True,
            estimated_cost_low=10.0,
            estimated_cost_high=15.0,
        )

    # A summary line must have been logged to stdout (captured by caplog).
    assert any("test-unconfigured-123" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Configured path (SQLite)
# ---------------------------------------------------------------------------

def test_persist_run_writes_row(sqlite_db):
    """A persisted run must appear in get_runs results."""
    from app.observability import persist_run, get_runs

    job_id = str(uuid.uuid4())
    persist_run(
        job_id=job_id,
        total_latency_s=12.3,
        model_used="gpt-4o",
        retry_count=1,
        grounding_passed=True,
        estimated_cost_low=5.50,
        estimated_cost_high=8.20,
        recommendation_snapshot={
            "compute_instance": "m6g.large",
            "database_instance": "db.t3.small",
            "database_engine": "PostgreSQL",
            "cache_instance": None,
            "cache_engine": None,
            "load_balancer_type": "Application Load Balancer",
            "min_instances": 2,
            "max_instances": 6,
        },
    )

    result = get_runs()
    assert result["observability_configured"] is True
    assert result["total"] == 1
    assert len(result["runs"]) == 1

    row = result["runs"][0]
    assert row["job_id"] == job_id
    assert abs(row["total_latency_s"] - 12.3) < 0.01
    assert row["model_used"] == "gpt-4o"
    assert row["retry_count"] == 1
    assert row["grounding_passed"] is True
    assert abs(row["estimated_cost_low"] - 5.50) < 0.01
    assert abs(row["estimated_cost_high"] - 8.20) < 0.01
    assert row["recommendation_snapshot"] == {
        "compute_instance": "m6g.large",
        "database_instance": "db.t3.small",
        "database_engine": "PostgreSQL",
        "cache_instance": None,
        "cache_engine": None,
        "load_balancer_type": "Application Load Balancer",
        "min_instances": 2,
        "max_instances": 6,
    }


def test_persist_run_null_fields_accepted(sqlite_db):
    """Nullable fields (cost, grounding_passed, latency) must accept None."""
    from app.observability import persist_run, get_runs

    job_id = str(uuid.uuid4())
    persist_run(
        job_id=job_id,
        total_latency_s=None,
        model_used=None,
        retry_count=0,
        grounding_passed=None,
        estimated_cost_low=None,
        estimated_cost_high=None,
    )

    result = get_runs()
    row = result["runs"][0]
    assert row["job_id"] == job_id
    assert row["total_latency_s"] is None
    assert row["model_used"] is None
    assert row["grounding_passed"] is None


def test_get_runs_pagination(sqlite_db):
    """Pagination: page 1 of page_size 2 returns first 2 rows; total reflects all."""
    from app.observability import persist_run, get_runs

    ids = [str(uuid.uuid4()) for _ in range(5)]
    for i, jid in enumerate(ids):
        persist_run(
            job_id=jid,
            total_latency_s=float(i),
            model_used="m",
            retry_count=0,
            grounding_passed=None,
            estimated_cost_low=None,
            estimated_cost_high=None,
        )

    page1 = get_runs(page=1, page_size=2)
    assert page1["total"] == 5
    assert len(page1["runs"]) == 2

    page3 = get_runs(page=3, page_size=2)
    assert page3["total"] == 5
    assert len(page3["runs"]) == 1  # only one row left on the last page


def test_duplicate_job_id_is_ignored(sqlite_db):
    """ON CONFLICT DO NOTHING: inserting the same job_id twice doesn't error or duplicate."""
    from app.observability import persist_run, get_runs

    job_id = str(uuid.uuid4())
    persist_run(
        job_id=job_id,
        total_latency_s=1.0,
        model_used="first",
        retry_count=0,
        grounding_passed=True,
        estimated_cost_low=None,
        estimated_cost_high=None,
    )
    persist_run(
        job_id=job_id,  # same id — should be silently ignored
        total_latency_s=2.0,
        model_used="second",
        retry_count=0,
        grounding_passed=False,
        estimated_cost_low=None,
        estimated_cost_high=None,
    )

    result = get_runs()
    assert result["total"] == 1
    # First write wins (ON CONFLICT DO NOTHING).
    assert result["runs"][0]["model_used"] == "first"


def test_grounding_failed_row_stored_correctly(sqlite_db):
    """grounding_passed=False is round-tripped correctly (not coerced to True)."""
    from app.observability import persist_run, get_runs

    job_id = str(uuid.uuid4())
    persist_run(
        job_id=job_id,
        total_latency_s=30.0,
        model_used="claude-3",
        retry_count=2,
        grounding_passed=False,
        estimated_cost_low=1.0,
        estimated_cost_high=2.0,
    )

    row = get_runs()["runs"][0]
    assert row["grounding_passed"] is False
    assert row["retry_count"] == 2
