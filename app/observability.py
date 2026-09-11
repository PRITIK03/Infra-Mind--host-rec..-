"""
Optional run-history observability layer.

Design principles
-----------------
- Graceful when unconfigured: if DATABASE_URL is unset, every public
  function is a no-op or returns an empty result — exactly the same
  pattern used for Tavily search.
- Lightweight: one table, one row per completed job.  Uses SQLAlchemy
  Core (not ORM) with ``create_engine`` so any DATABASE_URL-supported
  backend works (Postgres via Neon/Supabase, SQLite for local dev, etc.).
- Thread-safe: engine is created once at import time (if configured) and
  shared across threads via connection pooling.
- Never imported at module level by callers that don't need it — import
  is always guarded by ``if _engine is not None`` or similar so missing
  sqlalchemy does NOT crash the app when the package isn't installed.

Table schema (``run_history``)
-------------------------------
  job_id            TEXT PRIMARY KEY
  timestamp         TIMESTAMP WITH TIME ZONE  (UTC, server default)
  total_latency_s   FLOAT    seconds from job start to done/error
  model_used        TEXT     model name that produced the final response
  retry_count       INT      total LLM retry attempts across the run
  grounding_passed  BOOL     NULL when grounding check didn't run
  estimated_cost_low   FLOAT NULL
  estimated_cost_high  FLOAT NULL
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy engine — built once on first call, None when unconfigured.
# We import SQLAlchemy lazily so the whole app still starts without it.
# ---------------------------------------------------------------------------

_engine: Any = None  # sqlalchemy Engine or None
_initialized: bool = False


def _get_engine() -> Any:
    """Return the shared SQLAlchemy engine, or None if DB is not configured."""
    global _engine, _initialized
    if _initialized:
        return _engine

    _initialized = True
    from app.config import get_observability_settings
    settings = get_observability_settings()
    if settings.database_url is None:
        return None

    try:
        from sqlalchemy import create_engine  # type: ignore[import]
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
        _ensure_table(_engine)
        logger.info("Observability DB connected: %s", _redact_url(settings.database_url))
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Observability DB setup failed (%s); falling back to stdout-only logging.",
            exc,
        )
        _engine = None

    return _engine


def _redact_url(url: str) -> str:
    """Remove password from a DB URL for safe logging."""
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        redacted = parsed._replace(netloc=parsed.hostname or "")
        return urlunparse(redacted)
    except Exception:
        return "<db-url>"


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS run_history (
    job_id               TEXT        PRIMARY KEY,
    timestamp            TIMESTAMP   NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    total_latency_s      FLOAT,
    model_used           TEXT,
    retry_count          INTEGER     NOT NULL DEFAULT 0,
    grounding_passed     BOOLEAN,
    estimated_cost_low   FLOAT,
    estimated_cost_high  FLOAT
)
"""

# SQLite-compatible variant (no timezone cast, no AT TIME ZONE)
_TABLE_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS run_history (
    job_id               TEXT        PRIMARY KEY,
    timestamp            TEXT        NOT NULL DEFAULT (datetime('now')),
    total_latency_s      FLOAT,
    model_used           TEXT,
    retry_count          INTEGER     NOT NULL DEFAULT 0,
    grounding_passed     INTEGER,
    estimated_cost_low   FLOAT,
    estimated_cost_high  FLOAT
)
"""


def _ensure_table(engine: Any) -> None:
    """Create run_history table if it doesn't exist."""
    ddl = (
        _TABLE_DDL_SQLITE
        if engine.dialect.name == "sqlite"
        else _TABLE_DDL
    )
    with engine.begin() as conn:
        conn.execute(_text(ddl))


def _text(sql: str) -> Any:
    from sqlalchemy import text  # type: ignore[import]
    return text(sql)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def persist_run(
    *,
    job_id: str,
    total_latency_s: float | None,
    model_used: str | None,
    retry_count: int,
    grounding_passed: bool | None,
    estimated_cost_low: float | None,
    estimated_cost_high: float | None,
) -> None:
    """
    Persist one row to run_history.  No-op (stdout log only) when DB is
    not configured or unavailable.
    """
    # Always log to stdout — this is the fallback and the audit trail.
    logger.info(
        "run_summary job_id=%s latency=%.1fs model=%s retries=%d "
        "grounding_passed=%s cost_low=%s cost_high=%s",
        job_id,
        total_latency_s or 0.0,
        model_used or "unknown",
        retry_count,
        grounding_passed,
        estimated_cost_low,
        estimated_cost_high,
    )

    engine = _get_engine()
    if engine is None:
        return

    try:
        with engine.begin() as conn:
            conn.execute(
                _text(
                    """
                    INSERT INTO run_history
                        (job_id, total_latency_s, model_used, retry_count,
                         grounding_passed, estimated_cost_low, estimated_cost_high)
                    VALUES
                        (:job_id, :total_latency_s, :model_used, :retry_count,
                         :grounding_passed, :estimated_cost_low, :estimated_cost_high)
                    ON CONFLICT (job_id) DO NOTHING
                    """
                ),
                {
                    "job_id": job_id,
                    "total_latency_s": total_latency_s,
                    "model_used": model_used,
                    "retry_count": retry_count,
                    "grounding_passed": grounding_passed,
                    "estimated_cost_low": estimated_cost_low,
                    "estimated_cost_high": estimated_cost_high,
                },
            )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to persist run %s: %s", job_id, exc)


def get_runs(*, page: int = 1, page_size: int = 20) -> dict[str, Any]:
    """
    Return paginated run history.

    When DB is not configured returns:
        {"runs": [], "total": 0, "page": page, "page_size": page_size,
         "observability_configured": False}

    When configured returns rows ordered by timestamp descending.
    """
    engine = _get_engine()
    if engine is None:
        return {
            "runs": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "observability_configured": False,
        }

    offset = (page - 1) * page_size
    try:
        with engine.connect() as conn:
            total_row = conn.execute(
                _text("SELECT COUNT(*) FROM run_history")
            ).fetchone()
            total: int = int(total_row[0]) if total_row else 0

            rows = conn.execute(
                _text(
                    "SELECT job_id, timestamp, total_latency_s, model_used, "
                    "retry_count, grounding_passed, estimated_cost_low, "
                    "estimated_cost_high "
                    "FROM run_history "
                    "ORDER BY timestamp DESC "
                    "LIMIT :limit OFFSET :offset"
                ),
                {"limit": page_size, "offset": offset},
            ).fetchall()

        runs = [
            {
                "job_id": r[0],
                "timestamp": str(r[1]),
                "total_latency_s": r[2],
                "model_used": r[3],
                "retry_count": r[4],
                "grounding_passed": bool(r[5]) if r[5] is not None else None,
                "estimated_cost_low": r[6],
                "estimated_cost_high": r[7],
            }
            for r in rows
        ]
        return {
            "runs": runs,
            "total": total,
            "page": page,
            "page_size": page_size,
            "observability_configured": True,
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to fetch run history: %s", exc)
        return {
            "runs": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "observability_configured": True,
            "error": str(exc),
        }


def reset_engine_for_testing() -> None:
    """
    Reset module-level engine state.  Only for use in tests that need to
    inject a fresh DATABASE_URL mid-session via monkeypatching.
    """
    global _engine, _initialized
    _engine = None
    _initialized = False
