"""
In-memory job store for long-running agent executions.

KNOWN LIMITATION (v1):
    This store is process-local and in-memory only. It will NOT:
      - survive a server restart
      - share state across multiple workers / instances / pods
      - persist after the Python process exits
    For horizontal scaling or production durability, replace this with a
    Redis-backed store (e.g. Redis Hash + Pub/Sub for progress updates)
    and swap in the same interface. Using a single asyncio.Lock protects
    against concurrent access within one single-threaded async worker.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from app.agent.state import AgentState


JobStatus = Literal[
    "collecting",
    "awaiting_input",
    "running",
    "done",
    "error",
]


@dataclass
class Job:
    """Tracks a single agent execution across its full lifecycle."""

    job_id: str
    status: JobStatus
    current_stage: str
    state: AgentState
    next_question: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    # Set while the LLM layer is retrying after a rate-limit response,
    # e.g. "Retrying after rate limit (attempt 2 of 4)".  Cleared back
    # to None when the retry succeeds so the UI can stop showing it.
    retry_info: str | None = None
    created_at: float = field(default_factory=lambda: __import__("time").time())


class JobStore:
    """Thread-safe in-memory dict of job_id -> Job."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def update_stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.current_stage = stage

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        next_question: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            if next_question is not None:
                job.next_question = next_question
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error

    def update_state(self, job_id: str, state: AgentState) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.state = state
                if state.get("next_question"):
                    job.next_question = state["next_question"]

    def update_retry_info(self, job_id: str, retry_info: str | None) -> None:
        """Set or clear the real-time retry progress message.

        Pass a non-None string while a rate-limit retry is in progress
        (e.g. "Retrying after rate limit (attempt 2 of 4)").
        Pass None to clear it once the retry succeeds.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.retry_info = retry_info


STAGE_LABELS: dict[str, str] = {
    "collect_requirements": "Collecting requirements",
    "validate_requirements": "Validating requirements",
    "reason_system_design": "Reasoning about system design",
    "research_instances": "Researching compute options",
    "research_database": "Researching database options",
    "research_cache": "Researching cache options",
    "holistic_recommend": "Building final recommendation",
    "grounding_check": "Checking recommendation consistency",
    "generate_terraform": "Generating Terraform",
}


def label_for_node(node_name: str) -> str:
    """Return a human-readable stage label for a LangGraph node name."""
    return STAGE_LABELS.get(node_name, node_name.replace("_", " ").title())
