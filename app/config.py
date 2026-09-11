"""
Centralized environment configuration for the AWS Instance Advisor agent.

Loads all required environment variables once via python-dotenv. Settings
are split into separate groups (LLM vs. live-data) so a module that only
needs one group isn't blocked by unrelated missing variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _optional_int(name: str, default: int) -> int:
    """Read an integer env var, returning *default* if unset or non-numeric."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMSettings:
    """Config for the OpenAI-compatible LLM endpoint."""

    api_key: str
    base_url: str
    model_name: str
    max_tokens: int = 8192
    # Caps the model's internal reasoning/"thinking" budget as a subset of
    # max_tokens (not in addition to it). Keep meaningfully smaller than
    # max_tokens so visible JSON output still has room.
    reasoning_max_tokens: int = 2048
    api_key_secondary: str | None = None
    # Optional server-side model fallback list for OpenRouter.  When set,
    # OpenRouter will automatically reroute to the next model in the list if
    # the primary is rate-limited or unavailable — distinct from the
    # dual-key failover which handles per-account 429s.  Read as a
    # comma-separated string from LLM_FALLBACK_MODELS; defaults to empty.
    fallback_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class VantageSettings:
    """Config for the live EC2 instance-data source."""

    api_key: str


@dataclass(frozen=True)
class TavilySettings:
    """Config for Tavily-powered web search."""

    api_key: str


def get_llm_settings() -> LLMSettings:
    """Loads and validates LLM-related settings only."""
    raw_fallback = os.getenv("LLM_FALLBACK_MODELS", "")
    fallback_models: tuple[str, ...] = tuple(
        m.strip() for m in raw_fallback.split(",") if m.strip()
    )
    return LLMSettings(
        api_key=_require("API_KEY"),
        base_url=_require("BASE_URL"),
        model_name=_require("MODEL_NAME"),
        max_tokens=_optional_int("LLM_MAX_TOKENS", 8192),
        reasoning_max_tokens=_optional_int("LLM_REASONING_MAX_TOKENS", 2048),
        api_key_secondary=os.getenv("API_KEY_2") or None,
        fallback_models=fallback_models,
    )


def get_vantage_settings() -> VantageSettings:
    """Loads and validates Vantage-related settings only."""
    return VantageSettings(
        api_key=_require("VANTAGE_API_KEY"),
    )


def get_tavily_settings() -> TavilySettings:
    """Loads and validates Tavily-related settings only."""
    return TavilySettings(
        api_key=_require("TAVILY_API_KEY"),
    )


@dataclass(frozen=True)
class ObservabilitySettings:
    """
    Optional observability config.

    database_url: SQLAlchemy-compatible URL for run-history persistence
    (e.g. ``postgresql+psycopg2://user:pass@host/db`` for Neon/Supabase,
    or ``sqlite:///./runs.db`` for local development).
    When None, run summaries are logged to stdout only — no DB required.
    """

    database_url: str | None = None


def get_observability_settings() -> ObservabilitySettings:
    """Load observability settings.  Gracefully returns an unconfigured
    instance when DATABASE_URL is absent — matching the Tavily skip pattern."""
    raw = (os.getenv("DATABASE_URL") or "").strip() or None
    if raw is None:
        logger.debug(
            "DATABASE_URL is unset; observability DB persistence disabled. "
            "Run summaries will be logged to stdout only."
        )
    return ObservabilitySettings(database_url=raw)


@dataclass(frozen=True)
class APISettings:
    """Config for the FastAPI HTTP layer exposing the agent."""

    cors_allowed_origin: str
    port: int = 8000


def get_api_settings() -> APISettings:
    """Load API settings without silently widening production CORS."""
    origin = (os.getenv("CORS_ALLOWED_ORIGIN") or "").strip()
    if not origin:
        logger.warning(
            "CORS_ALLOWED_ORIGIN is unset; cross-origin browser requests are disabled. "
            "Set it explicitly for each environment."
        )
    return APISettings(
        cors_allowed_origin=origin,
        port=_optional_int("PORT", 8000),
    )
