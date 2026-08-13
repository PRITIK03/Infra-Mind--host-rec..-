"""
Centralized environment configuration for the AWS Instance Advisor agent.

Loads all required environment variables once via python-dotenv. Settings
are split into separate groups (LLM vs. live-data) so a module that only
needs one group isn't blocked by unrelated missing variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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
    return LLMSettings(
        api_key=_require("API_KEY"),
        base_url=_require("BASE_URL"),
        model_name=_require("MODEL_NAME"),
        max_tokens=_optional_int("LLM_MAX_TOKENS", 8192),
        reasoning_max_tokens=_optional_int("LLM_REASONING_MAX_TOKENS", 2048),
        api_key_secondary=os.getenv("API_KEY_2") or None,
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