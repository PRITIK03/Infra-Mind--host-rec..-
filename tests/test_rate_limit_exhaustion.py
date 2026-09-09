"""
Tests for bounded rate-limit retry logic in app/llm/client.py.

Verifies that _call_with_failover:
  - exhausts exactly MAX_RATE_LIMIT_ATTEMPTS calls (no more, no fewer)
  - raises RateLimitExhaustedError (not the raw RateLimitError)
  - uses exponential backoff between attempts
  - succeeds on the first attempt that does NOT raise RateLimitError
  - does NOT retry on non-rate-limit exceptions

Also verifies invoke_structured surfaces RateLimitExhaustedError cleanly
without swallowing it or wrapping it in StructuredOutputError.
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, call, patch

import pytest
from openai import RateLimitError

# Minimal env so config doesn't error on import
os.environ.setdefault("API_KEY", "primary-test-key")
os.environ.setdefault("BASE_URL", "https://example.com/v1")
os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("VANTAGE_API_KEY", "vantage-test-key")

from app.llm.client import (
    MAX_RATE_LIMIT_ATTEMPTS,
    BACKOFF_BASE_S,
    BACKOFF_MAX_S,
    RateLimitExhaustedError,
    StructuredOutputError,
    _call_with_failover,
    get_chat_model,
    invoke_structured,
)
from app.models.schemas import UserRequirements


def _make_rate_limit_error() -> RateLimitError:
    """Construct a minimal RateLimitError (requires a response mock)."""
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    response.json.return_value = {"error": {"message": "rate limited"}}
    body = {"error": {"message": "rate limited", "type": "requests", "code": "rate_limit_exceeded"}}
    return RateLimitError(message="rate limited", response=response, body=body)


# ---------------------------------------------------------------------------
# _call_with_failover — attempt counting
# ---------------------------------------------------------------------------


def test_call_with_failover_exhausts_exactly_max_attempts_single_key(monkeypatch):
    """With one key, all MAX_RATE_LIMIT_ATTEMPTS calls use the primary model."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    call_count = 0

    def _always_raises(model):
        nonlocal call_count
        call_count += 1
        raise _make_rate_limit_error()

    # Patch sleep so the test runs instantly
    with patch("app.llm.client.time.sleep"):
        with pytest.raises(RateLimitExhaustedError) as exc_info:
            _call_with_failover(_always_raises)

    assert call_count == MAX_RATE_LIMIT_ATTEMPTS, (
        f"Expected exactly {MAX_RATE_LIMIT_ATTEMPTS} attempts, got {call_count}"
    )
    assert "try again" in str(exc_info.value).lower()


def test_call_with_failover_exhausts_exactly_max_attempts_two_keys(monkeypatch):
    """With two keys, attempts alternate primary/secondary, still exactly MAX total."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.setenv("API_KEY_2", "secondary-key")
    get_chat_model.cache_clear()

    used_secondary: list[bool] = []

    def _track_and_raise(model):
        # Detect which model is being used by its api_key attribute
        used_secondary.append(model.openai_api_key.get_secret_value() == "secondary-key")
        raise _make_rate_limit_error()

    with patch("app.llm.client.time.sleep"):
        with pytest.raises(RateLimitExhaustedError):
            _call_with_failover(_track_and_raise)

    assert len(used_secondary) == MAX_RATE_LIMIT_ATTEMPTS
    # Even attempts (0, 2) → primary; odd (1, 3) → secondary
    assert used_secondary[0] is False   # attempt 0 — primary
    assert used_secondary[1] is True    # attempt 1 — secondary
    if MAX_RATE_LIMIT_ATTEMPTS > 2:
        assert used_secondary[2] is False  # attempt 2 — primary
    if MAX_RATE_LIMIT_ATTEMPTS > 3:
        assert used_secondary[3] is True   # attempt 3 — secondary


def test_call_with_failover_raises_RateLimitExhaustedError_not_raw_429(monkeypatch):
    """The caller sees RateLimitExhaustedError, not the raw openai.RateLimitError."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    with patch("app.llm.client.time.sleep"):
        with pytest.raises(RateLimitExhaustedError):
            _call_with_failover(lambda _: (_ for _ in ()).throw(_make_rate_limit_error()))

    # Confirm raw RateLimitError is NOT what bubbles up
    with patch("app.llm.client.time.sleep"):
        try:
            _call_with_failover(lambda _: (_ for _ in ()).throw(_make_rate_limit_error()))
        except RateLimitExhaustedError:
            pass  # correct
        except RateLimitError:
            pytest.fail("Raw RateLimitError escaped — should have been wrapped in RateLimitExhaustedError")


def test_call_with_failover_succeeds_on_first_attempt(monkeypatch):
    """If the first call succeeds, no retries happen and the result is returned."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    call_count = 0

    def _succeeds(model):
        nonlocal call_count
        call_count += 1
        return "ok"

    with patch("app.llm.client.time.sleep") as mock_sleep:
        result = _call_with_failover(_succeeds)

    assert result == "ok"
    assert call_count == 1
    mock_sleep.assert_not_called()  # no backoff on first-attempt success


def test_call_with_failover_succeeds_on_second_attempt(monkeypatch):
    """Fails once then succeeds — returns result, sleeps exactly once."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    attempt = 0

    def _fail_once(model):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise _make_rate_limit_error()
        return "recovered"

    with patch("app.llm.client.time.sleep") as mock_sleep:
        result = _call_with_failover(_fail_once)

    assert result == "recovered"
    assert attempt == 2
    # Exactly one sleep before the second attempt
    assert mock_sleep.call_count == 1
    slept = mock_sleep.call_args[0][0]
    # First backoff: BACKOFF_BASE_S * 2^0 = BACKOFF_BASE_S, capped at BACKOFF_MAX_S
    expected = min(BACKOFF_BASE_S * (2 ** 0), BACKOFF_MAX_S)
    assert slept == expected, f"Expected sleep {expected}s, got {slept}s"


def test_call_with_failover_does_not_retry_non_rate_limit_error(monkeypatch):
    """A ValueError (or any non-RateLimitError) propagates immediately, no retries."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    call_count = 0

    def _raises_value_error(model):
        nonlocal call_count
        call_count += 1
        raise ValueError("something else went wrong")

    with patch("app.llm.client.time.sleep") as mock_sleep:
        with pytest.raises(ValueError, match="something else went wrong"):
            _call_with_failover(_raises_value_error)

    assert call_count == 1       # no retries
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Backoff timing
# ---------------------------------------------------------------------------


def test_call_with_failover_uses_correct_exponential_backoff(monkeypatch):
    """Sleep durations follow BACKOFF_BASE_S * 2^(attempt-1), capped at BACKOFF_MAX_S."""
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    with patch("app.llm.client.time.sleep") as mock_sleep:
        with pytest.raises(RateLimitExhaustedError):
            _call_with_failover(lambda _: (_ for _ in ()).throw(_make_rate_limit_error()))

    sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
    # attempt 0 — no sleep; attempts 1..N-1 each have a sleep
    assert len(sleep_calls) == MAX_RATE_LIMIT_ATTEMPTS - 1

    for i, slept in enumerate(sleep_calls):
        expected = min(BACKOFF_BASE_S * (2 ** i), BACKOFF_MAX_S)
        assert slept == expected, (
            f"Attempt {i + 1}: expected sleep {expected}s, got {slept}s"
        )


# ---------------------------------------------------------------------------
# invoke_structured surfaces RateLimitExhaustedError cleanly
# ---------------------------------------------------------------------------


def test_invoke_structured_surfaces_rate_limit_exhausted_not_structured_error(monkeypatch):
    """
    When _call_with_failover raises RateLimitExhaustedError, invoke_structured
    must re-raise it as-is, NOT wrap it in StructuredOutputError.
    """
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    exhausted = RateLimitExhaustedError(
        "Rate limited after multiple attempts — please try again in a few minutes."
    )

    with patch("app.llm.client._call_with_failover", side_effect=exhausted):
        with pytest.raises(RateLimitExhaustedError) as exc_info:
            invoke_structured(UserRequirements, "describe my app")

    assert "try again" in str(exc_info.value).lower()
    # Must NOT be wrapped
    assert not isinstance(exc_info.value, StructuredOutputError)


def test_invoke_structured_does_not_retry_on_rate_limit_exhausted(monkeypatch):
    """
    RateLimitExhaustedError must propagate without triggering the
    parse-error retry path inside invoke_structured.
    """
    monkeypatch.setenv("API_KEY", "primary-key")
    monkeypatch.delenv("API_KEY_2", raising=False)
    get_chat_model.cache_clear()

    exhausted = RateLimitExhaustedError("Rate limited.")
    failover_call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal failover_call_count
        failover_call_count += 1
        raise exhausted

    with patch("app.llm.client._call_with_failover", side_effect=_side_effect):
        with pytest.raises(RateLimitExhaustedError):
            invoke_structured(UserRequirements, "describe my app")

    # invoke_structured calls _call_with_failover once for primary path;
    # on RateLimitExhaustedError it must stop immediately — no second attempt.
    assert failover_call_count == 1, (
        f"Expected 1 failover call (no retry on exhaustion), got {failover_call_count}"
    )
