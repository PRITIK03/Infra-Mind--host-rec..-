"""Standalone tests for EC2 instance-data parsing helpers."""

from __future__ import annotations

from app.tools.aws_instance_data import _item_to_candidate


def test_item_to_candidate_maps_gpu_fields():
    candidate = _item_to_candidate(
        {
            "instanceType": "g4dn.xlarge",
            "vCPU": 4,
            "memory": 16,
            "GPU": 1,
            "GPU_model": "NVIDIA T4",
            "GPU_memory": 16,
            "network_performance": "Up to 25 Gigabit",
        }
    )
    assert candidate is not None
    assert candidate.instance_type == "g4dn.xlarge"
    assert candidate.vcpu == 4
    assert candidate.memory_gib == 16
    assert candidate.gpu_count == 1
    assert candidate.gpu_model == "NVIDIA T4"
    assert candidate.gpu_memory_gib == 16


def test_item_to_candidate_handles_missing_gpu_as_zero():
    candidate = _item_to_candidate(
        {
            "instanceType": "m5.large",
            "vCPU": 2,
            "memory": 8,
        }
    )
    assert candidate is not None
    assert candidate.gpu_count == 0
    assert candidate.gpu_model is None


def test_item_to_candidate_returns_none_without_instance_type():
    assert _item_to_candidate({"vCPU": 2, "memory": 8}) is None


from unittest.mock import MagicMock, call, patch
import pytest
import httpx

from app.config import ConfigError
from app.tools.aws_instance_data import (
    InstanceDataUnavailableError,
    _get_json,
)


@patch("app.tools.aws_instance_data.get_vantage_settings")
@patch("httpx.get")
def test_get_json_success_on_first_attempt(mock_httpx_get, mock_settings):
    mock_settings.return_value.api_key = "test-key"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"instanceType": "m5.large", "vCPU": 2, "memory": 8}]
    mock_httpx_get.return_value = mock_resp

    data = _get_json("https://instances.vantage.sh/instances.json")

    assert data == [{"instanceType": "m5.large", "vCPU": 2, "memory": 8}]
    assert mock_httpx_get.call_count == 1


@patch("app.tools.aws_instance_data.time.sleep")
@patch("app.tools.aws_instance_data.get_vantage_settings")
@patch("httpx.get")
def test_get_json_retry_success_on_second_attempt(mock_httpx_get, mock_settings, mock_sleep):
    mock_settings.return_value.api_key = "test-key"
    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.json.return_value = [{"instanceType": "c6i.xlarge", "vCPU": 4, "memory": 8}]

    transient_exc = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body (incomplete chunked read)"
    )
    mock_httpx_get.side_effect = [transient_exc, success_resp]

    data = _get_json("https://instances.vantage.sh/instances.json")

    assert data == [{"instanceType": "c6i.xlarge", "vCPU": 4, "memory": 8}]
    assert mock_httpx_get.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("app.tools.aws_instance_data.time.sleep")
@patch("app.tools.aws_instance_data.get_vantage_settings")
@patch("httpx.get")
def test_get_json_retry_success_on_third_attempt(mock_httpx_get, mock_settings, mock_sleep):
    mock_settings.return_value.api_key = "test-key"
    server_error_resp = MagicMock()
    server_error_resp.status_code = 503
    server_error_resp.text = "Service Unavailable"

    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.json.return_value = [{"instanceType": "r6i.large", "vCPU": 2, "memory": 16}]

    transport_exc = httpx.TransportError("connection reset by peer")
    mock_httpx_get.side_effect = [transport_exc, server_error_resp, success_resp]

    data = _get_json("https://instances.vantage.sh/instances.json")

    assert data == [{"instanceType": "r6i.large", "vCPU": 2, "memory": 16}]
    assert mock_httpx_get.call_count == 3
    assert mock_sleep.call_args_list == [call(1.0), call(2.0)]


@patch("app.tools.aws_instance_data.time.sleep")
@patch("app.tools.aws_instance_data.get_vantage_settings")
@patch("httpx.get")
def test_get_json_all_attempts_fail_raises_instance_data_unavailable_error(
    mock_httpx_get, mock_settings, mock_sleep
):
    mock_settings.return_value.api_key = "test-key"
    transient_exc = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body (incomplete chunked read)"
    )
    mock_httpx_get.side_effect = transient_exc

    with pytest.raises(InstanceDataUnavailableError) as exc_info:
        _get_json("https://instances.vantage.sh/instances.json")

    err_msg = str(exc_info.value)
    assert "Failed to reach https://instances.vantage.sh/instances.json after 3 attempts" in err_msg
    assert "incomplete chunked read" in err_msg
    assert mock_httpx_get.call_count == 3
    assert mock_sleep.call_args_list == [call(1.0), call(2.0)]


@patch("app.tools.aws_instance_data.time.sleep")
@patch("app.tools.aws_instance_data.get_vantage_settings")
@patch("httpx.get")
def test_get_json_non_retryable_error_does_not_retry(mock_httpx_get, mock_settings, mock_sleep):
    mock_settings.return_value.api_key = "test-key"
    unauthorized_resp = MagicMock()
    unauthorized_resp.status_code = 401
    unauthorized_resp.text = "Unauthorized"
    mock_httpx_get.return_value = unauthorized_resp

    with pytest.raises(InstanceDataUnavailableError) as exc_info:
        _get_json("https://instances.vantage.sh/instances.json")

    err_msg = str(exc_info.value)
    assert "returned 401" in err_msg
    assert mock_httpx_get.call_count == 1
    assert mock_sleep.call_count == 0


@patch("app.tools.aws_instance_data.time.sleep")
@patch("app.tools.aws_instance_data.get_vantage_settings")
@patch("httpx.get")
def test_get_json_config_error_does_not_retry(mock_httpx_get, mock_settings, mock_sleep):
    mock_settings.side_effect = ConfigError("Missing required environment variable: VANTAGE_API_KEY")

    with pytest.raises(InstanceDataUnavailableError) as exc_info:
        _get_json("https://instances.vantage.sh/instances.json")

    assert "VANTAGE_API_KEY" in str(exc_info.value)
    assert mock_httpx_get.call_count == 0
    assert mock_sleep.call_count == 0

