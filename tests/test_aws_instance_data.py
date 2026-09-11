"""Standalone tests for EC2 instance-data parsing helpers."""

from __future__ import annotations

from app.tools.aws_instance_data import _item_to_candidate, _extract_hourly_price


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


def test_item_to_candidate_extracts_ec2_linux_ondemand_price():
    """EC2 pricing is a string under pricing[region]['linux']['ondemand']."""
    candidate = _item_to_candidate(
        {
            "instanceType": "m5.large",
            "vCPU": 2,
            "memory": 8,
            "pricing": {
                "us-east-1": {
                    "linux": {"ondemand": "0.096"},
                    "rhel": {"ondemand": "0.109"},
                }
            },
        }
    )
    assert candidate is not None
    assert candidate.hourly_price_usd == 0.096


def test_item_to_candidate_missing_pricing_yields_none():
    candidate = _item_to_candidate({"instanceType": "m5.large", "vCPU": 2, "memory": 8})
    assert candidate is not None
    assert candidate.hourly_price_usd is None


def test_item_to_candidate_missing_region_yields_none():
    candidate = _item_to_candidate(
        {
            "instanceType": "t3.medium",
            "vCPU": 2,
            "memory": 4,
            "pricing": {"eu-west-1": {"linux": {"ondemand": "0.05"}}},
        }
    )
    assert candidate.hourly_price_usd is None


def test_extract_hourly_price_handles_string_and_float():
    pricing = {"us-east-1": {"linux": {"ondemand": "0.204"}}}
    assert _extract_hourly_price(pricing, "us-east-1", "linux") == 0.204

    pricing2 = {"us-east-1": {"linux": {"ondemand": 0.204}}}
    assert _extract_hourly_price(pricing2, "us-east-1", "linux") == 0.204


def test_extract_hourly_price_returns_none_for_missing_or_invalid():
    assert _extract_hourly_price({}, "us-east-1", "linux") is None
    assert _extract_hourly_price({"us-east-1": {}}, "us-east-1", "linux") is None
    assert _extract_hourly_price({"us-east-1": {"linux": {}}}, "us-east-1", "linux") is None
    assert _extract_hourly_price({"us-east-1": {"linux": {"ondemand": "not-a-number"}}}, "us-east-1", "linux") is None
    assert _extract_hourly_price({"us-east-1": {"linux": {"ondemand": 0}}}, "us-east-1", "linux") is None
    assert _extract_hourly_price({"us-east-1": {"linux": {"ondemand": -1}}}, "us-east-1", "linux") is None


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


# ---------------------------------------------------------------------------
# RDS and Cache pricing extraction (fetch_*_instance_data with mocked _get_json)
# ---------------------------------------------------------------------------


from app.tools.aws_instance_data import (
    _extract_rds_hourly_price,
    fetch_rds_instance_data,
    fetch_cache_instance_data,
)


def _ec2_item(instance_type, price_str, **extra):
    item = {"instanceType": instance_type, "vCPU": 2, "memory": 8}
    if price_str is not None:
        item["pricing"] = {"us-east-1": {"linux": {"ondemand": price_str}}}
    item.update(extra)
    return item


def _rds_item(instance_type, engine_key, price_val):
    return {
        "instanceType": instance_type,
        "instance_type": instance_type,
        "vcpu": 2,
        "memory": 8,
        "family": "General purpose",
        "networkPerformance": "Up to 10 Gbps",
        "pricing": {
            "us-east-1": {
                engine_key: {"ondemand": price_val, "reserved": {}},
                "other_engine": {"ondemand": 0.999, "reserved": {}},
            }
        },
    }


def _cache_item(instance_type, engine, price_val):
    return {
        "instanceType": instance_type,
        "instance_type": instance_type,
        "cacheEngine": engine,
        "vcpu": 2,
        "memory": 6.4,
        "family": "Standard",
        "max_clients": "2000",
        "pricing": {
            "us-east-1": {
                engine: {"ondemand": price_val, "reserved": {}},
            }
        },
    }


@patch("app.tools.aws_instance_data._get_json")
def test_fetch_rds_extracts_hourly_price(mock_get_json):
    mock_get_json.return_value = [_rds_item("db.t3.medium", "PostgreSQL", 0.245)]
    candidates = fetch_rds_instance_data()
    assert candidates[0].hourly_price_usd == 0.245


@patch("app.tools.aws_instance_data._get_json")
def test_fetch_rds_falls_back_to_first_available_engine(mock_get_json):
    """If PostgreSQL isn't in pricing, should try MySQL then any key."""
    mock_get_json.return_value = [_rds_item("db.m1.large", "MySQL", 0.23)]
    candidates = fetch_rds_instance_data()
    assert candidates[0].hourly_price_usd == 0.23


@patch("app.tools.aws_instance_data._get_json")
def test_fetch_rds_missing_pricing_yields_none(mock_get_json):
    mock_get_json.return_value = [{"instanceType": "db.t3.medium", "instance_type": "db.t3.medium",
                                   "vcpu": 2, "memory": 8, "family": "General purpose"}]
    candidates = fetch_rds_instance_data()
    assert candidates[0].hourly_price_usd is None


@patch("app.tools.aws_instance_data._get_json")
def test_fetch_cache_extracts_hourly_price_by_engine(mock_get_json):
    from app.models.schemas import CacheEngine
    mock_get_json.return_value = [_cache_item("cache.r5.large", "Redis", 0.173)]
    candidates = fetch_cache_instance_data()
    assert candidates[0].hourly_price_usd == 0.173


@patch("app.tools.aws_instance_data._get_json")
def test_fetch_cache_missing_pricing_yields_none(mock_get_json):
    mock_get_json.return_value = [{"instanceType": "cache.r5.large", "instance_type": "cache.r5.large",
                                   "cacheEngine": "Redis", "vcpu": 2, "memory": 13.07,
                                   "family": "Memory optimized", "max_clients": "2000"}]
    candidates = fetch_cache_instance_data()
    assert candidates[0].hourly_price_usd is None


def test_extract_rds_hourly_price_picks_postgresql_then_mysql():
    pricing = {"us-east-1": {"PostgreSQL": {"ondemand": 0.245, "reserved": {}}}}
    assert _extract_rds_hourly_price(pricing) == 0.245

    pricing2 = {"us-east-1": {"MySQL": {"ondemand": 0.23, "reserved": {}}}}
    assert _extract_rds_hourly_price(pricing2) == 0.23


def test_extract_rds_hourly_price_falls_back_to_first_numeric():
    """When no canonical engine key matches, should try any ondemand value."""
    pricing = {"us-east-1": {"14": {"ondemand": 0.29, "reserved": {}}}}
    assert _extract_rds_hourly_price(pricing) == 0.29


def test_extract_rds_hourly_price_returns_none_for_missing():
    assert _extract_rds_hourly_price({}) is None
    assert _extract_rds_hourly_price({"us-east-1": {}}) is None

