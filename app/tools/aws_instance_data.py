"""
Live AWS EC2 instance-type data source.

Fetches current EC2 instance specifications (vCPU, memory, network
performance) directly from the Vantage Instances API over HTTP.

Note: bypasses the installed `instances_api_client` package's own parsing,
which has a confirmed bug — it reads the vCPU field as "vcpu" (lowercase)
while the API actually returns "vCPU", so it always parses as 0.
"""

from __future__ import annotations

import http.client
import time
from typing import Any

import httpx

from app.config import ConfigError, get_vantage_settings
from app.models.schemas import CacheCandidate, CacheEngine, DatabaseCandidate, InstanceCandidate


class InstanceDataUnavailableError(RuntimeError):
    """Raised when live EC2 instance data cannot be retrieved."""


_RETRYABLE_EXCEPTIONS = (
    httpx.TransportError,
    httpx.TimeoutException,
    httpx.DecodingError,
    httpx.RemoteProtocolError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    http.client.HTTPException,
)

# Default region for on-demand pricing lookup.
_DEFAULT_REGION = "us-east-1"


def _extract_hourly_price(pricing: Any, region: str, subkey: str) -> float | None:
    """
    Pull the ``ondemand`` hourly price from a nested Vantage ``pricing`` dict.

    Structure for EC2:  pricing[region]["linux"]["ondemand"]  (string)
    Structure for RDS:  pricing[region][<engine>]["ondemand"] (float)
    Structure for cache: pricing[region][<engine>]["ondemand"] (float)

    Returns None when the price is absent, malformed, or not a positive number.
    """
    if not isinstance(pricing, dict):
        return None
    region_data = pricing.get(region)
    if not isinstance(region_data, dict):
        return None
    sub = region_data.get(subkey)
    if not isinstance(sub, dict):
        return None
    raw = sub.get("ondemand")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.RequestError):
        return True
    return False


def _get_service_urls() -> dict[str, str]:
    """
    Return the service → JSON URL mapping from instances_api_client.

    Resolved lazily (called at fetch time, not at module import time) so
    that instances_api_client.client is always fully initialised before we
    read from it.  A module-level try/except import was the root cause of a
    production bug: if instances_api_client.client was mid-initialisation
    when aws_instance_data first loaded (import-order race during uvicorn
    startup), the ImportError fallback fired and bound a dict that only
    contained 'ec2', silently dropping rds/cache URLs for the rest of the
    process lifetime.
    """
    try:
        from instances_api_client.client import GLOBAL_SERVICES_JSON_URLS
        return dict(GLOBAL_SERVICES_JSON_URLS)
    except (ImportError, AttributeError):
        # Hard fallback — only used if the package is genuinely absent.
        # Raises clearly rather than silently dropping service URLs.
        return {
            "ec2": "https://instances.vantage.sh/instances.json",
            "rds": "https://instances.vantage.sh/rds/instances.json",
            "cache": "https://instances.vantage.sh/cache/instances.json",
        }


def _get_user_agent() -> str:
    try:
        from instances_api_client.client import USER_AGENT
        return USER_AGENT
    except (ImportError, AttributeError):
        return "instances-api-client-python"


def _auth_headers() -> dict[str, str]:
    try:
        settings = get_vantage_settings()
    except ConfigError as exc:
        raise InstanceDataUnavailableError(str(exc)) from exc
    return {"User-Agent": _get_user_agent(), "Authorization": f"Bearer {settings.api_key}"}


def _get_json(url: str, max_attempts: int = 3, backoff_base: float = 1.0) -> Any:
    headers = _auth_headers()
    last_error_msg = ""

    for attempt in range(1, max_attempts + 1):
        try:
            resp = httpx.get(url, headers=headers, timeout=30.0)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except (ValueError, httpx.DecodingError) as exc:
                    if isinstance(exc, httpx.DecodingError):
                        last_error_msg = str(exc) or "Incomplete chunked read"
                    else:
                        raise InstanceDataUnavailableError(
                            f"Invalid JSON from {url}: {exc}"
                        ) from None
            elif resp.status_code in {500, 502, 503, 504}:
                last_error_msg = f"HTTP {resp.status_code}: {resp.text}"
            else:
                raise InstanceDataUnavailableError(
                    f"{url} returned {resp.status_code}: {resp.text}"
                ) from None
        except InstanceDataUnavailableError:
            raise
        except Exception as exc:
            if _is_retryable_exception(exc):
                last_error_msg = str(exc) or exc.__class__.__name__
            else:
                raise InstanceDataUnavailableError(f"Failed to reach {url}: {exc}") from None

        if attempt < max_attempts:
            time.sleep(backoff_base * (2 ** (attempt - 1)))

    raise InstanceDataUnavailableError(
        f"Failed to reach {url} after {max_attempts} attempts: {last_error_msg}"
    ) from None


def _item_to_candidate(item: dict[str, Any]) -> InstanceCandidate | None:
    instance_type = item.get("instanceType") or item.get("instance_type")
    if not instance_type:
        return None

    raw_vcpu = item.get("vCPU", item.get("vcpu", 0))
    try:
        vcpu = int(raw_vcpu)
    except (TypeError, ValueError):
        vcpu = 0

    raw_gpu_count = item.get("GPU") or 0
    try:
        gpu_count = int(raw_gpu_count)
    except (TypeError, ValueError):
        gpu_count = 0

    return InstanceCandidate(
        instance_type=instance_type,
        vcpu=vcpu,
        memory_gib=item.get("memory") or item.get("memory_gib") or 0,
        gpu_count=gpu_count,
        gpu_model=item.get("GPU_model"),
        gpu_memory_gib=item.get("GPU_memory"),
        network_performance=item.get("network_performance") or item.get("networkPerformance"),
        hourly_price_usd=_extract_hourly_price(item.get("pricing"), _DEFAULT_REGION, "linux"),
    )


def fetch_ec2_instance_data() -> list[InstanceCandidate]:
    """Fetches the full current list of EC2 instance types. No caching yet by design."""
    url = _get_service_urls().get("ec2")
    if not url:
        raise InstanceDataUnavailableError(
            "EC2 instances URL is not configured in the live data client."
        )
    items = _get_json(url)
    if not isinstance(items, list):
        raise InstanceDataUnavailableError("Unexpected instances JSON structure.")

    candidates = [c for c in (_item_to_candidate(i) for i in items if isinstance(i, dict)) if c]
    if not candidates:
        raise InstanceDataUnavailableError("Live EC2 instance data source returned no results.")
    return candidates


def fetch_rds_instance_data() -> list[DatabaseCandidate]:
    """Fetches the current list of RDS database instance classes."""
    url = _get_service_urls().get("rds")
    if not url:
        raise InstanceDataUnavailableError(
            "RDS instances URL is not configured in the live data client."
        )
    items = _get_json(url)
    if not isinstance(items, list):
        raise InstanceDataUnavailableError("Unexpected RDS instances JSON structure.")

    candidates: list[DatabaseCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        instance_type = item.get("instance_type") or item.get("instanceType")
        if not instance_type:
            continue
        try:
            vcpu = int(item.get("vcpu", 0))
        except (TypeError, ValueError):
            vcpu = 0
        try:
            memory_gib = float(item.get("memory", 0))
        except (TypeError, ValueError):
            memory_gib = 0.0
        candidates.append(
            DatabaseCandidate(
                instance_type=instance_type,
                family=item.get("family", "unknown"),
                vcpu=vcpu,
                memory_gib=memory_gib,
                network_performance=(
                    item.get("network_performance")
                    or item.get("networkPerformance")
                    or "unknown"
                ),
                hourly_price_usd=_extract_rds_hourly_price(item.get("pricing")),
            )
        )

    if not candidates:
        raise InstanceDataUnavailableError("Live RDS instance data source returned no results.")
    return candidates


def _extract_rds_hourly_price(pricing: Any) -> float | None:
    """
    RDS pricing is nested by engine key (e.g. 'PostgreSQL', 'MySQL').
    Try canonical engine names in priority order, then fall back to the
    first numeric 'ondemand' value found.  Region is fixed to us-east-1.
    """
    # Priority order for engine lookups
    engine_keys = ("PostgreSQL", "MySQL", "MariaDB", "oracle-ee", "oracle-se2")
    price = None
    region_data = None
    if isinstance(pricing, dict):
        region_data = pricing.get(_DEFAULT_REGION)
    if isinstance(region_data, dict):
        for ek in engine_keys:
            price = _extract_hourly_price(pricing, _DEFAULT_REGION, ek)
            if price is not None:
                return price
        # Fallback: any key with a numeric ondemand under us-east-1
        for subkey, subval in region_data.items():
            if isinstance(subval, dict) and "ondemand" in subval:
                price = _extract_hourly_price(pricing, _DEFAULT_REGION, subkey)
                if price is not None:
                    return price
    return None


def fetch_cache_instance_data() -> list[CacheCandidate]:
    """Fetches ElastiCache node types — one entry per node-type/engine combination."""
    url = _get_service_urls().get("cache")
    if not url:
        raise InstanceDataUnavailableError(
            "ElastiCache instances URL is not configured in the live data client."
        )
    items = _get_json(url)
    if not isinstance(items, list):
        raise InstanceDataUnavailableError("Unexpected ElastiCache instances JSON structure.")

    candidates: list[CacheCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        instance_type = item.get("instance_type") or item.get("instanceType")
        engine_raw = item.get("cacheEngine")
        if not instance_type or engine_raw not in ("Memcached", "Redis", "Valkey"):
            continue
        try:
            vcpu = int(item.get("vcpu", 0))
        except (TypeError, ValueError):
            vcpu = 0
        try:
            memory_gib = float(item.get("memory", 0))
        except (TypeError, ValueError):
            memory_gib = 0.0
        max_clients_raw = item.get("max_clients")
        try:
            max_clients = int(max_clients_raw) if max_clients_raw is not None else None
        except (TypeError, ValueError):
            max_clients = None
        candidates.append(
            CacheCandidate(
                instance_type=instance_type,
                family=item.get("family", "unknown"),
                engine=CacheEngine(engine_raw),
                vcpu=vcpu,
                memory_gib=memory_gib,
                network_performance=(
                    item.get("network_performance")
                    or item.get("networkPerformance")
                    or "unknown"
                ),
                max_clients=max_clients,
                hourly_price_usd=_extract_hourly_price(
                    item.get("pricing"), _DEFAULT_REGION, engine_raw
                ),
            )
        )

    if not candidates:
        raise InstanceDataUnavailableError("Live cache instance data source returned no results.")
    return candidates