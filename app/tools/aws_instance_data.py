"""
Live AWS EC2 instance-type data source.

Fetches current EC2 instance specifications (vCPU, memory, network
performance) directly from the Vantage Instances API over HTTP.

Note: bypasses the installed `instances_api_client` package's own parsing,
which has a confirmed bug — it reads the vCPU field as "vcpu" (lowercase)
while the API actually returns "vCPU", so it always parses as 0.
"""

from __future__ import annotations

from typing import Any

import httpx
try:
    from instances_api_client.client import GLOBAL_SERVICES_JSON_URLS, USER_AGENT
except ImportError:
    USER_AGENT = "instances-api-client-python"
    GLOBAL_SERVICES_JSON_URLS = {
        "ec2": "https://instances.vantage.sh/instances.json",
    }

from app.config import ConfigError, get_vantage_settings
from app.models.schemas import InstanceCandidate


class InstanceDataUnavailableError(RuntimeError):
    """Raised when live EC2 instance data cannot be retrieved."""


def _auth_headers() -> dict[str, str]:
    try:
        settings = get_vantage_settings()
    except ConfigError as exc:
        raise InstanceDataUnavailableError(str(exc)) from exc
    return {"User-Agent": USER_AGENT, "Authorization": f"Bearer {settings.api_key}"}


def _get_json(url: str) -> Any:
    try:
        resp = httpx.get(url, headers=_auth_headers(), timeout=30.0)
    except httpx.RequestError as exc:
        raise InstanceDataUnavailableError(f"Failed to reach {url}: {exc}") from exc

    if resp.status_code != 200:
        raise InstanceDataUnavailableError(f"{url} returned {resp.status_code}: {resp.text}")

    try:
        return resp.json()
    except ValueError as exc:
        raise InstanceDataUnavailableError(f"Invalid JSON from {url}: {exc}") from exc


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
        hourly_price_usd=None,  # add pricing lookup in a later phase
    )


def fetch_ec2_instance_data() -> list[InstanceCandidate]:
    """Fetches the full current list of EC2 instance types. No caching yet by design."""
    url = GLOBAL_SERVICES_JSON_URLS.get("ec2")
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