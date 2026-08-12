"""
Unit tests for instance-researcher filtering helpers.

Uses synthetic candidates only — no live Vantage calls.
"""

from __future__ import annotations

from unittest.mock import patch

from app.agent.nodes.instance_researcher import (
    _family_matches,
    _filter_by_profile,
    _narrow_by_size,
    _select_diverse,
    research_instances,
)
from app.models.schemas import (
    InstanceCandidate,
    ResourceProfile,
    TechnicalNeeds,
    TrafficPattern,
)


def _needs(**kwargs) -> TechnicalNeeds:
    base = dict(
        estimated_concurrency=100,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical / fixed size for steady load",
        reasoning="test",
    )
    base.update(kwargs)
    return TechnicalNeeds(**base)


def _candidate(
    instance_type: str,
    vcpu: int,
    memory_gib: float,
    gpu_count: int = 0,
) -> InstanceCandidate:
    return InstanceCandidate(
        instance_type=instance_type,
        vcpu=vcpu,
        memory_gib=memory_gib,
        gpu_count=gpu_count,
        gpu_model="NVIDIA A10G" if gpu_count else None,
        gpu_memory_gib=24.0 if gpu_count else None,
    )


def test_family_matches_real_generation_suffixes():
    assert _family_matches("g4dn.xlarge", ("g4", "g5", "p4"))
    assert _family_matches("p4d.24xlarge", ("g4", "g5", "p4"))
    assert _family_matches("m6i.large", ("m5", "m6", "m7"))
    assert not _family_matches("t3.medium", ("m5", "m6", "m7"))


def test_gpu_path_uses_live_gpu_count_not_family_names():
    pool = [
        _candidate("g4dn.xlarge", 4, 16, gpu_count=1),
        _candidate("p4d.24xlarge", 96, 1152, gpu_count=8),
        _candidate("m5.large", 2, 8, gpu_count=0),
    ]
    matched = _filter_by_profile(
        pool,
        _needs(requires_gpu=True, resource_profile=ResourceProfile.GPU_BOUND),
    )
    types = {c.instance_type for c in matched}
    assert types == {"g4dn.xlarge", "p4d.24xlarge"}


def test_cpu_bound_excludes_gpu_and_matches_compute_prefixes():
    pool = [
        _candidate("c6i.large", 2, 4),
        _candidate("c7i.xlarge", 4, 8),
        _candidate("m5.large", 2, 8),
        _candidate("g5.xlarge", 4, 16, gpu_count=1),
    ]
    matched = _filter_by_profile(
        pool,
        _needs(resource_profile=ResourceProfile.CPU_BOUND),
    )
    types = {c.instance_type for c in matched}
    assert types == {"c6i.large", "c7i.xlarge"}


def test_horizontal_sizing_prefers_smaller_vcpu_band():
    pool = [
        _candidate("m5.large", 2, 8),
        _candidate("m5.xlarge", 4, 16),
        _candidate("m5.2xlarge", 8, 32),
        _candidate("m5.4xlarge", 16, 64),
        _candidate("m5.8xlarge", 32, 128),
        _candidate("m5.12xlarge", 48, 192),
        _candidate("m5.16xlarge", 64, 256),
        _candidate("m5.24xlarge", 96, 384),
    ]
    needs = _needs(
        estimated_concurrency=400,
        traffic_pattern=TrafficPattern.BURSTY,
        scaling_recommendation="horizontal with auto scaling due to bursty peaks",
    )
    narrowed = _narrow_by_size(pool, needs)
    assert narrowed
    assert max(c.vcpu for c in narrowed) < 96


def test_select_diverse_caps_candidate_count():
    pool = [_candidate(f"m5.{i}xlarge", max(i, 1) * 2, max(i, 1) * 8) for i in range(1, 80)]
    selected = _select_diverse(pool, limit=10)
    assert len(selected) <= 10
    assert len({c.instance_type for c in selected}) == len(selected)


@patch("app.agent.nodes.instance_researcher.fetch_ec2_instance_data")
def test_research_instances_returns_gpu_candidates(mock_fetch):
    mock_fetch.return_value = [
        _candidate("g4dn.xlarge", 4, 16, gpu_count=1),
        _candidate("m5.large", 2, 8, gpu_count=0),
    ]
    state = {
        "requirements": None,
        "latest_user_message": None,
        "next_question": None,
        "technical_needs": _needs(
            requires_gpu=True,
            resource_profile=ResourceProfile.GPU_BOUND,
            estimated_concurrency=20,
        ),
        "instance_candidates": None,
        "recommendation": None,
    }
    state = research_instances(state)
    assert state["instance_candidates"]
    assert all(c.gpu_count > 0 for c in state["instance_candidates"])
