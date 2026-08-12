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
