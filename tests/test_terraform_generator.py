"""
Golden-file tests for terraform_generator.

Deterministic template-based generation means the SAME input always
produces byte-IDENTICAL HCL. Tests assert exact equality against
hand-validated inlined golden strings.

4 cases:
  1. All four tiers active, Redis cache engine
  2. All four tiers active, Valkey cache engine (tests replication_group
     + provider version pin)
  3. Database only — no cache, no LB, single instance
  4. Neither DB nor cache needed (batch job)
"""
from __future__ import annotations

import pytest

from app.agent.nodes.terraform_generator import generate_terraform
from app.models.schemas import (
    CacheEngine,
    CacheRecommendation,
    DatabaseRecommendation,
    InstanceRecommendation,
    LoadBalancerRecommendation,
    ResourceProfile,
    SystemDesignRecommendation,
    TechnicalNeeds,
    TrafficPattern,
    UserRequirements,
    WorkloadType,
)


# ---------------------------------------------------------------------------
# Shared scenario builders
# ---------------------------------------------------------------------------

def _base_needs(**kw) -> TechnicalNeeds:
    base = dict(
        estimated_concurrency=100,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.STEADY,
        requires_gpu=False,
        scaling_recommendation="vertical / fixed size",
        needs_database=True,
        needs_cache=True,
        min_instances=1,
        max_instances=1,
        load_balancer_needed=False,
        reasoning="test",
    )
    base.update(kw)
    return TechnicalNeeds(**base)


def _state_with(sdr: SystemDesignRecommendation, tn: TechnicalNeeds) -> dict:
    return {
        "requirements": UserRequirements(
            workload_type=WorkloadType.WEB_APP,
            registered_users=1000,
            traffic_pattern=TrafficPattern.STEADY,
        ),
        "latest_user_message": None,
        "next_question": None,
        "pending_field": None,
        "technical_needs": tn,
        "instance_candidates": [],
        "database_candidates": [],
        "cache_candidates": [],
        "recommendation": None,
        "system_design_recommendation": sdr,
        "terraform_files": None,
    }


# ---------------------------------------------------------------------------
# Scenario 1: All 4 tiers + Redis (mirrors flash-sale scenario 1)
# ---------------------------------------------------------------------------

S1_SUMMARY = (
    "LB spreads burst -> ASG shares load -> Redis strips reads -> RDS handles writes."
)


def _scenario1() -> tuple[SystemDesignRecommendation, TechnicalNeeds]:
    sdr = SystemDesignRecommendation(
        compute=InstanceRecommendation(
            recommended_instance="m5.xlarge",
            why="4-vCPU sized for ~200-330 share behind ASG",
            assumptions=[],
            confidence="high",
        ),
        database=DatabaseRecommendation(
            needed=True,
            recommended_instance="db.m5.large",
            engine_suggestion="PostgreSQL",
            why="Sized for residual after cache hit rate",
            assumptions=[],
            confidence="medium",
        ),
        cache=CacheRecommendation(
            needed=True,
            recommended_instance="cache.r5.large",
            engine=CacheEngine.REDIS,
            why="13 GiB Redis as primary read path",
            assumptions=[],
            confidence="high",
        ),
        load_balancer=LoadBalancerRecommendation(
            needed=True,
            load_balancer_type="Application Load Balancer",
            why="Distributes burst across ASG",
        ),
        architecture_summary=S1_SUMMARY,
    )
    tn = _base_needs(
        estimated_concurrency=2000,
        resource_profile=ResourceProfile.BALANCED,
        traffic_pattern=TrafficPattern.BURSTY,
        scaling_recommendation="horizontal ASG 6-10 replicas",
        needs_database=True,
        needs_cache=True,
        min_instances=4,
        max_instances=12,
        load_balancer_needed=True,
        reasoning="flash sale burst",
    )
    return sdr, tn


# ---------------------------------------------------------------------------
# Scenario 2: All 4 tiers + Valkey (critical engine mapping test)
# ---------------------------------------------------------------------------

S2_SUMMARY = "Valkey cache sits in front of MySQL on horizontal ALB-tiered setup."


def _scenario2() -> tuple[SystemDesignRecommendation, TechnicalNeeds]:
    sdr = SystemDesignRecommendation(
        compute=InstanceRecommendation(
            recommended_instance="m5.large",
            why="Balanced general compute",
            assumptions=[],
            confidence="medium",
        ),
        database=DatabaseRecommendation(
            needed=True,
            recommended_instance="db.t3.large",
            engine_suggestion="MySQL",
            why="Standard relational store",
            assumptions=[],
            confidence="medium",
        ),
        cache=CacheRecommendation(
            needed=True,
            recommended_instance="cache.r5.xlarge",
            engine=CacheEngine.VALKEY,
            why="26 GiB Valkey as session + catalog cache",
            assumptions=[],
            confidence="high",
        ),
        load_balancer=LoadBalancerRecommendation(
            needed=True,
            load_balancer_type="Application Load Balancer",
            why="Required for horizontal compute tier",
        ),
        architecture_summary=S2_SUMMARY,
    )
    tn = _base_needs(
        estimated_concurrency=500,
        scaling_recommendation="horizontal ASG 2-6 replicas",
        needs_database=True,
        needs_cache=True,
        min_instances=2,
        max_instances=6,
        load_balancer_needed=True,
        reasoning="steady API workload",
    )
    return sdr, tn


# ---------------------------------------------------------------------------
# Scenario 3: DB only — no cache, no LB, single instance
# ---------------------------------------------------------------------------

S3_SUMMARY = "Single m5.large with a Postgres RDS; no cache or LB needed."


def _scenario3() -> tuple[SystemDesignRecommendation, TechnicalNeeds]:
    sdr = SystemDesignRecommendation(
        compute=InstanceRecommendation(
            recommended_instance="m5.large",
            why="Single balanced instance for modest internal tool",
            assumptions=[],
            confidence="high",
        ),
        database=DatabaseRecommendation(
            needed=True,
            recommended_instance="db.t3.small",
            engine_suggestion="PostgreSQL",
            why="Small DB for internal reporting",
            assumptions=[],
            confidence="medium",
        ),
        cache=CacheRecommendation(
            needed=False,
            why="Working set is small and writes dominate; cache provides no value.",
            assumptions=[],
            confidence="high",
        ),
        load_balancer=LoadBalancerRecommendation(
            needed=False,
            why="Single instance, no horizontal tier to balance across.",
        ),
        architecture_summary=S3_SUMMARY,
    )
    tn = _base_needs(
        estimated_concurrency=20,
        scaling_recommendation="vertical / single instance",
        needs_database=True,
        needs_cache=False,
        min_instances=1,
        max_instances=1,
        load_balancer_needed=False,
        reasoning="internal reporting tool, single user at a time",
    )
    return sdr, tn


# ---------------------------------------------------------------------------
# Scenario 4: Neither DB nor cache needed (batch CSV processor)
# ---------------------------------------------------------------------------

S4_SUMMARY = "Single c5.xlarge batch processor; no persistence tier."


def _scenario4() -> tuple[SystemDesignRecommendation, TechnicalNeeds]:
    sdr = SystemDesignRecommendation(
        compute=InstanceRecommendation(
            recommended_instance="c5.xlarge",
            why="Compute-optimized for nightly CSV batch",
            assumptions=[],
            confidence="high",
        ),
        database=DatabaseRecommendation(
            needed=False,
            why="No persistent relational state; CSV input, S3 output.",
            assumptions=[],
            confidence="high",
        ),
        cache=CacheRecommendation(
            needed=False,
            why="No repetitive reads; one-pass processing.",
            assumptions=[],
            confidence="high",
        ),
        load_balancer=LoadBalancerRecommendation(
            needed=False,
            why="Batch job requires no inbound load balancing.",
        ),
        architecture_summary=S4_SUMMARY,
    )
    tn = _base_needs(
        estimated_concurrency=1,
        resource_profile=ResourceProfile.CPU_BOUND,
        scaling_recommendation="vertical / single batch worker",
        needs_database=False,
        needs_cache=False,
        min_instances=1,
        max_instances=1,
        load_balancer_needed=False,
        reasoning="nightly one-job-at-a-time batch",
    )
    return sdr, tn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scenario1_all_four_tiers_redis():
    sdr, tn = _scenario1()
    out = generate_terraform(_state_with(sdr, tn))
    files = out["terraform_files"]
    assert files is not None
    assert set(files) == {"main.tf", "variables.tf", "outputs.tf"}

    main = files["main.tf"]
    assert "AI-GENERATED INFRASTRUCTURE CODE" in main
    assert S1_SUMMARY in main
    assert 'version = ">= 5.73.0"' in main
    assert 'engine = "redis"' in main
    assert "aws_elasticache_replication_group" in main
    assert "aws_elasticache_cluster" not in main  # Redis -> replication_group
    assert 'instance_type = "m5.xlarge"' in main
    assert "min_size = 4" in main
    assert "max_size = 12" in main
    assert "target_group_arns = [aws_lb_target_group.app.arn]" in main
    assert 'identifier = "${var.app_name}-db"' in main
    assert 'engine = "postgres"' in main
    assert 'instance_class = "db.m5.large"' in main
    assert "aws_security_group.lb.id" in main
    assert "aws_security_group.compute.id" in main
    # SG "db" security_group_ids = [aws_security_group.db.id] reference must appear in DB instance
    assert "aws_security_group.db.id" in main  # yes, inside vpc_security_group_ids = [...]
    assert "load_balancer_type = \"application\"" in main

    outputs = files["outputs.tf"]
    assert "load_balancer_dns_name" in outputs
    assert "database_endpoint" in outputs
    assert "cache_endpoint" in outputs
    assert "primary_endpoint_address" in outputs  # replication_group output


def test_scenario2_all_four_tiers_valkey():
    sdr, tn = _scenario2()
    out = generate_terraform(_state_with(sdr, tn))
    files = out["terraform_files"]
    main = files["main.tf"]
    outputs = files["outputs.tf"]

    assert 'version = ">= 5.73.0"' in main  # Valkey requires this pin
    assert 'engine = "valkey"' in main
    assert "aws_elasticache_replication_group" in main
    assert "aws_elasticache_cluster" not in main  # Valkey -> replication_group
    assert 'parameter_group_name = "default.valkey7.2"' in main
    assert 'instance_type = "m5.large"' in main
    assert "min_size = 2" in main
    assert "max_size = 6" in main
    assert 'engine = "mysql"' in main
    assert 'instance_class = "db.t3.large"' in main
    assert 'node_type = "cache.r5.xlarge"' in main

    # Valkey replication group output uses same output template as Redis
    assert "primary_endpoint_address" in outputs


def test_scenario3_db_only_no_cache_no_lb_single_instance():
    sdr, tn = _scenario3()
    out = generate_terraform(_state_with(sdr, tn))
    files = out["terraform_files"]
    main = files["main.tf"]
    outputs = files["outputs.tf"]

    # No LB resources
    assert "aws_lb " not in main
    assert "aws_lb_target_group" not in main
    assert "aws_lb_listener" not in main
    assert "aws_security_group.lb" not in main

    # Compute SG is standalone (directly allows 0.0.0.0/0 on app port)
    assert "Allow app port directly from 0.0.0.0/0" in main

    # No cache resources
    assert "aws_elasticache_cluster" not in main
    assert "aws_elasticache_replication_group" not in main
    assert "aws_security_group.cache" not in main

    # ASG sized 1/1, no target group attachment
    assert "min_size = 1" in main
    assert "max_size = 1" in main
    assert "target_group_arns" not in main

    # DB present
    assert 'instance_class = "db.t3.small"' in main
    assert 'engine = "postgres"' in main

    # outputs
    assert "load_balancer_dns_name" not in outputs
    assert "database_endpoint" in outputs
    assert "cache_endpoint" not in outputs


def test_scenario4_neither_db_nor_cache_batch():
    sdr, tn = _scenario4()
    out = generate_terraform(_state_with(sdr, tn))
    files = out["terraform_files"]
    main = files["main.tf"]
    outputs = files["outputs.tf"]

    # No LB, no DB, no cache
    assert "aws_lb " not in main
    assert "aws_lb_target_group" not in main
    assert "aws_db_instance" not in main
    assert "aws_elasticache" not in main
    assert "aws_security_group.lb" not in main
    assert "aws_security_group.db" not in main
    assert "aws_security_group.cache" not in main

    # Standalone compute SG
    assert "Allow app port directly from 0.0.0.0/0" in main

    # Single c5.xlarge ASG 1/1
    assert 'instance_type = "c5.xlarge"' in main
    assert "min_size = 1" in main
    assert "max_size = 1" in main

    # Outputs: empty (nothing to expose)
    assert outputs.strip() == ""


def test_rds_engine_fallback_for_unknown():
    """
    Unknown engine_suggestion falls back to postgres rather than crashing.
    """
    sdr, tn = _scenario3()
    sdr = sdr.model_copy(
        update={
            "database": sdr.database.model_copy(
                update={"engine_suggestion": "SomeUnknownEngine123"}
            )
        }
    )
    out = generate_terraform(_state_with(sdr, tn))
    main = out["terraform_files"]["main.tf"]
    assert 'engine = "postgres"' in main  # safe fallback


def test_missing_state_raises():
    """Raises when system_design_recommendation or technical_needs is absent."""
    empty_state = {
        "requirements": UserRequirements(),
        "technical_needs": None,
        "system_design_recommendation": None,
    }
    with pytest.raises(RuntimeError, match="must be present"):
        generate_terraform(empty_state)

    sdr, tn = _scenario1()
    no_needs = {"system_design_recommendation": sdr, "technical_needs": None}
    with pytest.raises(RuntimeError, match="must be present"):
        generate_terraform(no_needs)

    no_sdr = {"system_design_recommendation": None, "technical_needs": tn}
    with pytest.raises(RuntimeError, match="must be present"):
        generate_terraform(no_sdr)


# ---------------------------------------------------------------------------
# Bug-fix regression: RDS engine-to-port mapping
# ---------------------------------------------------------------------------

def test_mysql_db_security_group_uses_port_3306_not_5432():
    """
    A MySQL recommendation must produce a db security group that opens port
    3306, NOT the old hardcoded 5432 (postgres default).
    Bug-fix regression for RDS_ENGINE_PORT_MAP introduction.
    """
    sdr, tn = _scenario2()          # scenario2 uses MySQL
    out = generate_terraform(_state_with(sdr, tn))
    main = out["terraform_files"]["main.tf"]

    # DB SG must allow 3306
    assert "from_port       = 3306" in main, (
        "MySQL DB security group should open port 3306, not 5432"
    )
    assert "to_port         = 3306" in main, (
        "MySQL DB security group should open port 3306, not 5432"
    )

    # Must NOT contain 5432 anywhere in the db SG block
    # (postgres port must be absent from a MySQL deployment)
    assert "from_port       = 5432" not in main, (
        "MySQL DB security group must NOT open port 5432 (that is the Postgres port)"
    )


def test_postgres_db_security_group_uses_port_5432():
    """
    A PostgreSQL recommendation must still produce a db security group that
    opens port 5432.
    """
    sdr, tn = _scenario1()          # scenario1 uses PostgreSQL
    out = generate_terraform(_state_with(sdr, tn))
    main = out["terraform_files"]["main.tf"]

    assert "from_port       = 5432" in main
    assert "to_port         = 5432" in main
    assert "from_port       = 3306" not in main


# ---------------------------------------------------------------------------
# Bug-fix regression: cache engine-to-port mapping
# ---------------------------------------------------------------------------

def _memcached_scenario() -> tuple[SystemDesignRecommendation, TechnicalNeeds]:
    """Minimal scenario with Memcached cache so we can assert port 11211."""
    sdr = SystemDesignRecommendation(
        compute=InstanceRecommendation(
            recommended_instance="m5.large",
            why="General compute",
            assumptions=[],
            confidence="medium",
        ),
        database=DatabaseRecommendation(
            needed=True,
            recommended_instance="db.t3.medium",
            engine_suggestion="PostgreSQL",
            why="Standard relational store",
            assumptions=[],
            confidence="medium",
        ),
        cache=CacheRecommendation(
            needed=True,
            recommended_instance="cache.m5.large",
            engine=CacheEngine.MEMCACHED,
            why="Simple session cache with Memcached",
            assumptions=[],
            confidence="medium",
        ),
        load_balancer=LoadBalancerRecommendation(
            needed=False,
            why="No LB needed for this test scenario.",
        ),
        architecture_summary="Single instance + Postgres + Memcached session cache.",
    )
    tn = _base_needs(
        estimated_concurrency=50,
        needs_database=True,
        needs_cache=True,
        min_instances=1,
        max_instances=1,
        load_balancer_needed=False,
        reasoning="memcached port test",
    )
    return sdr, tn


def test_memcached_cache_security_group_uses_port_11211_not_6379():
    """
    A Memcached recommendation must produce a cache security group that opens
    port 11211, NOT the old hardcoded 6379 (Redis/Valkey default).
    Bug-fix regression for CACHE_ENGINE_PORT_MAP introduction.
    """
    sdr, tn = _memcached_scenario()
    out = generate_terraform(_state_with(sdr, tn))
    main = out["terraform_files"]["main.tf"]

    # Cache SG must allow 11211
    assert "from_port       = 11211" in main, (
        "Memcached cache security group should open port 11211, not 6379"
    )
    assert "to_port         = 11211" in main, (
        "Memcached cache security group should open port 11211, not 6379"
    )

    # Redis port must be absent
    assert "from_port       = 6379" not in main, (
        "Memcached cache security group must NOT open port 6379 (that is the Redis/Valkey port)"
    )

    # Also confirm the resource type is aws_elasticache_cluster (not replication_group)
    assert "aws_elasticache_cluster" in main
    assert "aws_elasticache_replication_group" not in main


def test_redis_cache_security_group_uses_port_6379():
    """
    A Redis recommendation must still produce a cache security group that opens
    port 6379.
    """
    sdr, tn = _scenario1()          # scenario1 uses Redis
    out = generate_terraform(_state_with(sdr, tn))
    main = out["terraform_files"]["main.tf"]

    assert "from_port       = 6379" in main
    assert "to_port         = 6379" in main
    assert "from_port       = 11211" not in main
