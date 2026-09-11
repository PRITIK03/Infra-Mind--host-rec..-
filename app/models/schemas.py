"""
Pydantic schemas shared across the AWS Instance Advisor agent.

Defines the structured data contracts between graph nodes:
- UserRequirements: raw information collected from the user
- TechnicalNeeds: the system design reasoner's interpretation of those requirements
- InstanceCandidate: a single EC2 instance type fetched from the live data source
- DatabaseCandidate: a single RDS instance class fetched from the live data source
- CacheCandidate: a single ElastiCache node type, per engine, fetched from the live data source
- InstanceRecommendation: the final structured EC2 recommendation
- DatabaseRecommendation: recommendation for the database tier (or explicit note that none is needed)
- CacheRecommendation: recommendation for the cache tier (validated engine + node type)
- LoadBalancerRecommendation: load balancer guidance derived from technical_needs
- SystemDesignRecommendation: full V2 recommendation combining all four tiers
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class WorkloadType(str, Enum):
    """Broad category of the application being hosted."""
    WEB_APP = "web_app"
    API_SERVICE = "api_service"
    BATCH_PROCESSING = "batch_processing"
    ML_INFERENCE = "ml_inference"
    OTHER = "other"


class TrafficPattern(str, Enum):
    """Shape of expected traffic over time."""
    STEADY = "steady"
    BURSTY = "bursty"
    UNKNOWN = "unknown"


class ResourceProfile(str, Enum):
    """Dominant resource bottleneck of the workload."""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    GPU_BOUND = "gpu_bound"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


class UserRequirements(BaseModel):
    """
    Raw requirement information gathered from the user during the
    requirement-collection step. Optional fields get filled in
    incrementally as requirement_validator asks follow-up questions.
    """

    workload_type: Optional[WorkloadType] = Field(
        default=None,
        description="Kind of application: web app, API, batch job, ML inference, etc.",
    )
    registered_users: Optional[int] = Field(
        default=None, ge=0,
        description="Total registered/expected customers — NOT concurrent users.",
    )
    estimated_concurrent_users: Optional[int] = Field(
        default=None, ge=0,
        description="Estimated users active at the same time, if known.",
    )
    requests_per_second: Optional[float] = Field(
        default=None, ge=0,
        description="Estimated peak requests per second, if known.",
    )
    traffic_pattern: TrafficPattern = Field(
        default=TrafficPattern.UNKNOWN,
        description="Whether traffic is steady or bursty with peak windows.",
    )
    peak_hours: Optional[str] = Field(
        default=None,
        description="Description of peak traffic window, e.g. '5 PM to 8 PM daily'.",
    )
    resource_profile: ResourceProfile = Field(
        default=ResourceProfile.UNKNOWN,
        description="Whether the workload is CPU-bound, memory-bound, or balanced, if known.",
    )
    gpu_required: Optional[bool] = Field(
        default=None,
        description="Whether the workload needs GPU acceleration (e.g. local ML inference).",
    )
    latency_requirement_ms: Optional[int] = Field(
        default=None, ge=0,
        description="Max acceptable response latency in ms, if specified.",
    )
    budget_constraint_usd_monthly: Optional[float] = Field(
        default=None, ge=0,
        description="Monthly budget ceiling in USD, if specified.",
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any other free-text context provided by the user.",
    )

    def missing_critical_fields(self) -> list[str]:
        """
        Returns field names still needed before the reasoner can proceed.
        Used by requirement_validator to decide: ask a follow-up, or continue.
        """
        critical: list[str] = []
        if self.workload_type is None:
            critical.append("workload_type")
        if (
            self.registered_users is None
            and self.estimated_concurrent_users is None
            and self.requests_per_second is None
        ):
            critical.append("expected_scale")
        if self.traffic_pattern == TrafficPattern.UNKNOWN:
            critical.append("traffic_pattern")
        return critical


class TechnicalNeeds(BaseModel):
    """
    Output of system_design_reasoner: raw requirements translated into
    technical characteristics matchable against real EC2 instance data.
    """

    estimated_concurrency: int = Field(
        ..., ge=0,
        description="Reasoner's best estimate of concurrent load, derived from users/RPS/stated concurrency.",
    )
    resource_profile: ResourceProfile = Field(
        default=ResourceProfile.UNKNOWN,
        description="Dominant resource bottleneck; UNKNOWN if the model omitted it.",
    )
    traffic_pattern: TrafficPattern = Field(
        default=TrafficPattern.UNKNOWN,
        description="Traffic shape; UNKNOWN if the model omitted it.",
    )
    requires_gpu: bool = False
    scaling_recommendation: str = Field(
        ...,
        description="Vertical vs horizontal scaling view, e.g. 'horizontal with auto scaling due to bursty peak window'.",
    )
    needs_database: bool = Field(
        default=True,
        description="Whether this workload needs a persistent relational database.",
    )
    needs_cache: bool = Field(
        default=False,
        description="Whether a caching layer would meaningfully help this workload.",
    )
    min_instances: int = Field(
        default=1, ge=1,
        description="Minimum number of compute instances recommended.",
    )
    max_instances: int = Field(
        default=1, ge=1,
        description="Maximum number of compute instances recommended to handle peak load.",
    )
    load_balancer_needed: bool = Field(
        default=False,
        description="Whether a load balancer is warranted — generally true when max_instances > 1.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of how the above was derived from the raw requirements.",
    )


class InstanceCandidate(BaseModel):
    """A single EC2 instance type as fetched from the live data source."""

    instance_type: str = Field(..., description="e.g. 't3.medium', 'm5.large'")
    vcpu: int
    memory_gib: float
    gpu_count: int = Field(default=0, ge=0, description="Number of GPUs, 0 if none.")
    gpu_model: Optional[str] = Field(default=None, description="GPU model, e.g. 'NVIDIA A10G'.")
    gpu_memory_gib: Optional[float] = Field(
        default=None,
        description="Total GPU memory across all GPUs on the instance, in GiB.",
    )
    network_performance: Optional[str] = None
    hourly_price_usd: Optional[float] = Field(
        default=None, description="On-demand hourly price, if available from the source."
    )


class InstanceRecommendation(BaseModel):
    """Final structured output returned to the user."""

    recommended_instance: str = Field(..., description="The recommended EC2 instance type.")
    why: str = Field(..., description="Reasoning for why this instance fits the workload.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made where information was estimated rather than confirmed.",
    )
    confidence: str = Field(..., description="Confidence level: 'low', 'medium', or 'high'.")
    alternative_instance: Optional[str] = Field(
        default=None, description="A viable alternative instance type, if applicable."
    )
    trade_off: Optional[str] = Field(
        default=None, description="Cost/performance/complexity trade-off vs the alternative."
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v.lower() not in allowed:
            raise ValueError(f"confidence must be one of {allowed}")
        return v.lower()


class CacheEngine(str, Enum):
    MEMCACHED = "Memcached"
    REDIS = "Redis"
    VALKEY = "Valkey"


class DatabaseCandidate(BaseModel):
    """A single RDS database instance class as fetched from the live data source."""

    instance_type: str = Field(..., description="e.g. 'db.t3.medium', 'db.r5.large'")
    family: str = Field(
        ...,
        description="'General purpose', 'Memory optimized', or 'Micro instances'.",
    )
    vcpu: int
    memory_gib: float
    network_performance: str = Field(
        default="unknown",
        description=(
            "Raw descriptive network performance — values are inconsistent free text, "
            "not parseable numerics."
        ),
    )
    hourly_price_usd: Optional[float] = Field(
        default=None,
        description="On-demand hourly price for us-east-1, if available from the source.",
    )


class CacheCandidate(BaseModel):
    """
    A single ElastiCache node type, specific to one engine.
    Node types are not universal across engines — a given type may only
    support Memcached, Redis, or Valkey, never all three.
    """

    instance_type: str = Field(..., description="e.g. 'cache.t3.medium', 'cache.r6g.large'")
    family: str = Field(
        ...,
        description="'Standard', 'Memory optimized', or 'Network optimized'.",
    )
    engine: CacheEngine
    vcpu: int
    memory_gib: float
    network_performance: str = "unknown"
    max_clients: Optional[int] = None
    hourly_price_usd: Optional[float] = Field(
        default=None,
        description="On-demand hourly price for us-east-1, if available from the source.",
    )


class DatabaseRecommendation(BaseModel):
    """Recommendation for the database tier, or an explicit note that none is needed."""

    needed: bool
    recommended_instance: Optional[str] = None
    engine_suggestion: Optional[str] = Field(
        default=None,
        description=(
            "Suggested DB engine (e.g. 'PostgreSQL', 'MySQL') — advisory reasoning, "
            "not validated against a fixed list, since RDS instance types generally "
            "support multiple engines."
        ),
    )
    why: str
    assumptions: list[str] = Field(default_factory=list)
    confidence: str
    alternative_instance: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v.lower() not in allowed:
            raise ValueError(f"confidence must be one of {allowed}")
        return v.lower()


class CacheRecommendation(BaseModel):
    """Recommendation for the cache tier, or an explicit note that none is needed."""

    needed: bool
    recommended_instance: Optional[str] = None
    engine: Optional[CacheEngine] = Field(
        default=None,
        description=(
            "Must match an actual (instance_type, engine) pair in the live candidate "
            "set — validated, not just advisory, since cache node types are "
            "engine-specific."
        ),
    )
    why: str
    assumptions: list[str] = Field(default_factory=list)
    confidence: str
    alternative_instance: Optional[str] = None
    alternative_engine: Optional[CacheEngine] = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v.lower() not in allowed:
            raise ValueError(f"confidence must be one of {allowed}")
        return v.lower()


class LoadBalancerRecommendation(BaseModel):
    """
    Load balancer guidance — derived from technical_needs rather than
    live-fetched, since there is no 'instance type' for a load balancer.
    """

    needed: bool
    load_balancer_type: Optional[str] = Field(
        default=None,
        description=(
            "e.g. 'Application Load Balancer', 'Network Load Balancer' "
            "— omitted if not needed."
        ),
    )
    why: str


class EstimatedCost(BaseModel):
    """
    Estimated monthly on-demand cost for the recommended architecture.

    All fields are Optional[float]: None means pricing data was unavailable
    for that tier rather than a fabricated zero.
    """

    compute_monthly_low: Optional[float] = Field(
        default=None,
        description="Monthly cost at min_instances (hourly_price × min × 730h).",
    )
    compute_monthly_high: Optional[float] = Field(
        default=None,
        description="Monthly cost at max_instances (hourly_price × max × 730h).",
    )
    database_monthly: Optional[float] = Field(
        default=None,
        description="Monthly cost for the recommended RDS instance (× 730h), or None if DB not needed / pricing unavailable.",
    )
    cache_monthly: Optional[float] = Field(
        default=None,
        description="Monthly cost for the recommended cache node (× 730h), or None if cache not needed / pricing unavailable.",
    )
    total_monthly_low: Optional[float] = Field(
        default=None,
        description="Sum of compute low + database + cache (any None → None).",
    )
    total_monthly_high: Optional[float] = Field(
        default=None,
        description="Sum of compute high + database + cache (any None → None).",
    )


class SystemDesignRecommendation(BaseModel):
    """
    The full V2 recommendation: compute + database + cache + load balancing,
    reasoned about together as a coherent architecture.
    """

    compute: InstanceRecommendation
    database: DatabaseRecommendation
    cache: CacheRecommendation
    load_balancer: LoadBalancerRecommendation
    architecture_summary: str = Field(
        ...,
        description=(
            "Short summary of how the pieces work together — e.g. how the cache "
            "reduces database load, why this instance count."
        ),
    )
    estimated_cost: Optional[EstimatedCost] = Field(
        default=None,
        description=(
            "Estimated monthly on-demand cost for the recommended architecture. "
            "None when pricing data is unavailable."
        ),
    )
    # Grounding check results — populated by grounding_check node.
    # grounding_passed=None means the check has not run yet (pre-grounding).
    # grounding_passed=False means the check ran and found unresolved issues
    # after one bounded retry — the recommendation is visible but flagged.
    grounding_passed: Optional[bool] = Field(
        default=None,
        description=(
            "True if the grounding/consistency check passed; False if it failed "
            "after a bounded retry (recommendation flagged, not suppressed); "
            "None if the check has not run yet."
        ),
    )
    grounding_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Issues found by the grounding check that remain unresolved. "
            "Empty when grounding_passed is True or None."
        ),
    )
