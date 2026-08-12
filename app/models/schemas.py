"""
Pydantic schemas shared across the AWS Instance Advisor agent.

Defines the structured data contracts between graph nodes:
- UserRequirements: raw information collected from the user
- TechnicalNeeds: the system design reasoner's interpretation of those requirements
- InstanceCandidate: a single EC2 instance type fetched from the live data source
- InstanceRecommendation: the final structured recommendation returned to the user
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
    resource_profile: ResourceProfile
    traffic_pattern: TrafficPattern
    requires_gpu: bool = False
    scaling_recommendation: str = Field(
        ...,
        description="Vertical vs horizontal scaling view, e.g. 'horizontal with auto scaling due to bursty peak window'.",
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