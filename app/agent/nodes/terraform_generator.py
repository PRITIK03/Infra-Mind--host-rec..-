"""
Terminal node: converts a completed SystemDesignRecommendation into deployable
Terraform (HCL) files.

Orchestrates terraform file generation by delegating to resource builders in
the app.agent.nodes.terraform package.
"""
from __future__ import annotations

from app.agent.nodes.terraform.cache import (
    CACHE_ENGINE_PORT_MAP,
    CACHE_ENGINE_RESOURCE_MAP,
    _cache_memcached_cluster,
    _cache_replication_group,
    _param_group,
)
from app.agent.nodes.terraform.compute import _launch_template_asg
from app.agent.nodes.terraform.database import (
    RDS_ENGINE_PORT_MAP,
    RDS_PORT,
    _rds_engine,
    _rds_instance,
)
from app.agent.nodes.terraform.header import (
    _AI_WARNING_BLOCK,
    _comment_block,
    _header,
)
from app.agent.nodes.terraform.load_balancer import (
    _LB_ALB,
    _LB_NLB,
    _OUTPUT_ALB,
    _OUTPUT_MEMCACHED,
    _OUTPUT_RDS,
    _OUTPUT_REPLICATION_GROUP,
)
from app.agent.nodes.terraform.security_groups import (
    _SG_COMPUTE_FROM_LB,
    _SG_COMPUTE_STANDALONE,
    _SG_LB,
    _sg_cache_from_compute,
    _sg_db_from_compute,
)
from app.agent.state import AgentState
from app.models.schemas import (
    CacheEngine,
    SystemDesignRecommendation,
    TechnicalNeeds,
)

APP_PORT = 80

_VARIABLES_TF = """\
variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "Short application name, used as a resource name prefix."
  type        = string
  default     = "app"
}

variable "app_port" {
  description = "TCP port the application listens on inside the compute tier."
  type        = number
  default     = 80
}

variable "db_username" {
  description = "RDS master username — replace with a secrets-managed value before apply."
  type        = string
  default     = "dbadmin"
}

variable "db_password" {
  description = "RDS master password — supply via TF_VAR_db_password or -var; do not commit this value."
  type        = string
  sensitive   = true
}
"""


def _build_main(
    sdr: SystemDesignRecommendation,
    tn: TechnicalNeeds,
) -> str:
    parts: list[str] = []

    parts.append(_AI_WARNING_BLOCK)

    parts.append(_comment_block(sdr.architecture_summary))
    parts.append("")

    parts.append(_header())
    parts.append("")

    # Security groups
    if sdr.load_balancer.needed:
        parts.append(_SG_LB)
        parts.append("")
        parts.append(_SG_COMPUTE_FROM_LB)
    else:
        parts.append(_SG_COMPUTE_STANDALONE)
    parts.append("")

    if sdr.database.needed:
        db_engine = _rds_engine(sdr.database.engine_suggestion)
        db_port = RDS_ENGINE_PORT_MAP.get(db_engine, RDS_PORT)
        parts.append(_sg_db_from_compute(db_port))
        parts.append("")

    if sdr.cache.needed and sdr.cache.engine:
        cache_port = CACHE_ENGINE_PORT_MAP[sdr.cache.engine]
        parts.append(_sg_cache_from_compute(cache_port))
        parts.append("")

    # Launch template + ASG
    parts.append(_launch_template_asg(
        compute_instance=sdr.compute.recommended_instance,
        min_instances=tn.min_instances,
        max_instances=tn.max_instances,
        with_target_group=sdr.load_balancer.needed,
    ))
    parts.append("")

    # Database
    if sdr.database.needed and sdr.database.recommended_instance:
        db_engine = _rds_engine(sdr.database.engine_suggestion)
        parts.append(_rds_instance(
            db_engine=db_engine,
            db_instance=sdr.database.recommended_instance,
        ))
        parts.append("")

    # Cache
    if sdr.cache.needed and sdr.cache.recommended_instance and sdr.cache.engine:
        cache_info = CACHE_ENGINE_RESOURCE_MAP[sdr.cache.engine]
        cache_engine_str = cache_info["engine"]
        cache_instance = sdr.cache.recommended_instance
        cache_port = cache_info["port"]
        if cache_info["resource_type"] == "aws_elasticache_cluster":
            parts.append(_cache_memcached_cluster(
                cache_instance=cache_instance,
                cache_port=cache_port,
            ))
        else:
            parts.append(_cache_replication_group(
                cache_engine=cache_engine_str,
                cache_instance=cache_instance,
                cache_port=cache_port,
                param_group=_param_group(sdr.cache.engine),
            ))
        parts.append("")

    # Load balancer
    if sdr.load_balancer.needed:
        lb_type = (sdr.load_balancer.load_balancer_type or "").lower()
        if "network" in lb_type or "nlb" in lb_type:
            parts.append(_LB_NLB)
        else:
            parts.append(_LB_ALB)
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _build_outputs(
    sdr: SystemDesignRecommendation,
) -> str:
    parts: list[str] = []
    if sdr.load_balancer.needed:
        parts.append(_OUTPUT_ALB)
    if sdr.database.needed and sdr.database.recommended_instance:
        parts.append(_OUTPUT_RDS)
    if sdr.cache.needed and sdr.cache.engine and sdr.cache.recommended_instance:
        if sdr.cache.engine == CacheEngine.MEMCACHED:
            parts.append(_OUTPUT_MEMCACHED)
        else:
            parts.append(_OUTPUT_REPLICATION_GROUP)
    if not parts:
        return ""
    return "\n".join(parts).rstrip() + "\n"


def generate_terraform(state: AgentState) -> AgentState:
    sdr: SystemDesignRecommendation | None = state.get("system_design_recommendation")
    tn: TechnicalNeeds | None = state.get("technical_needs")

    if sdr is None or tn is None:
        raise RuntimeError(
            "system_design_recommendation and technical_needs must be present "
            "before calling generate_terraform."
        )

    main_tf = _build_main(sdr, tn)
    outputs_tf = _build_outputs(sdr)

    state["terraform_files"] = {
        "main.tf": main_tf,
        "variables.tf": _VARIABLES_TF,
        "outputs.tf": outputs_tf,
    }
    return state
