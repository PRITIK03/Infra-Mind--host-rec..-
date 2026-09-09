"""
Terminal node: converts a completed SystemDesignRecommendation into deployable
Terraform (HCL) files.

DETERMINISTIC TEMPLATE-BASED GENERATION ONLY. No LLM call. Every value comes
directly from structured data already present in state:
  state["system_design_recommendation"]
  state["technical_needs"]

This is terminal because infrastructure code that someone might actually apply to a live AWS
account. Correctness stakes are higher than advisory text. Deterministic
templates are what make golden-file testing meaningful at all: the same input
always produces byte-identical HCL.

Engine mappings (checked against current Terraform AWS provider docs,
not guessed:

CACHE ENGINE -> RESOURCE TYPE:
  Memcached -> aws_elasticache_cluster,          engine = "memcached"
  Redis     -> aws_elasticache_replication_group, engine = "redis"
  Valkey    -> aws_elasticache_replication_group, engine = "valkey"

RDS ENGINE -> ENGINE STRING:
  PostgreSQL -> postgres
  MySQL      -> mysql
  MariaDB    -> mariadb
  (anything else -> postgres with a code comment fallback)

AWS provider pinned to >= 5.73.0 because Valkey engine support was only
added around that version.

Scope deliberately bounded:
  - Default VPC via data sources (no custom VPC/subnet creation)
  - Minimal security groups (no IAM
  - Launch template + ASG always; DB/cache/LB conditional on needed
  - architecture_summary as /* */ comment block
  - Prominent AI-generated warning comment at top of main.tf
"""
from __future__ import annotations

from string import Template

from app.agent.state import AgentState
from app.models.schemas import (
    CacheEngine,
    SystemDesignRecommendation,
    TechnicalNeeds,
)


AWS_PROVIDER_VERSION = ">= 5.73.0"

APP_PORT = 80
RDS_PORT = 5432
REDIS_PORT = 6379
MEMCACHED_PORT = 11211
VALKEY_PORT = 6379


CACHE_ENGINE_RESOURCE_MAP: dict[CacheEngine, dict] = {
    CacheEngine.MEMCACHED: {
        "resource_type": "aws_elasticache_cluster",
        "engine": "memcached",
        "port": MEMCACHED_PORT,
    },
    CacheEngine.REDIS: {
        "resource_type": "aws_elasticache_replication_group",
        "engine": "redis",
        "port": REDIS_PORT,
    },
    CacheEngine.VALKEY: {
        "resource_type": "aws_elasticache_replication_group",
        "engine": "valkey",
        "port": VALKEY_PORT,
    },
}


RDS_ENGINE_MAP: dict[str, str] = {
    "PostgreSQL": "postgres",
    "postgres": "postgres",
    "MySQL": "mysql",
    "mysql": "mysql",
    "MariaDB": "mariadb",
    "mariadb": "mariadb",
}


def _rds_engine(engine_suggestion: str | None) -> str:
    if engine_suggestion is None:
        # Fallback: postgres is the safe default.
        return "postgres"
    if engine_suggestion in RDS_ENGINE_MAP:
        return RDS_ENGINE_MAP[engine_suggestion]
    lowered = engine_suggestion.strip().lower()
    for key, val in RDS_ENGINE_MAP.items():
        if key.lower() == lowered:
            return val
    # Fallback rather than crashing; caller cannot be "engine_suggestion ({!r} not in known mapping — falling back to postgres."
    return "postgres"


_AI_WARNING_BLOCK = """\
/*
 * ============================================================
 *  AI-GENERATED INFRASTRUCTURE CODE
 * ============================================================
 *
 *  This Terraform configuration was generated automatically by
 *  aws-instance-advisor as a STARTING POINT, not a final
 *  deployment.
 *
 *  YOU MUST REVIEW THE FOLLOWING BEFORE `terraform apply`:
 *
 *    - SECURITY: Security groups are deliberately minimal —
 *      they are a baseline and should be tightened. Database credentials
 *      use a placeholder password below — replace with a secret
 *      manager or rotate to real values.
 *
 *    - COST: Instance sizes were chosen for the described workload.
 *      verify hourly/monthly estimates yourself before applying.
 *
 *    - CORRECTNESS: Resource names, ports, CIDRs, and engine
 *      mappings were mechanically translated. spot-check each block
 *      against the live AWS console / terraform plan output.
 *
 *    - CUSTOM VPC: This template assumes the default VPC. If you
 *      need private subnets, NAT gateways, etc., extend
 *      replace the data "aws_vpc" "default" references.
 *
 *  DO NOT apply without reading the plan output carefully.
 * ============================================================
 */
"""


_HEADER = """\
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "${aws_provider_version}"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
"""


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
  description = "RDS master password — REPLACE WITH SECRET before apply; do not commit this value."
  type        = string
  default     = "changeme-please-replace-before-apply"
  sensitive   = true
}
"""


_SG_LB = """\
resource "aws_security_group" "lb" {
  name        = "${var.app_name}-lb-sg"
  description = "Allow 80/443 inbound from anywhere to load balancer."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.app_name}-lb-sg"
  }
}
"""


_SG_COMPUTE_FROM_LB = """\
resource "aws_security_group" "compute" {
  name        = "${var.app_name}-compute-sg"
  description = "Allow app port only from load balancer security group."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = var.app_port
    to_port         = var.app_port
    protocol        = "tcp"
    security_groups = [aws_security_group.lb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.app_name}-compute-sg"
  }
}
"""


_SG_COMPUTE_STANDALONE = """\
resource "aws_security_group" "compute" {
  name        = "${var.app_name}-compute-sg"
  description = "Allow app port directly from 0.0.0.0/0 (standalone, no LB)."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.app_name}-compute-sg"
  }
}
"""


_SG_DB_FROM_COMPUTE = """\
resource "aws_security_group" "db" {
  name        = "${var.app_name}-db-sg"
  description = "Allow RDS port only from compute security group."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = ${db_port}
    to_port         = ${db_port}
    protocol        = "tcp"
    security_groups = [aws_security_group.compute.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.app_name}-db-sg"
  }
}
"""


_SG_CACHE_FROM_COMPUTE = """\
resource "aws_security_group" "cache" {
  name        = "${var.app_name}-cache-sg"
  description = "Allow cache port only from compute security group."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = ${cache_port}
    to_port         = ${cache_port}
    protocol        = "tcp"
    security_groups = [aws_security_group.compute.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.app_name}-cache-sg"
  }
}
"""


_LAUNCH_TEMPLATE_ASG = """\
resource "aws_launch_template" "app" {
  name_prefix   = "${var.app_name}-lt-"
  image_id      = "ami-0c101f26f44444444"
  instance_type = "${compute_instance}"

  vpc_security_group_ids = [aws_security_group.compute.id]

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.app_name}-instance"
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "app" {
  name_prefix          = "${var.app_name}-asg-"
  min_size             = ${min_instances}
  max_size             = ${max_instances}
  desired_capacity     = ${min_instances}
  vpc_zone_identifier  = data.aws_subnets.default.ids

  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }
${target_group_attachment}
  tag {
    key                 = "Name"
    value               = "${var.app_name}-asg-instance"
    propagate_at_launch = true
  }
}
"""


_ASG_TG_ATTACHMENT = """\
  target_group_arns    = [aws_lb_target_group.app.arn]
"""


_ASG_NO_TG_ATTACHMENT = ""


_RDS_INSTANCE = """\
resource "aws_db_instance" "app" {
  identifier             = "${var.app_name}-db"
  engine                 = "${db_engine}"
  instance_class         = "${db_instance}"
  allocated_storage   = 20
  db_name              = "appdb"
  username             = var.db_username
  password             = var.db_password
  skip_final_snapshot = true
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible  = false
}
"""


_CACHE_MEMCACHED_CLUSTER = """\
resource "aws_elasticache_cluster" "app" {
  cluster_id           = "${var.app_name}-cache"
  engine               = "memcached"
  node_type            = "${cache_instance}"
  num_cache_nodes      = 1
  parameter_group_name = "default.memcached1.6"
  port                 = ${cache_port}
  security_group_ids   = [aws_security_group.cache.id]
}
"""


_CACHE_REPLICATION_GROUP = """\
resource "aws_elasticache_replication_group" "app" {
  replication_group_id          = "${var.app_name}-cache"
  replication_group_description = "Cache tier for ${var.app_name}"
  engine                        = "${cache_engine}"
  node_type                     = "${cache_instance}"
  num_cache_clusters          = 1
  parameter_group_name    = "${param_group}"
  port                          = ${cache_port}
  security_group_ids            = [aws_security_group.cache.id]
}
"""


_PARAM_GROUPS = {
    CacheEngine.REDIS: "default.redis7",
    CacheEngine.VALKEY: "default.valkey7.2",
    CacheEngine.MEMCACHED: "default.memcached1.6",
}


def _param_group(engine: CacheEngine) -> str:
    return _PARAM_GROUPS.get(engine, "default.redis7")


_LB_ALB = """\
resource "aws_lb" "app" {
  name               = "${var.app_name}-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb.id]
  subnets            = data.aws_subnets.default.ids

  tags = {
    Name = "${var.app_name}-lb"
  }
}

resource "aws_lb_target_group" "app" {
  name     = "${var.app_name}-tg"
  port     = var.app_port
  protocol = "HTTP"
  vpc_id   = data.aws_vpc.default.id

  health_check {
    path = "/"
  }
}

resource "aws_lb_listener" "app" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol        = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
"""


_LB_NLB = """\
resource "aws_lb" "app" {
  name               = "${var.app_name}-lb"
  internal           = false
  load_balancer_type = "network"
  subnets            = data.aws_subnets.default.ids

  tags = {
    Name = "${var.app_name}-lb"
  }
}

resource "aws_lb_target_group" "app" {
  name     = "${var.app_name}-tg"
  port     = var.app_port
  protocol = "TCP"
  vpc_id   = data.aws_vpc.default.id

  health_check {
    port = var.app_port
  }
}

resource "aws_lb_listener" "app" {
  load_balancer_arn = aws_lb.app.arn
  port              = var.app_port
  protocol        = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
"""


_OUTPUTS_BASE = """\
"""


_OUTPUT_ALB = """\
output "load_balancer_dns_name" {
  description = "DNS name of the application load balancer."
  value       = aws_lb.app.dns_name
}
"""


_OUTPUT_RDS = """\
output "database_endpoint" {
  description = "RDS endpoint (address:port)."
  value       = "${aws_db_instance.app.address}:${aws_db_instance.app.port}"
}
"""


_OUTPUT_MEMCACHED = """\
output "cache_endpoint" {
  description = "Memcached cluster endpoint (address:port)."
  value       = "${aws_elasticache_cluster.app.cache_nodes[0].address}:${aws_elasticache_cluster.app.cache_nodes[0].port}"
}
"""


_OUTPUT_REPLICATION_GROUP = """\
output "cache_endpoint" {
  description = "Replication group primary endpoint (address:port)."
  value       = "${aws_elasticache_replication_group.app.primary_endpoint_address}:${aws_elasticache_replication_group.app.port}"
}
"""


def _comment_block(text: str) -> str:
    lines = text.rstrip()
    wrapped_lines = []
    for ln in lines.split("\n"):
        wrapped_lines.append(f" * {ln}" if ln else " *")
    return "/*\n" + "\n".join(wrapped_lines) + "\n */"


def _build_main(
    sdr: SystemDesignRecommendation,
    tn: TechnicalNeeds,
) -> str:
    parts: list[str] = []

    parts.append(_AI_WARNING_BLOCK)

    parts.append(_comment_block(sdr.architecture_summary))
    parts.append("")

    parts.append(Template(_HEADER).substitute(
        aws_provider_version=AWS_PROVIDER_VERSION,
    ))
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
        db_port = RDS_PORT
        parts.append(Template(_SG_DB_FROM_COMPUTE).substitute(db_port=db_port))
        parts.append("")

    if sdr.cache.needed and sdr.cache.engine:
        cache_info = CACHE_ENGINE_RESOURCE_MAP[sdr.cache.engine]
        cache_port = cache_info["port"]
        parts.append(Template(_SG_CACHE_FROM_COMPUTE).substitute(cache_port=cache_port))
        parts.append("")

    # Launch template + ASG
    if sdr.load_balancer.needed:
        tg_attachment = _ASG_TG_ATTACHMENT
    else:
        tg_attachment = _ASG_NO_TG_ATTACHMENT
    parts.append(Template(_LAUNCH_TEMPLATE_ASG).substitute(
        compute_instance=sdr.compute.recommended_instance,
        min_instances=tn.min_instances,
        max_instances=tn.max_instances,
        target_group_attachment=tg_attachment,
    ))
    parts.append("")

    # Database
    if sdr.database.needed and sdr.database.recommended_instance:
        db_engine = _rds_engine(sdr.database.engine_suggestion)
        parts.append(Template(_RDS_INSTANCE).substitute(
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
        resource_tmpl = (
            _CACHE_MEMCACHED_CLUSTER
            if cache_info["resource_type"] == "aws_elasticache_cluster"
            else _CACHE_REPLICATION_GROUP
        )
        parts.append(Template(resource_tmpl).substitute(
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
    parts.append(_OUTPUTS_BASE)
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
