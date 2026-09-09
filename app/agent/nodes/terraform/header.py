"""
Terraform provider, data source headers, warning blocks, and comment helpers.
"""
from __future__ import annotations

AWS_PROVIDER_VERSION = ">= 5.73.0"

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


def _header() -> str:
    return f"""\
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "{AWS_PROVIDER_VERSION}"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

data "aws_vpc" "default" {{
  default = true
}}

data "aws_subnets" "default" {{
  filter {{
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }}
}}

data "aws_ami" "app" {{
  most_recent = true
  owners      = ["amazon"]

  filter {{
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }}
}}
"""


def _comment_block(text: str) -> str:
    lines = text.rstrip()
    wrapped_lines = []
    for ln in lines.split("\n"):
        wrapped_lines.append(f" * {ln}" if ln else " *")
    return "/*\n" + "\n".join(wrapped_lines) + "\n */"
