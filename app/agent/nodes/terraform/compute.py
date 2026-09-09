"""
Compute resource builder (Launch Template + Auto Scaling Group).
"""
from __future__ import annotations


def _launch_template_asg(
    *,
    compute_instance: str,
    min_instances: int,
    max_instances: int,
    with_target_group: bool,
) -> str:
    tg_line = (
        "  target_group_arns = [aws_lb_target_group.app.arn]\n"
        if with_target_group
        else ""
    )
    return f"""\
resource "aws_launch_template" "app" {{
  name_prefix = "${{var.app_name}}-lt-"
  image_id = data.aws_ami.app.id
  instance_type = "{compute_instance}"

  vpc_security_group_ids = [aws_security_group.compute.id]

  tag_specifications {{
    resource_type = "instance"
    tags = {{
      Name = "${{var.app_name}}-instance"
    }}
  }}

  lifecycle {{
    create_before_destroy = true
  }}
}}

resource "aws_autoscaling_group" "app" {{
  name_prefix = "${{var.app_name}}-asg-"
  min_size = {min_instances}
  max_size = {max_instances}
  desired_capacity = {min_instances}
  vpc_zone_identifier = data.aws_subnets.default.ids

  launch_template {{
    id = aws_launch_template.app.id
    version = "$Latest"
  }}
{tg_line}  tag {{
    key = "Name"
    value = "${{var.app_name}}-asg-instance"
    propagate_at_launch = true
  }}
}}
"""
