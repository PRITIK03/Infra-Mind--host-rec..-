"""
Load balancer resources (ALB/NLB) and output templates.
"""
from __future__ import annotations

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
