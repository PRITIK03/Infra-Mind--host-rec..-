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

/*
 * LB spreads burst -> ASG shares load -> Redis strips reads -> RDS handles writes.
 */

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.73.0"
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


resource "aws_security_group" "db" {
  name        = "${var.app_name}-db-sg"
  description = "Allow RDS port only from compute security group."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 5432
    to_port         = 5432
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


resource "aws_security_group" "cache" {
  name        = "${var.app_name}-cache-sg"
  description = "Allow cache port only from compute security group."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 6379
    to_port         = 6379
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


resource "aws_launch_template" "app" {
  name_prefix = "${var.app_name}-lt-"
  image_id = "ami-0c101f26f44444444"
  instance_type = "m5.xlarge"

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
  name_prefix = "${var.app_name}-asg-"
  min_size = 4
  max_size = 12
  desired_capacity = 4
  vpc_zone_identifier = data.aws_subnets.default.ids

  launch_template {
    id = aws_launch_template.app.id
    version = "$Latest"
  }
  target_group_arns = [aws_lb_target_group.app.arn]
  tag {
    key = "Name"
    value = "${var.app_name}-asg-instance"
    propagate_at_launch = true
  }
}


resource "aws_db_instance" "app" {
  identifier = "${var.app_name}-db"
  engine = "postgres"
  instance_class = "db.m5.large"
  allocated_storage = 20
  db_name = "appdb"
  username = var.db_username
  password = var.db_password
  skip_final_snapshot = true
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible = false
}


resource "aws_elasticache_replication_group" "app" {
  replication_group_id = "${var.app_name}-cache"
  replication_group_description = "Cache tier for ${var.app_name}"
  engine = "redis"
  node_type = "cache.r5.large"
  num_cache_clusters = 1
  parameter_group_name = "default.redis7"
  port = 6379
  security_group_ids = [aws_security_group.cache.id]
}


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
