output "load_balancer_dns_name" {
  description = "DNS name of the application load balancer."
  value       = aws_lb.app.dns_name
}

output "database_endpoint" {
  description = "RDS endpoint (address:port)."
  value       = "${aws_db_instance.app.address}:${aws_db_instance.app.port}"
}

output "cache_endpoint" {
  description = "Replication group primary endpoint (address:port)."
  value       = "${aws_elasticache_replication_group.app.primary_endpoint_address}:${aws_elasticache_replication_group.app.port}"
}
