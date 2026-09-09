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
