variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "env" {
  description = "Environment name (e.g. dev, prod). Used for unique resource names."
  type        = string
  default     = "dev"
}

variable "project" {
  description = "Project name prefix for resource naming"
  type        = string
  default     = "recsys"
}

variable "model_prefix" {
  description = "S3 prefix (under ML_BUCKET) pointing at the deployed serving bundle"
  type        = string
  default     = "ml/models/20260908-014740"
}
