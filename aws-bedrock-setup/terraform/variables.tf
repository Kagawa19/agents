# variables.tf - Defines all variables used in the Terraform configuration

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "aws_access_key" {
  description = "AWS Access Key"
  type        = string
  sensitive   = true
}

variable "aws_secret_key" {
  description = "AWS Secret Key"
  type        = string
  sensitive   = true
}

variable "project_name" {
  description = "Name of the multiagent LLM system project"
  type        = string
  default     = "multiagent-llm-system"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# Bedrock-specific variables
variable "bedrock_model_ids" {
  description = "List of Bedrock model IDs to enable"
  type        = list(string)
  default     = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
  ]
}

# IAM Role for Bedrock Access
variable "bedrock_role_name" {
  description = "IAM Role name for Bedrock access"
  type        = string
  default     = "multiagent-bedrock-access-role"
}