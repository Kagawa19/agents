# main.tf - Main Terraform configuration file

# This is the main Terraform file for AWS Bedrock setup
# It sets up the core resources needed for the project

# Random suffix to ensure unique resource names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 bucket for storing data related to Bedrock processing (optional)
resource "aws_s3_bucket" "bedrock_data_bucket" {
  bucket = "${var.project_name}-data-${random_string.suffix.result}"
  
  tags = {
    Name        = "${var.project_name}-data-bucket"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Set bucket ownership controls
resource "aws_s3_bucket_ownership_controls" "bedrock_data_bucket_ownership" {
  bucket = aws_s3_bucket.bedrock_data_bucket.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

# Lock down bucket access to private
resource "aws_s3_bucket_acl" "bedrock_data_bucket_acl" {
  depends_on = [aws_s3_bucket_ownership_controls.bedrock_data_bucket_ownership]
  bucket = aws_s3_bucket.bedrock_data_bucket.id
  acl    = "private"
}