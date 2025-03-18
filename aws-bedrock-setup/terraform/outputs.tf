# outputs.tf - Defines outputs after Terraform apply

# Output AWS Access Credentials (sensitive values)
output "aws_access_key_id" {
  value     = aws_iam_access_key.multiagent_user_key.id
  sensitive = true
  description = "The access key ID for the multiagent user"
}

output "aws_secret_access_key" {
  value     = aws_iam_access_key.multiagent_user_key.secret
  sensitive = true
  description = "The secret access key for the multiagent user"
}

# Output the Bedrock role ARN - This is what you need for AWS_ROLE_ARN
output "bedrock_role_arn" {
  value = aws_iam_role.bedrock_access_role.arn
  description = "ARN of the IAM role for Bedrock access. Use this for the AWS_ROLE_ARN environment variable."
}

# Output S3 bucket name
output "aws_s3_bucket_name" {
  value = aws_s3_bucket.bedrock_data_bucket.bucket
  description = "Name of the S3 bucket created for storing data"
}

# Output enabled Bedrock models
output "enabled_bedrock_models" {
  value = var.bedrock_model_ids
  description = "List of Bedrock model IDs that should be enabled in the AWS console"
}