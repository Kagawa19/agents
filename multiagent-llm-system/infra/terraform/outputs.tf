# Output AWS Access Credentials (BE VERY CAREFUL WITH THIS)
output "aws_access_key_id" {
  value     = aws_iam_access_key.multiagent_user_key.id
  sensitive = true
}

output "aws_secret_access_key" {
  value     = aws_iam_access_key.multiagent_user_key.secret
  sensitive = true
}

output "bedrock_role_arn" {
  value = aws_iam_role.bedrock_access_role.arn
}

# Optional: Output enabled Bedrock models
output "enabled_bedrock_models" {
  value = var.bedrock_model_ids
}