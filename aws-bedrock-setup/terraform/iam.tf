# iam.tf - Defines all IAM resources for AWS Bedrock access

# Create IAM User for Multiagent LLM System
resource "aws_iam_user" "multiagent_user" {
  name = "${var.project_name}-user"
  path = "/system/"
  
  tags = {
    Name        = "${var.project_name} System User"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Create IAM Access Key for the user
resource "aws_iam_access_key" "multiagent_user_key" {
  user = aws_iam_user.multiagent_user.name
}

# IAM Role for Bedrock Access
resource "aws_iam_role" "bedrock_access_role" {
  name = var.bedrock_role_name
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
      },
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_user.multiagent_user.arn
        }
      }
    ]
  })
  
  tags = {
    Name        = "Bedrock Access Role"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Comprehensive Bedrock Access Policy
resource "aws_iam_role_policy" "bedrock_access_policy" {
  name = "${var.project_name}-bedrock-access-policy"
  role = aws_iam_role.bedrock_access_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          # Basic model invocation
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:ListFoundationModels",
          "bedrock:GetFoundationModel",
          
          # Model management
          "bedrock:CreateProvisionedModelThroughput",
          "bedrock:DeleteProvisionedModelThroughput",
          "bedrock:UpdateProvisionedModelThroughput",
          "bedrock:ListProvisionedModelThroughputs",
          "bedrock:GetProvisionedModelThroughput",
          
          # Model customization
          "bedrock:CreateModelCustomizationJob",
          "bedrock:GetModelCustomizationJob",
          "bedrock:ListModelCustomizationJobs",
          "bedrock:StopModelCustomizationJob",
          
          # Guardrails
          "bedrock:ApplyGuardrail",
          "bedrock:CreateGuardrail",
          "bedrock:GetGuardrail",
          "bedrock:ListGuardrails",
          
          # Logging
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# AWS Marketplace access policy for model subscription
resource "aws_iam_role_policy" "marketplace_access_policy" {
  name = "${var.project_name}-marketplace-access-policy"
  role = aws_iam_role.bedrock_access_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "aws-marketplace:Subscribe",
          "aws-marketplace:Unsubscribe",
          "aws-marketplace:ViewSubscriptions"
        ]
        Resource = "*"
      }
    ]
  })
}

# User policy for AWS Marketplace access
resource "aws_iam_user_policy" "user_marketplace_access" {
  name = "marketplace-access"
  user = aws_iam_user.multiagent_user.name
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "aws-marketplace:Subscribe",
          "aws-marketplace:Unsubscribe",
          "aws-marketplace:ViewSubscriptions"
        ]
        Resource = "*"
        Condition = {
          "ForAnyValue:StringEquals" = {
            "aws-marketplace:ProductId" = [
              "prod-6dw3qvchef7zy",  # Claude 3 Sonnet
              "prod-ozonys2hmmpeu",  # Claude 3 Haiku
              "prod-m5ilt4siql27k",  # Claude 3.5 Sonnet
              "prod-cx7ovbu5wex7g",  # Claude 3.5 Sonnet v2
              "prod-4dlfvry4v5hbi",  # Claude 3.7 Sonnet
              "prod-5oba7y7jpji56",  # Claude 3.5 Haiku
              "prod-fm3feywmwerog"   # Claude 3 Opus
            ]
          }
        }
      }
    ]
  })
}

# Policy to allow the user to assume the Bedrock role
resource "aws_iam_user_policy" "allow_assume_role" {
  name = "allow-assume-bedrock-role"
  user = aws_iam_user.multiagent_user.name
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Resource = aws_iam_role.bedrock_access_role.arn
      }
    ]
  })
}

# Additional policy for S3 access to store/retrieve data
resource "aws_iam_user_policy" "s3_access" {
  name = "s3-data-access"
  user = aws_iam_user.multiagent_user.name
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.bedrock_data_bucket.arn,
          "${aws_s3_bucket.bedrock_data_bucket.arn}/*"
        ]
      }
    ]
  })
}