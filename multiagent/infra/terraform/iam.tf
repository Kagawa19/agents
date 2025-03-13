# Create IAM User for Multiagent LLM System
resource "aws_iam_user" "multiagent_user" {
  name = "${var.project_name}-user"
  path = "/system/"
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
}

# Bedrock Access Policy
resource "aws_iam_role_policy" "bedrock_access_policy" {
  name = "${var.project_name}-bedrock-access-policy"
  role = aws_iam_role.bedrock_access_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:ListFoundationModels",
          "bedrock:GetFoundationModel"
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach additional policy for general AWS access
resource "aws_iam_user_policy_attachment" "multiagent_user_policy" {
  user       = aws_iam_user.multiagent_user.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}