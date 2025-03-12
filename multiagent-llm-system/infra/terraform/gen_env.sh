#!/bin/bash

# Terraform output to .env file
terraform output -json | jq -r 'to_entries[] | "\(.key)=\(.value.value)"' > .env

# Append additional configuration
cat << EOF >> .env
AWS_REGION=${AWS_REGION:-us-east-1}
BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
EOF

echo "Environment file generated successfully!"