import boto3
import json
import os
from dotenv import load_dotenv

def test_bedrock_direct():
    """Test direct access to AWS Bedrock without creating new IAM resources."""
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Get credentials from environment variables
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        # Create a session with the credentials
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # Create a Bedrock runtime client
        bedrock_client = session.client('bedrock-runtime')
        
        # Print success message
        print("Successfully created Bedrock client!")
        
        # Prepare a simple test request for Claude
        request = {
            "prompt": "\n\nHuman: Hello Claude! Can you briefly explain quantum computing?\n\nAssistant:",
            "max_tokens_to_sample": 300,
            "temperature": 0.7
        }
        
        # Invoke the model
        print("Invoking Claude model...")
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(request),
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse and print the response
        response_body = json.loads(response["body"].read())
        print("\nClaude's response:")
        print(response_body.get("completion", "No response"))
        
        print("\nTest successful! You now know your AWS credentials can access Bedrock.")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
        if "AccessDeniedException" in str(e) or "AccessDenied" in str(e):
            print("\nIt looks like your AWS user doesn't have permission to access Bedrock.")
            print("Ask your AWS administrator to grant you the 'bedrock:InvokeModel' permission.")
        
        elif "ResourceNotFoundException" in str(e):
            print("\nThe Claude model is not available in your AWS account or region.")
            print("Make sure Bedrock and the Claude model are enabled in your AWS account.")
        
        elif "EndpointConnectionError" in str(e):
            print("\nCouldn't connect to the Bedrock endpoint.")
            print("Check your internet connection and AWS region configuration.")
        
        return False

if __name__ == "__main__":
    print("Testing direct access to AWS Bedrock...")
    test_bedrock_direct()