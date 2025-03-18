"""
Bedrock API integration.
Provides a client for interacting with Amazon Bedrock models.
"""

import logging
import json
import boto3
import botocore
from typing import Dict, Any, Optional, List
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class BedrockTool:
    """
    Tool for interacting with Amazon Bedrock models.
    Provides methods for text generation, summarization, and more.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the Bedrock tool.
        
        Args:
            config: Configuration parameters including AWS credentials
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.config = config
        self.tracer = tracer
        self.region = config.get("aws_region", "us-east-1")
        self.client = None
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Bedrock runtime client."""
        try:
            # Create initial session with access keys
            session = boto3.Session(
                aws_access_key_id=self.config.get("aws_access_key_id"),
                aws_secret_access_key=self.config.get("aws_secret_access_key"),
                region_name=self.region
            )
            
            # Check if role ARN is provided for role assumption
            role_arn = self.config.get("aws_role_arn")
            if role_arn:
                try:
                    # Assume the Bedrock role
                    sts_client = session.client('sts')
                    assumed_role = sts_client.assume_role(
                        RoleArn=role_arn,
                        RoleSessionName="BedrockSession"
                    )
                    
                    # Create a new session with the assumed role credentials
                    credentials = assumed_role['Credentials']
                    session = boto3.Session(
                        aws_access_key_id=credentials['AccessKeyId'],
                        aws_secret_access_key=credentials['SecretAccessKey'],
                        aws_session_token=credentials['SessionToken'],
                        region_name=self.region
                    )
                    logger.info(f"Successfully assumed role: {role_arn}")
                except Exception as e:
                    logger.error(f"Error assuming role {role_arn}: {str(e)}")
                    # Continue with direct access keys if role assumption fails
            
            # Create the Bedrock runtime client
            self.client = session.client(
                service_name="bedrock-runtime",
                region_name=self.region
            )
            
            logger.info("Initialized Bedrock client")
        except Exception as e:
            logger.error(f"Error initializing Bedrock client: {str(e)}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if the Bedrock client is available."""
        return self.client is not None
    
    async def invoke_model(
        self,
        model_id: str,
        body: Dict[str, Any],
        accept: str = "application/json",
        content_type: str = "application/json"
    ) -> Dict[str, Any]:
        """
        Invoke a Bedrock model with the given parameters.
        
        Args:
            model_id: ID of the model to invoke
            body: Request body for the model
            accept: Accept header for the response
            content_type: Content type header for the request
            
        Returns:
            Model response
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="bedrock_invoke_model",
                input={"model_id": model_id}
            )
        
        try:
            if not self.client:
                raise ValueError("Bedrock client not initialized")
            
            # Serialize the request body
            body_json = json.dumps(body)
            
            # Invoke the model
            response = self.client.invoke_model(
                modelId=model_id,
                body=body_json,
                accept=accept,
                contentType=content_type
            )
            
            # Parse the response
            response_body = json.loads(response["body"].read())
            
            if span:
                span.update(output={"status": "success"})
            
            return response_body
        
        except Exception as e:
            logger.error(f"Error invoking Bedrock model {model_id}: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            raise
    
    def get_model_id(self, model_name: str) -> str:
        """
        Get the model ID for the given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model ID
        """
        # Model ID mapping
        model_ids = {
            "claude": "anthropic.claude-v2",
            "claude-instant": "anthropic.claude-instant-v1",
            "claude-2": "anthropic.claude-v2",
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "titan": "amazon.titan-text-express-v1",
            "titan-embed": "amazon.titan-embed-text-v1",
            "llama2": "meta.llama2-13b-chat-v1"
        }
        
        # Get model ID from mapping, with fallback to the name itself
        return model_ids.get(model_name.lower(), model_name)
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "claude",
        max_tokens: int = 500,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using a Bedrock model.
        
        Args:
            prompt: Text prompt
            model: Model to use (claude, titan, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            stop_sequences: Optional stop sequences
            
        Returns:
            Generated text response
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="bedrock_generate_text",
                input={"model": model, "max_tokens": max_tokens}
            )
        
        try:
            # Get model ID
            model_id = self.get_model_id(model)
            
            # Prepare request body based on model type
            if "claude" in model_id.lower():
                # Claude format
                body = {
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": max_tokens,
                    "temperature": temperature
                }
                
                if stop_sequences:
                    body["stop_sequences"] = stop_sequences
            
            elif "titan" in model_id.lower():
                # Titan format
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                        "stopSequences": stop_sequences or []
                    }
                }
            
            elif "llama" in model_id.lower():
                # Llama format
                body = {
                    "prompt": prompt,
                    "max_gen_len": max_tokens,
                    "temperature": temperature
                }
            
            else:
                # Default format
                body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            
            # Invoke the model
            response = await self.invoke_model(model_id, body)
            
            # Parse response based on model type
            if "claude" in model_id.lower():
                result = {
                    "text": response.get("completion", ""),
                    "model": model_id
                }
            
            elif "titan" in model_id.lower():
                result = {
                    "text": response.get("results", [{}])[0].get("outputText", ""),
                    "model": model_id
                }
            
            elif "llama" in model_id.lower():
                result = {
                    "text": response.get("generation", ""),
                    "model": model_id
                }
            
            else:
                # Default parsing
                result = {
                    "text": response.get("text", ""),
                    "model": model_id
                }
            
            if span:
                span.update(output={"status": "success"})
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating text with Bedrock: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"text": f"Error: {str(e)}", "error": str(e)}