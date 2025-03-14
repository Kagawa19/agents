"""
Amazon Bedrock Tool implementation.
Provides functions for interacting with Amazon Bedrock models.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Union

from multiagent.app.monitoring.tracer import LangfuseTracer
from multiagent.app.monitoring.metrics import track_llm_call


logger = logging.getLogger(__name__)


class BedrockTool:
    """
    Tool for interacting with Amazon Bedrock models.
    Provides functions for text generation and embedding generation.
    """
    
    def __init__(self, config: Dict[str, Any], tracer: LangfuseTracer):
        """
        Initialize the Bedrock tool.
        
        Args:
            config: Configuration including AWS credentials and region
            tracer: LangfuseTracer instance for monitoring
        """
        self.config = config
        self.tracer = tracer
        
        # Initialize AWS clients
        try:
            import boto3
            
            self.session = boto3.Session(
                aws_access_key_id=config.get("aws_access_key_id"),
                aws_secret_access_key=config.get("aws_secret_access_key"),
                region_name=config.get("region_name", "us-east-1")
            )
            
            self.bedrock_runtime = self.session.client(
                service_name="bedrock-runtime",
                region_name=config.get("region_name", "us-east-1")
            )
            
            self.default_model = config.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
            logger.info(f"Bedrock tool initialized with model: {self.default_model}")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock tool: {e}")
            self.initialized = False
    
    def generate_text(
        self, 
        prompt: str, 
        model_id: Optional[str] = None, 
        max_tokens: int = 1000, 
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text using Bedrock models.
        
        Args:
            prompt: Prompt to process
            model_id: Bedrock model ID (defaults to configured default)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
            system_message: Optional system message (for models that support it)
            
        Returns:
            Dictionary containing the generated text
        """
        with self.tracer.span("bedrock_generate_text"):
            start_time = time.time()
            
            if not self.initialized:
                return {"error": "Bedrock tool not initialized"}
            
            # Use default model if not specified
            model_id = model_id or self.default_model
            
            try:
                # Prepare request body based on model provider
                request_body = self._prepare_request_body(
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_message=system_message
                )
                
                # Invoke model
                response = self.bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body)
                )
                
                # Parse response
                response_body = json.loads(response["body"].read().decode("utf-8"))
                
                # Extract text based on model provider
                generated_text, tokens_used = self._extract_response(model_id, response_body)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Prepare result
                result = {
                    "text": generated_text,
                    "model": model_id,
                    "tokens_used": tokens_used,
                    "execution_time": execution_time
                }
                
                # Track metrics
                track_llm_call(
                    provider="bedrock",
                    model=model_id,
                    tokens=tokens_used,
                    call_type="text_generation",
                    latency=execution_time
                )
                
                # Log generation
                self.tracer.log_generation(
                    model=model_id,
                    prompt=prompt,
                    response=generated_text,
                    metadata={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "execution_time": execution_time,
                        "tokens_used": tokens_used
                    }
                )
                
                return result
            except Exception as e:
                logger.error(f"Error generating text: {str(e)}")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_llm_call(
                    provider="bedrock",
                    model=model_id,
                    tokens=0,
                    call_type="error",
                    latency=execution_time
                )
                
                # Log event
                self.tracer.log_event(
                    event_type="bedrock_error",
                    event_data={
                        "error": str(e),
                        "model": model_id,
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error generating text: {str(e)}",
                    "model": model_id,
                    "execution_time": execution_time
                }
    
    def embed_text(
        self, 
        text: Union[str, List[str]], 
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for text using Bedrock.
        
        Args:
            text: Text to embed (string or list of strings)
            model_id: Bedrock model ID for embeddings
            
        Returns:
            Dictionary containing the embedding(s)
        """
        with self.tracer.span("bedrock_embed_text"):
            start_time = time.time()
            
            if not self.initialized:
                return {"error": "Bedrock tool not initialized"}
            
            # Default to an embedding model if not specified
            model_id = model_id or "amazon.titan-embed-text-v1"
            
            try:
                # Convert input to list if it's a string
                if isinstance(text, str):
                    texts = [text]
                else:
                    texts = text
                
                # Prepare embeddings result
                embeddings_result = []
                
                # Process each text separately
                for i, single_text in enumerate(texts):
                    # Prepare request body based on model
                    if "amazon.titan-embed" in model_id:
                        request_body = {
                            "inputText": single_text
                        }
                    elif "cohere.embed" in model_id:
                        request_body = {
                            "texts": [single_text],
                            "input_type": "search_document"
                        }
                    else:
                        # Default to Titan format
                        request_body = {
                            "inputText": single_text
                        }
                    
                    # Invoke model
                    response = self.bedrock_runtime.invoke_model(
                        modelId=model_id,
                        body=json.dumps(request_body)
                    )
                    
                    # Parse response
                    response_body = json.loads(response["body"].read().decode("utf-8"))
                    
                    # Extract embedding based on model
                    if "amazon.titan-embed" in model_id:
                        embedding = response_body.get("embedding", [])
                    elif "cohere.embed" in model_id:
                        embedding = response_body.get("embeddings", [[]])[0]
                    else:
                        embedding = response_body.get("embedding", [])
                    
                    # Add to results
                    embeddings_result.append({
                        "text": single_text[:100] + "..." if len(single_text) > 100 else single_text,
                        "embedding": embedding
                    })
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_llm_call(
                    provider="bedrock",
                    model=model_id,
                    tokens=sum(len(t.split()) for t in texts),
                    call_type="embedding",
                    latency=execution_time
                )
                
                # Log event
                self.tracer.log_event(
                    event_type="bedrock_embed_text",
                    event_data={
                        "model": model_id,
                        "text_count": len(texts),
                        "embedding_dimension": len(embeddings_result[0]["embedding"]) if embeddings_result else 0,
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "embeddings": embeddings_result,
                    "execution_time": execution_time,
                    "model": model_id
                }
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_llm_call(
                    provider="bedrock",
                    model=model_id,
                    tokens=0,
                    call_type="error",
                    latency=execution_time
                )
                
                # Log event
                self.tracer.log_event(
                    event_type="bedrock_error",
                    event_data={
                        "error": str(e),
                        "model": model_id,
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error generating embedding: {str(e)}",
                    "model": model_id,
                    "execution_time": execution_time
                }
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using Bedrock models.
        
        Args:
            messages: List of message dictionaries with role and content
            model_id: Bedrock model ID
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing the response
        """
        with self.tracer.span("bedrock_chat_completion"):
            start_time = time.time()
            
            if not self.initialized:
                return {"error": "Bedrock tool not initialized"}
            
            # Use default model if not specified
            model_id = model_id or self.default_model
            
            try:
                # Extract system message if present
                system_message = None
                user_messages = []
                
                for message in messages:
                    if message.get("role") == "system":
                        system_message = message.get("content", "")
                    else:
                        user_messages.append(message)
                
                # For simplicity, concatenate user messages into a single prompt
                # This is a simplification; in a production system, you'd handle the conversation history properly
                if len(user_messages) > 0:
                    prompt = user_messages[-1].get("content", "")
                else:
                    prompt = ""
                
                # Generate text
                return self.generate_text(
                    prompt=prompt,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_message=system_message
                )
            except Exception as e:
                logger.error(f"Error in chat completion: {str(e)}")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_llm_call(
                    provider="bedrock",
                    model=model_id,
                    tokens=0,
                    call_type="error",
                    latency=execution_time
                )
                
                # Log event
                self.tracer.log_event(
                    event_type="bedrock_error",
                    event_data={
                        "error": str(e),
                        "model": model_id,
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error in chat completion: {str(e)}",
                    "model": model_id,
                    "execution_time": execution_time
                }
    
    def _prepare_request_body(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare request body based on model provider.
        
        Args:
            model_id: Bedrock model ID
            prompt: Prompt text
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
            system_message: Optional system message
            
        Returns:
            Request body dictionary
        """
        # Anthropic Claude models
        if "anthropic.claude" in model_id:
            request_body = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "anthropic_version": "bedrock-2023-05-31"
            }
            
            # Format with system message if available
            if system_message:
                request_body["system"] = system_message
                request_body["messages"] = [{"role": "user", "content": prompt}]
            else:
                # Older Claude models use this format
                if "claude-3" in model_id:
                    request_body["messages"] = [{"role": "user", "content": prompt}]
                else:
                    request_body["prompt"] = f"\n\nHuman: {prompt}\n\nAssistant:"
        
        # AI21 Jurassic models
        elif "ai21.j" in model_id:
            request_body = {
                "prompt": prompt,
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        
        # Amazon Titan models
        elif "amazon.titan" in model_id:
            request_body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature
                }
            }
        
        # Cohere models
        elif "cohere" in model_id:
            request_body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        
        # Meta Llama models
        elif "meta.llama" in model_id:
            request_body = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature
            }
        
        # Default to Claude format
        else:
            request_body = {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature
            }
        
        return request_body
    
    def _extract_response(self, model_id: str, response_body: Dict[str, Any]) -> tuple:
        """
        Extract text and tokens from response based on model provider.
        
        Args:
            model_id: Bedrock model ID
            response_body: Response body from Bedrock
            
        Returns:
            Tuple of (generated_text, tokens_used)
        """
        # Anthropic Claude models
        if "anthropic.claude" in model_id:
            if "claude-3" in model_id and "content" in response_body:
                # Claude 3 format
                generated_text = response_body.get("content", [])[0].get("text", "")
                tokens_used = response_body.get("usage", {}).get("output_tokens", 0)
            else:
                # Older Claude format
                generated_text = response_body.get("completion", "")
                tokens_used = len(generated_text.split())  # Approximation
        
        # AI21 Jurassic models
        elif "ai21.j" in model_id:
            generated_text = response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
            tokens_used = response_body.get("completions", [{}])[0].get("finishReason", {}).get("tokens", 0)
        
        # Amazon Titan models
        elif "amazon.titan" in model_id:
            generated_text = response_body.get("results", [{}])[0].get("outputText", "")
            tokens_used = len(generated_text.split())  # Approximation
        
        # Cohere models
        elif "cohere" in model_id:
            generated_text = response_body.get("generations", [{}])[0].get("text", "")
            tokens_used = response_body.get("meta", {}).get("billed_units", {}).get("output_tokens", 0)
        
        # Meta Llama models
        elif "meta.llama" in model_id:
            generated_text = response_body.get("generation", "")
            tokens_used = len(generated_text.split())  # Approximation
        
        # Default extraction
        else:
            generated_text = response_body.get("completion", "")
            tokens_used = len(generated_text.split())  # Approximation
        
        return generated_text, tokens_used