"""
Text generation with Claude.
Provides methods for generating text with Anthropic's Claude models via Bedrock.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

async def generate_text(
    client: Any,
    prompt: str,
    model: str = "claude",
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Generate text using Claude via Bedrock.
    
    Args:
        client: Bedrock client
        prompt: Text prompt
        model: Model to use (claude, claude-instant, etc.)
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter (not used by Claude)
        stop_sequences: Optional stop sequences
        system_prompt: Optional system prompt
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Generated text response
    """
    span = None
    if tracer:
        span = tracer.span(
            name="claude_generate_text",
            input={
                "model": model,
                "prompt_length": len(prompt),
                "max_tokens": max_tokens
            }
        )
    
    try:
        # Import BedrockModels to get model ID
        from multiagent.app.tools.bedrock.bedrock_models import BedrockModels
        
        # Get model info
        bedrock_models = BedrockModels()
        model_info = bedrock_models.get_model_info(model)
        model_id = model_info["id"]
        
        # Check if streaming is supported
        if model_info["provider"] != "anthropic":
            logger.warning(f"Streaming not supported for {model}, falling back to non-streaming")
            return await generate_text(
                client=client,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                tracer=tracer
            )
        
        # Format prompt
        formatted_prompt = bedrock_models.format_prompt(model, prompt, system_prompt)
        formatted_prompt["max_tokens_to_sample"] = max_tokens
        formatted_prompt["temperature"] = temperature
        
        # Set streaming mode
        acceptance = "application/json"
        
        # Convert to JSON
        body_json = json.dumps(formatted_prompt)
        
        # Note: Currently, Bedrock may not support true streaming in the Python SDK
        # This is a placeholder for when streaming is fully supported
        response = await invoke_model(
            client=client,
            model_id=model_id,
            body=formatted_prompt,
            tracer=tracer
        )
        
        # Return full response for now
        if model_info["provider"] == "anthropic":
            generated_text = response.get("completion", "")
        else:
            generated_text = str(response)
        
        return {
            "text": generated_text,
            "model": model_id,
            "is_complete": True
        }
        
    except Exception as e:
        logger.error(f"Error in generate_stream: {str(e)}")
        return {
            "text": f"Error: {str(e)}",
            "error": str(e),
            "is_complete": True
        }

async def invoke_model(
    client: Any,
    model_id: str,
    body: Dict[str, Any],
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Invoke a Bedrock model with the given parameters.
    
    Args:
        client: Bedrock client
        model_id: ID of the model to invoke
        body: Request body for the model
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Model response
    """
    span = None
    if tracer:
        span = tracer.span(
            name="bedrock_invoke_model",
            input={"model_id": model_id}
        )
    
    try:
        if not client:
            raise ValueError("Bedrock client not initialized")
        
        # Serialize the request body
        body_json = json.dumps(body)
        
        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,
            body=body_json,
            accept="application/json",
            contentType="application/json"
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

async def classify_text(
    client: Any,
    text: str,
    categories: List[str],
    model: str = "claude",
    temperature: float = 0.1,
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Classify text into one of the provided categories.
    
    Args:
        client: Bedrock client
        text: Text to classify
        categories: List of possible categories
        model: Model to use
        temperature: Temperature for sampling
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Classification result with category and confidence
    """
    # Create prompt for classification
    categories_list = "\n".join([f"- {category}" for category in categories])
    prompt = f"""
    Please classify the following text into exactly one of the categories listed below.
    Return only the category name without any explanation or additional text.
    
    Text to classify:
    "{text}"
    
    Categories:
    {categories_list}
    
    Classification:
    """
    
    # Generate classification
    response = await generate_text(
        client=client,
        prompt=prompt,
        model=model,
        max_tokens=50,
        temperature=temperature,
        tracer=tracer
    )
    
    # Extract category from response
    classification = response.get("text", "").strip()
    
    # Check if classification is in categories
    if classification not in categories:
        # Try to match with a category
        for category in categories:
            if category.lower() in classification.lower():
                classification = category
                break
        else:
            # Still not found, use the first category
            classification = categories[0] if categories else "unknown"
    
    return {
        "category": classification,
        "confidence": 1.0 if temperature == 0 else 0.8,  # Simplified confidence
        "model": response.get("model", model)
    }

async def extract_entities(
    client: Any,
    text: str,
    entity_types: Optional[List[str]] = None,
    model: str = "claude",
    tracer: Optional[LangfuseTracer] = None
) -> List[Dict[str, Any]]:
    """
    Extract entities from text.
    
    Args:
        client: Bedrock client
        text: Text to extract entities from
        entity_types: Optional list of entity types to extract
        model: Model to use
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        List of extracted entities
    """
    # Set default entity types if not provided
    if not entity_types:
        entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "QUANTITY"]
    
    # Create entity types list
    entity_types_list = ", ".join(entity_types)
    
    # Create prompt for entity extraction
    prompt = f"""
    Extract all entities of the following types from the text: {entity_types_list}
    
    Text: "{text}"
    
    Format your response as a JSON array of objects with the following structure:
    [
      {{
        "entity": "The entity text",
        "type": "The entity type",
        "start": start_position,
        "end": end_position
      }}
    ]
    
    Only return the JSON array, nothing else.
    """
    
    # Generate response
    response = await generate_text(
        client=client,
        prompt=prompt,
        model=model,
        max_tokens=1000,
        temperature=0.1,
        tracer=tracer
    )
    
    # Extract JSON from response
    response_text = response.get("text", "").strip()
    
    try:
        # Try to parse as JSON
        entities = json.loads(response_text)
        
        # Ensure result is a list
        if not isinstance(entities, list):
            return []
        
        return entities
    except json.JSONDecodeError:
        # Failed to parse as JSON, try to extract JSON part
        try:
            # Look for JSON array start and end
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_part = response_text[start_idx:end_idx]
                entities = json.loads(json_part)
                
                # Ensure result is a list
                if not isinstance(entities, list):
                    return []
                
                return entities
        except (json.JSONDecodeError, ValueError):
            # Still failed, return empty list
            return []
        
        return []
_model_info(model)
        model_id = model_info["id"]
        
        # Format prompt according to the model's expected format
        formatted_prompt = bedrock_models.format_prompt(model, prompt, system_prompt)
        
        # Add generation parameters for Claude
        if model_info["provider"] == "anthropic":
            formatted_prompt["max_tokens_to_sample"] = max_tokens
            formatted_prompt["temperature"] = temperature
            formatted_prompt["top_p"] = top_p
            
            if stop_sequences:
                formatted_prompt["stop_sequences"] = stop_sequences
            else:
                formatted_prompt["stop_sequences"] = ["\n\nHuman:"]
        
        # Invoke model
        response = await invoke_model(
            client=client,
            model_id=model_id,
            body=formatted_prompt,
            tracer=tracer
        )
        
        # Extract text from response
        if model_info["provider"] == "anthropic":
            generated_text = response.get("completion", "")
        elif model_info["provider"] == "amazon":
            generated_text = response.get("results", [{}])[0].get("outputText", "")
        elif model_info["provider"] == "meta":
            generated_text = response.get("generation", "")
        elif model_info["provider"] == "ai21":
            generated_text = response.get("completions", [{}])[0].get("data", {}).get("text", "")
        elif model_info["provider"] == "cohere":
            generated_text = response.get("generations", [{}])[0].get("text", "")
        else:
            generated_text = str(response)
        
        result = {
            "text": generated_text,
            "model": model_id
        }
        
        if span:
            span.update(output={"generated_length": len(generated_text)})
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating text with Claude: {str(e)}")
        
        if span:
            span.update(output={"error": str(e)})
        
        return {"text": f"Error: {str(e)}", "error": str(e)}

async def generate_chat_response(
    client: Any,
    messages: List[Dict[str, str]],
    model: str = "claude",
    max_tokens: int = 500,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Generate a response for a chat conversation.
    
    Args:
        client: Bedrock client
        messages: List of message dictionaries with "role" and "content"
        model: Model to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        system_prompt: Optional system prompt
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Generated chat response
    """
    span = None
    if tracer:
        span = tracer.span(
            name="claude_chat_response",
            input={
                "model": model,
                "message_count": len(messages)
            }
        )
    
    try:
        # Convert messages to Claude prompt format
        claude_prompt = ""
        
        if system_prompt:
            claude_prompt += f"\n\nHuman: <system>\n{system_prompt}\n</system>\n\n"
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            if role == "user" or role == "human":
                claude_prompt += f"\n\nHuman: {content}"
            elif role == "assistant" or role == "ai":
                claude_prompt += f"\n\nAssistant: {content}"
            elif role == "system" and not system_prompt:
                # Only use if no system prompt was provided earlier
                system_prompt = content
        
        # Add final Assistant prompt
        claude_prompt += "\n\nAssistant:"
        
        # Generate response
        response = await generate_text(
            client=client,
            prompt=claude_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=None,  # Already included in claude_prompt
            tracer=tracer
        )
        
        if span:
            span.update(output={"status": "success"})
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        
        if span:
            span.update(output={"error": str(e)})
        
        return {"text": f"Error: {str(e)}", "error": str(e)}

async def generate_stream(
    client: Any,
    prompt: str,
    model: str = "claude",
    max_tokens: int = 500,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Generate a streaming response from Claude.
    
    Args:
        client: Bedrock client
        prompt: Text prompt
        model: Model to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        system_prompt: Optional system prompt
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Dictionary with generated_text and is_complete status
    """
    try:
        # Import BedrockModels to get model ID
        from multiagent.app.tools.bedrock.bedrock_models import BedrockModels
        
        # Get model info
        bedrock_models = BedrockModels()
        model_info = bedrock_models.get