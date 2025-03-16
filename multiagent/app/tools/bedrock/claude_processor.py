"""
Claude-specific processing.
Provides methods for processing text with Claude models.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

async def process_structured_output(
    client: Any,
    prompt: str,
    output_format: Dict[str, Any],
    model: str = "claude",
    temperature: float = 0.2,
    max_tokens: int = 1000,
    system_prompt: Optional[str] = None,
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Get structured output from Claude according to a specified format.
    
    Args:
        client: Bedrock client
        prompt: Text prompt
        output_format: Dictionary describing the expected output format
        model: Model to use
        temperature: Temperature for sampling
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Structured output according to the specified format
    """
    span = None
    if tracer:
        span = tracer.span(
            name="claude_structured_output",
            input={
                "prompt_length": len(prompt),
                "model": model
            }
        )
    
    try:
        # Convert output format to a JSON schema
        format_json = json.dumps(output_format, indent=2)
        
        # Create complete prompt with format instructions
        complete_prompt = f"""
        {prompt}
        
        Please provide your response in the following JSON format exactly, with no additional text, explanation, or conversation:
        {format_json}
        
        Return only the JSON object described above, nothing else.
        """
        
        # Import generate_text
        from multiagent.app.tools.bedrock.claude_generate import generate_text
        
        # Generate response
        response = await generate_text(
            client=client,
            prompt=complete_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            tracer=tracer
        )
        
        # Extract JSON from response
        response_text = response.get("text", "").strip()
        
        try:
            # Try to parse as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Failed to parse as JSON, try to extract JSON part
            try:
                # Look for JSON object start and end
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_part = response_text[start_idx:end_idx]
                    return json.loads(json_part)
            except (json.JSONDecodeError, ValueError):
                # Still failed, return a simplified result
                if span:
                    span.update(output={"error": "Failed to parse JSON response"})
                
                return {
                    "response": response_text,
                    "error": "Failed to parse as structured output"
                }
        
    except Exception as e:
        logger.error(f"Error in structured output: {str(e)}")
        
        if span:
            span.update(output={"error": str(e)})
        
        return {"error": str(e)}

async def extract_json(
    client: Any,
    text: str,
    model: str = "claude",
    max_attempts: int = 3,
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Extract a JSON object from text.
    
    Args:
        client: Bedrock client
        text: Text containing JSON
        model: Model to use
        max_attempts: Maximum number of attempts
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Extracted JSON object
    """
    span = None
    if tracer:
        span = tracer.span(
            name="claude_extract_json",
            input={"text_length": len(text)}
        )
    
    try:
        # Try direct JSON parsing first
        try:
            # Check if the text is already valid JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # Not valid JSON, continue with extraction
            pass
        
        # Create prompt for JSON extraction
        prompt = f"""
        Extract the JSON object from the following text. If there are multiple JSON objects, extract the most complete one.
        Return only the JSON object, nothing else.
        
        Text:
        {text}
        
        Extracted JSON:
        """
        
        # Import generate_text
        from multiagent.app.tools.bedrock.claude_generate import generate_text
        
        # Try multiple attempts
        for attempt in range(max_attempts):
            # Generate response
            response = await generate_text(
                client=client,
                prompt=prompt,
                model=model,
                max_tokens=2000,
                temperature=0.1,
                tracer=tracer
            )
            
            response_text = response.get("text", "").strip()
            
            try:
                # Try to parse as JSON
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Failed to parse as JSON, try to extract JSON part
                try:
                    # Look for JSON object start and end
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}") + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_part = response_text[start_idx:end_idx]
                        return json.loads(json_part)
                except (json.JSONDecodeError, ValueError):
                    # Try again in next attempt or return error
                    if attempt == max_attempts - 1:
                        if span:
                            span.update(output={"error": "Failed to extract valid JSON"})
                        
                        return {
                            "error": "Failed to extract valid JSON",
                            "text": response_text
                        }
        
        # Should not reach here
        return {"error": "Failed to extract JSON after maximum attempts"}
        
    except Exception as e:
        logger.error(f"Error extracting JSON: {str(e)}")
        
        if span:
            span.update(output={"error": str(e)})
        
        return {"error": str(e)}

async def format_text(
    client: Any,
    text: str,
    format_type: str,
    model: str = "claude",
    tracer: Optional[LangfuseTracer] = None
) -> str:
    """
    Format text according to a specified format type.
    
    Args:
        client: Bedrock client
        text: Text to format
        format_type: Type of formatting (markdown, html, etc.)
        model: Model to use
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Formatted text
    """
    # Normalize format type
    format_type = format_type.lower()
    
    # Create prompt for formatting
    prompt = f"""
    Format the following text as {format_type}.
    Only return the formatted text, no additional explanation.
    
    Text to format:
    {text}
    
    {format_type.upper()} formatted result:
    """
    
    # Import generate_text
    from multiagent.app.tools.bedrock.claude_generate import generate_text
    
    # Generate formatted text
    response = await generate_text(
        client=client,
        prompt=prompt,
        model=model,
        max_tokens=len(text) * 2,  # Formatted text might be longer
        temperature=0.1,
        tracer=tracer
    )
    
    return response.get("text", "").strip()

async def fix_json(
    client: Any,
    broken_json: str,
    expected_schema: Optional[Dict[str, Any]] = None,
    model: str = "claude",
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Fix broken or malformed JSON.
    
    Args:
        client: Bedrock client
        broken_json: Broken JSON string
        expected_schema: Optional expected schema
        model: Model to use
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Fixed JSON object
    """
    span = None
    if tracer:
        span = tracer.span(
            name="claude_fix_json",
            input={"json_length": len(broken_json)}
        )
    
    try:
        # Try direct JSON parsing first
        try:
            # Check if the JSON is already valid
            return json.loads(broken_json)
        except json.JSONDecodeError:
            # Not valid JSON, continue with fixing
            pass
        
        # Create prompt for JSON fixing
        prompt = f"""
        Fix the following broken JSON. Return only the fixed JSON object, nothing else.
        
        Broken JSON:
        {broken_json}
        """
        
        # Add expected schema if provided
        if expected_schema:
            schema_json = json.dumps(expected_schema, indent=2)
            prompt += f"""
            
            The JSON should conform to the following schema:
            {schema_json}
            """
        
        prompt += """
        
        Fixed JSON:
        """
        
        # Import generate_text
        from multiagent.app.tools.bedrock.claude_generate import generate_text
        
        # Generate fixed JSON
        response = await generate_text(
            client=client,
            prompt=prompt,
            model=model,
            max_tokens=len(broken_json) * 2,
            temperature=0.1,
            tracer=tracer
        )
        
        response_text = response.get("text", "").strip()
        
        try:
            # Try to parse as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Failed to parse as JSON, try to extract JSON part
            try:
                # Look for JSON object start and end
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_part = response_text[start_idx:end_idx]
                    return json.loads(json_part)
            except (json.JSONDecodeError, ValueError):
                # Still failed, return error
                if span:
                    span.update(output={"error": "Failed to fix JSON"})
                
                return {
                    "error": "Failed to fix JSON",
                    "text": response_text
                }
        
    except Exception as e:
        logger.error(f"Error fixing JSON: {str(e)}")
        
        if span:
            span.update(output={"error": str(e)})
        
        return {"error": str(e)}

async def extract_schema(
    client: Any,
    data_sample: Union[str, Dict[str, Any]],
    model: str = "claude",
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Extract a JSON schema from a data sample.
    
    Args:
        client: Bedrock client
        data_sample: Sample data to extract schema from
        model: Model to use
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        JSON schema
    """
    # Convert data sample to string if needed
    if isinstance(data_sample, dict):
        data_sample = json.dumps(data_sample, indent=2)
    
    # Create prompt for schema extraction
    prompt = f"""
    Create a JSON schema that describes the structure of the following data sample.
    Include data types, required fields, and any constraints you can infer.
    Return only the JSON schema, nothing else.
    
    Data sample:
    {data_sample}
    
    JSON schema:
    """
    
    # Import generate_text
    from multiagent.app.tools.bedrock.claude_generate import generate_text
    
    # Generate schema
    response = await generate_text(
        client=client,
        prompt=prompt,
        model=model,
        max_tokens=2000,
        temperature=0.1,
        tracer=tracer
    )
    
    # Try to parse as JSON
    try:
        schema = json.loads(response.get("text", "").strip())
        return schema
    except json.JSONDecodeError:
        # Failed to parse as JSON, try to extract JSON part
        response_text = response.get("text", "").strip()
        
        try:
            # Look for JSON object start and end
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_part = response_text[start_idx:end_idx]
                return json.loads(json_part)
        except (json.JSONDecodeError, ValueError):
            # Still failed, return error
            return {
                "error": "Failed to extract schema",
                "text": response_text
            }

async def extract_tables(
    client: Any,
    text: str,
    model: str = "claude",
    tracer: Optional[LangfuseTracer] = None
) -> List[Dict[str, Any]]:
    """
    Extract tables from text.
    
    Args:
        client: Bedrock client
        text: Text containing tables
        model: Model to use
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        List of extracted tables
    """
    # Create prompt for table extraction
    prompt = f"""
    Extract all tables from the following text. Convert each table to a JSON object with column headers as keys.
    Return an array of tables, where each table is represented as an array of row objects.
    
    Text:
    {text}
    
    Extracted tables:
    """
    
    # Import generate_text
    from multiagent.app.tools.bedrock.claude_generate import generate_text
    
    # Generate response
    response = await generate_text(
        client=client,
        prompt=prompt,
        model=model,
        max_tokens=len(text) * 2,
        temperature=0.1,
        tracer=tracer
    )
    
    # Try to parse as JSON
    response_text = response.get("text", "").strip()
    
    try:
        tables = json.loads(response_text)
        
        # Ensure result is a list
        if not isinstance(tables, list):
            tables = [tables]
        
        return tables
    except json.JSONDecodeError:
        # Failed to parse as JSON, try to extract JSON part
        try:
            # Look for JSON array start and end
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_part = response_text[start_idx:end_idx]
                tables = json.loads(json_part)
                
                # Ensure result is a list
                if not isinstance(tables, list):
                    tables = [tables]
                
                return tables
        except (json.JSONDecodeError, ValueError):
            # Still failed, return empty list
            return []