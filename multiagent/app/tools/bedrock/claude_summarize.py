"""
Text summarization with Claude.
Provides methods for summarizing content using Claude.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class ClaudeSummarize:
    """Handles text summarization with Claude."""
    
    def __init__(self):
        """Initialize the Claude summarizer."""
        self.logger = logging.getLogger(__name__)
    
    async def summarize(
        self, 
        text: str, 
        length: str = "medium", 
        style: str = "informative", 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Summarizes text with Claude.
        
        Args:
            text: Text to summarize
            length: Summary length ("short", "medium", "long")
            style: Summarization style ("informative", "concise", "detailed", "bullet", "eli5")
            **kwargs: Additional parameters including:
                - client: Bedrock client (required)
                - model_id: Model ID to use (default: anthropic.claude-3-sonnet-20240229-v1:0)
                - max_tokens: Maximum tokens in response (default based on length)
                - temperature: Temperature for generation (default: 0.1)
                - tracer: Optional LangfuseTracer instance
                - format: Output format (default: "text")
                - focus: Areas to focus on in the summary
                - validate: Whether to validate summary quality (default: False)
        
        Returns:
            Dict containing summary and metadata
        """
        try:
            # Extract parameters
            client = kwargs.get("client")
            if not client:
                raise ValueError("client is required for summarization")
                
            model_id = kwargs.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
            tracer = kwargs.get("tracer")
            format = kwargs.get("format", "text")
            focus = kwargs.get("focus", "")
            validate = kwargs.get("validate", False)
            
            span = None
            if tracer:
                span = tracer.span(
                    name="claude_summarize",
                    input={
                        "text_length": len(text),
                        "summary_length": length,
                        "style": style,
                        "model_id": model_id
                    }
                )
            
            # Determine token limits based on requested length
            token_limits = {
                "short": 150,
                "medium": 350,
                "long": 750
            }
            
            # Get max_tokens, defaulting to the appropriate limit if not specified
            max_tokens = kwargs.get("max_tokens", token_limits.get(length, 350))
            temperature = kwargs.get("temperature", 0.1)  # Lower temp for more focused summaries
            
            # Create summarization prompt
            prompt = self._create_summarization_prompt(
                text=text,
                length=length,
                style=style,
                format=format,
                focus=focus
            )
            
            # Import generate_text
            from multiagent.app.tools.bedrock.claude_generate import generate_text
            
            # Generate summary
            response = await generate_text(
                client=client,
                prompt=prompt,
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                tracer=tracer
            )
            
            summary = response.get("text", "").strip()
            
            # Validate summary if requested
            validation_result = None
            if validate:
                validation_result = await self._validate_summary(
                    client=client,
                    original_text=text,
                    summary=summary,
                    model=model_id,
                    tracer=tracer
                )
            
            # Process summary based on format
            processed_summary = self._process_summary(summary, format)
            
            # Prepare result
            result = {
                "summary": processed_summary,
                "length": length,
                "style": style,
                "format": format,
                "model_id": model_id,
                "char_count": len(processed_summary),
                "word_count": len(processed_summary.split())
            }
            
            # Add validation result if available
            if validation_result:
                result["validation"] = validation_result
            
            if span:
                span.update(output={
                    "summary_char_count": len(processed_summary),
                    "summary_word_count": len(processed_summary.split())
                })
            
            return result
            
        except Exception as e:
            error_message = f"Error in Claude summarization: {str(e)}"
            self.logger.error(error_message)
            
            if span:
                span.update(output={"error": str(e)})
            
            return {
                "error": error_message
            }
    
    def _create_summarization_prompt(
        self, 
        text: str, 
        length: str = "medium", 
        style: str = "informative",
        format: str = "text",
        focus: str = ""
    ) -> str:
        """
        Create a prompt for Claude summarization.
        
        Args:
            text: Text to summarize
            length: Summary length
            style: Summarization style
            format: Output format
            focus: Areas to focus on
            
        Returns:
            Formatted prompt
        """
        # Map length to word count guidelines
        length_guidelines = {
            "short": "approximately 75-100 words",
            "medium": "approximately 150-200 words",
            "long": "approximately 300-400 words"
        }
        
        word_count = length_guidelines.get(length, "an appropriate length")
        
        # Style-specific instructions
        style_instructions = {
            "informative": "Create an informative summary that captures the main points and key details.",
            "concise": "Create a concise summary that focuses only on the most essential information.",
            "detailed": "Create a detailed summary that preserves important nuances and supporting evidence.",
            "bullet": "Create a bulleted summary that breaks down the key points in a structured format.",
            "eli5": "Create a summary that explains the content in simple terms, as if explaining to a non-expert."
        }
        
        style_instruction = style_instructions.get(style, style_instructions["informative"])
        
        # Format-specific instructions
        format_instructions = {
            "text": "Provide the summary as continuous text.",
            "bullet": "Provide the summary as a bulleted list of key points.",
            "outline": "Provide the summary as a hierarchical outline with main points and supporting details.",
            "json": "Provide the summary as a JSON object with 'main_points' and 'supporting_details' arrays."
        }
        
        format_instruction = format_instructions.get(format, format_instructions["text"])
        
        # Focus instruction
        focus_instruction = ""
        if focus:
            focus_instruction = f"Focus particularly on aspects related to: {focus}."
        
        # Claude-specific optimizations
        prompt = f"""
        Human: I need you to summarize the following text. {style_instruction} The summary should be {word_count}. {format_instruction} {focus_instruction}
        
        <text_to_summarize>
        {text}
        </text_to_summarize>
        
        Please create a high-quality summary that:
        1. Captures the most important information and main points
        2. Maintains the original meaning and intent
        3. Is coherent and flows well
        4. Avoids introducing information not present in the original text
        5. Is self-contained and understandable without the original text
        
        Assistant:
        """
        
        return prompt
    
    def _process_summary(self, summary: str, format: str = "text") -> str:
        """
        Process the summary based on requested format.
        
        Args:
            summary: Generated summary
            format: Requested format
            
        Returns:
            Processed summary
        """
        if format == "text":
            # Remove any format artifacts and ensure proper paragraphs
            cleaned = re.sub(r"Summary:", "", summary)
            cleaned = re.sub(r"^Here's a summary.*?:", "", cleaned)
            cleaned = cleaned.strip()
            return cleaned
            
        elif format == "bullet":
            # Ensure proper bullet formatting
            if not any(line.strip().startswith(("•", "-", "*")) for line in summary.split("\n")):
                # Convert to bullets if not already in bullet format
                lines = [line.strip() for line in summary.split("\n") if line.strip()]
                return "\n".join([f"• {line}" for line in lines])
            return summary
            
        elif format == "json":
            # Try to extract JSON if present, or create it
            try:
                # Look for JSON structure in the text
                json_match = re.search(r"\{.*\}", summary, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # Validate JSON
                    json.loads(json_str)
                    return json_str
                else:
                    # Create JSON structure from text
                    paragraphs = [p.strip() for p in summary.split("\n\n") if p.strip()]
                    if len(paragraphs) > 1:
                        main_points = paragraphs[0]
                        supporting_details = "\n\n".join(paragraphs[1:])
                    else:
                        main_points = paragraphs[0]
                        supporting_details = ""
                    
                    json_obj = {
                        "main_points": main_points,
                        "supporting_details": supporting_details
                    }
                    return json.dumps(json_obj, indent=2)
            except:
                # Fall back to text if JSON processing fails
                return summary
                
        else:
            # Default processing
            return summary.strip()
    
    async def _validate_summary(
        self,
        client: Any,
        original_text: str,
        summary: str,
        model: str,
        tracer: Optional[LangfuseTracer] = None
    ) -> Dict[str, Any]:
        """
        Validate the quality of a summary.
        
        Args:
            client: Bedrock client
            original_text: Original text
            summary: Generated summary
            model: Model to use
            tracer: Optional LangfuseTracer instance
            
        Returns:
            Dictionary with validation metrics
        """
        # Import generate_text
        from multiagent.app.tools.bedrock.claude_generate import generate_text
        
        # Create validation prompt
        validation_prompt = f"""
        Human: Evaluate the quality of the following summary against the original text.
        
        Original text:
        {original_text[:5000]}  # Limit to avoid token issues
        
        Summary:
        {summary}
        
        Please rate the summary on the following aspects from 1-10 (10 being perfect):
        1. Accuracy: Does the summary accurately represent the original text without introducing errors?
        2. Completeness: Does the summary include all the main points from the original text?
        3. Conciseness: Is the summary appropriately concise without unnecessary details?
        4. Coherence: Does the summary flow well and make logical sense on its own?
        
        For each aspect, provide a numeric score and a brief explanation. Then provide an overall score.
        
        Format your response as a JSON object with the following structure:
        {{
            "accuracy": {{
                "score": numeric_score,
                "explanation": "explanation"
            }},
            "completeness": {{
                "score": numeric_score,
                "explanation": "explanation"
            }},
            "conciseness": {{
                "score": numeric_score,
                "explanation": "explanation"
            }},
            "coherence": {{
                "score": numeric_score,
                "explanation": "explanation"
            }},
            "overall": numeric_score
        }}
        
        Assistant:
        """
        
        # Generate validation
        validation_response = await generate_text(
            client=client,
            prompt=validation_prompt,
            model=model,
            max_tokens=800,
            temperature=0.1,
            tracer=tracer
        )
        
        validation_text = validation_response.get("text", "").strip()
        
        # Extract JSON
        try:
            # Find JSON in response
            json_match = re.search(r"\{.*\}", validation_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                validation_result = json.loads(json_str)
                return validation_result
            else:
                # Parse scores if JSON not found
                scores = re.findall(r"(\w+):\s*(\d+)", validation_text)
                result = {}
                for aspect, score in scores:
                    result[aspect.lower()] = {"score": int(score)}
                return result
        except:
            # Return simple validation result if parsing fails
            return {
                "validation_text": validation_text,
                "parsing_error": "Could not parse structured validation results"
            }