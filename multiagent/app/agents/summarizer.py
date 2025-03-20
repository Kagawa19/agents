"""
Summarizer agent implementation.
Responsible for summarizing content from various sources and generating concise reports.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import time
import json
import asyncio
from datetime import datetime

from multiagent.app.agents.base import BaseAgent
from multiagent.app.monitoring.tracer import LangfuseTracer


logger = logging.getLogger(__name__)


class SummarizerAgent(BaseAgent):
    """
    Agent for summarizing content and generating concise reports.
    Can use either OpenAI or Bedrock (Claude) for summarization tasks.
    """
    
    def __init__(
        self,
        agent_id: str,
        tracer: LangfuseTracer,
        openai_tool: Any,
        bedrock_tool: Optional[Any] = None,
    ):
        """
        Initialize the summarizer agent.
        
        Args:
            agent_id: Unique identifier for the agent
            tracer: LangfuseTracer instance for monitoring
            openai_tool: OpenAI tool for summarization capabilities
            bedrock_tool: Bedrock tool for alternative AI provider (Claude)
        """
        super().__init__(agent_id=agent_id, tracer=tracer)
        self.openai_tool = openai_tool
        self.bedrock_tool = bedrock_tool
        self.config = {}
        self.initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the summarizer with configuration parameters.
        
        Args:
            config: Configuration parameters for the summarizer
        """
        self.config = config
        
        # Set up summarizer parameters
        self.max_summary_length = config.get("max_summary_length", 1000)
        self.min_summary_length = config.get("min_summary_length", 200)
        self.default_temperature = config.get("temperature", 0.3)
        
        # AI provider preferences
        self.preferred_ai_provider = config.get("preferred_ai_provider", "auto")  # "auto", "openai", or "bedrock"
        
        # Set initialized flag
        self.initialized = True
        
        logger.info(f"Initialized {self.agent_id} with config: {json.dumps(config, default=str)}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate the input data before execution.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check for either text or documents
        if "text" not in input_data and "documents" not in input_data:
            logger.error(f"Input data must contain either 'text' or 'documents' field")
            return False
            
        # Check if summary type is valid
        if "summary_type" in input_data:
            valid_types = ["concise", "detailed", "bullet_points", "executive", "technical"]
            if input_data["summary_type"] not in valid_types:
                logger.warning(f"Invalid summary_type: {input_data['summary_type']}. Using default: 'concise'")
                
        return True
    
    async def _get_ai_provider(self, task_type: str = "summary") -> tuple:
        """
        Select the appropriate AI provider based on settings and availability.
        
        Args:
            task_type: The type of task being performed (default: "summary")
            
        Returns:
            Tuple of (ai_tool, provider_name)
            
        Raises:
            ValueError: If no AI tool is available
        """
        # Check for specific task preferences in config
        task_preference = self.config.get(f"preferred_provider_{task_type}", self.preferred_ai_provider)
        
        # Logic for provider selection
        if task_preference == "bedrock" and self.bedrock_tool:
            return self.bedrock_tool, "bedrock"
        elif task_preference == "openai" and self.openai_tool:
            return self.openai_tool, "openai"
        elif task_preference == "auto":
            # For summarization, Claude (Bedrock) generally produces better results
            if self.bedrock_tool:
                return self.bedrock_tool, "bedrock"
            elif self.openai_tool:
                return self.openai_tool, "openai"
        
        # Fall back to whatever is available
        if self.openai_tool:
            return self.openai_tool, "openai"
        elif self.bedrock_tool:
            return self.bedrock_tool, "bedrock"
        
        # No tool available
        raise ValueError(f"No AI tool available for summarization")
    
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 1000, 
        temperature: float = 0.3,
        task_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Generate text using the appropriate AI provider.
        
        Args:
            prompt: The text prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            task_type: Type of task for provider selection
            
        Returns:
            Response containing the generated text
        """
        ai_tool, provider = await self._get_ai_provider(task_type)
        
        try:
            if provider == "bedrock":
                # Use Claude via Bedrock
                response = await ai_tool.generate_text(
                    model="claude",  # Specify Claude model
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                # Use OpenAI
                response = await ai_tool.generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            return response
        except Exception as e:
            logger.error(f"Error generating text with {provider}: {str(e)}")
            
            # Try fallback if available
            if provider == "bedrock" and self.openai_tool:
                logger.info("Falling back to OpenAI")
                return await self.openai_tool.generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            elif provider == "openai" and self.bedrock_tool:
                logger.info("Falling back to Bedrock")
                return await self.bedrock_tool.generate_text(
                    model="claude",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                # Re-raise if no fallback available
                raise
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the summarization task.
        
        Args:
            input_data: Dictionary containing:
                - text: Text to summarize (optional if documents provided)
                - documents: List of documents to summarize (optional if text provided)
                - query: The query or context for summarization (optional)
                - summary_type: Type of summary to generate (default: "concise")
                - max_length: Maximum length of the summary (default: from config)
                - min_length: Minimum length of the summary (default: from config)
                - temperature: Temperature for generation (default: from config)
                
        Returns:
            Dictionary containing:
                - summary: The generated summary
                - metadata: Information about the summarization process
                - processing_time: Time taken for summarization
        """
        start_time = time.time()
        logger.info(f"Starting summarization task...")
        
        # Track execution with a trace span
        with self.tracer.span(name="summarizer_execute") as span:
            span.update(input=input_data)
            
            try:
                # Process input data
                text_to_summarize = self._prepare_text_for_summarization(input_data)
                
                # Get summarization parameters
                summary_type = input_data.get("summary_type", "concise")
                max_length = input_data.get("max_length", self.max_summary_length)
                min_length = input_data.get("min_length", self.min_summary_length)
                temperature = input_data.get("temperature", self.default_temperature)
                context = input_data.get("query", None)
                
                # Generate summary based on type
                if summary_type == "bullet_points":
                    summary = await self._generate_bullet_point_summary(
                        text_to_summarize, 
                        context,
                        max_length,
                        temperature
                    )
                elif summary_type == "executive":
                    summary = await self._generate_executive_summary(
                        text_to_summarize, 
                        context,
                        max_length,
                        temperature
                    )
                elif summary_type == "technical":
                    summary = await self._generate_technical_summary(
                        text_to_summarize, 
                        context,
                        max_length,
                        temperature
                    )
                elif summary_type == "detailed":
                    summary = await self._generate_detailed_summary(
                        text_to_summarize, 
                        context,
                        max_length,
                        temperature
                    )
                else: # Default to concise
                    summary = await self._generate_concise_summary(
                        text_to_summarize, 
                        context,
                        max_length,
                        temperature
                    )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Calculate metadata about the summary
                word_count_original = len(text_to_summarize.split())
                word_count_summary = len(summary.split())
                compression_ratio = word_count_summary / word_count_original if word_count_original > 0 else 0
                
                # Build response
                result = {
                    "summary": summary,
                    "metadata": {
                        "summary_type": summary_type,
                        "original_word_count": word_count_original,
                        "summary_word_count": word_count_summary,
                        "compression_ratio": compression_ratio,
                        "summarized_at": datetime.utcnow().isoformat()
                    },
                    "processing_time": processing_time,
                    "status": "completed"
                }
                
                logger.info(f"Summarization completed in {processing_time:.2f} seconds")
                span.update(output=result)
                return result
                
            except Exception as e:
                error_msg = f"Error in summarizer execution: {str(e)}"
                logger.error(error_msg)
                span.update(output={"status": "error", "error": error_msg})
                
                # Return error result
                return {
                    "status": "error",
                    "error": error_msg,
                    "processing_time": time.time() - start_time
                }
    
    def _prepare_text_for_summarization(self, input_data: Dict[str, Any]) -> str:
        """
        Prepare text from input data for summarization.
        
        Args:
            input_data: Input data containing text or documents
            
        Returns:
            Prepared text for summarization
        """
        # Check if we have direct text
        if "text" in input_data and input_data["text"]:
            return input_data["text"]
        
        # Process documents
        if "documents" in input_data and input_data["documents"]:
            documents = input_data["documents"]
            combined_text = ""
            
            # Combine documents into a single text
            for doc in documents:
                if isinstance(doc, str):
                    combined_text += doc + "\n\n"
                elif isinstance(doc, dict):
                    # Extract content from document dict
                    if "content" in doc:
                        combined_text += doc["content"] + "\n\n"
                    elif "text" in doc:
                        combined_text += doc["text"] + "\n\n"
                    else:
                        # Try to use the whole doc as text
                        combined_text += str(doc) + "\n\n"
            
            return combined_text
        
        return ""
    
    async def _generate_concise_summary(
        self, 
        text: str, 
        context: Optional[str] = None,
        max_length: int = 1000,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a concise summary of the text.
        
        Args:
            text: Text to summarize
            context: Optional context or query
            max_length: Maximum length in words
            temperature: Temperature for generation
            
        Returns:
            Concise summary
        """
        prompt = self._create_summary_prompt(
            text=text,
            context=context,
            instructions=f"""
            Create a concise summary of the following text.
            
            Guidelines:
            - Focus on the most important information and key points
            - Use clear, straightforward language
            - Be objective and accurate
            - Keep the summary between {self.min_summary_length} and {max_length} words
            - Maintain a logical flow of ideas
            - Avoid unnecessary details and examples
            """,
            max_length=max_length
        )
        
        response = await self.generate_text(
            prompt=prompt,
            max_tokens=int(max_length * 1.5),  # Allow some buffer
            temperature=temperature
        )
        
        return response.get("text", "")
    
    async def _generate_bullet_point_summary(
        self, 
        text: str, 
        context: Optional[str] = None,
        max_length: int = 1000,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a bullet point summary of the text.
        
        Args:
            text: Text to summarize
            context: Optional context or query
            max_length: Maximum length in words
            temperature: Temperature for generation
            
        Returns:
            Bullet point summary
        """
        prompt = self._create_summary_prompt(
            text=text,
            context=context,
            instructions=f"""
            Create a bullet-point summary of the following text.
            
            Guidelines:
            - Extract 7-12 key points from the text
            - Use bullet points (•) for each main point
            - Keep each bullet point clear and concise
            - Organize points in a logical sequence
            - Include only the most significant information
            - The entire summary should not exceed {max_length} words
            - Use sub-bullets where appropriate to organize related information
            """,
            max_length=max_length
        )
        
        response = await self.generate_text(
            prompt=prompt,
            max_tokens=int(max_length * 1.5),
            temperature=temperature
        )
        
        return response.get("text", "")
    
    async def _generate_executive_summary(
        self, 
        text: str, 
        context: Optional[str] = None,
        max_length: int = 1000,
        temperature: float = 0.3
    ) -> str:
        """
        Generate an executive summary of the text.
        
        Args:
            text: Text to summarize
            context: Optional context or query
            max_length: Maximum length in words
            temperature: Temperature for generation
            
        Returns:
            Executive summary
        """
        prompt = self._create_summary_prompt(
            text=text,
            context=context,
            instructions=f"""
            Create an executive summary of the following text.
            
            Guidelines:
            - Focus on strategic implications, key findings, and actionable insights
            - Begin with an overview of the most important conclusion
            - Include any critical data points, metrics, or KPIs
            - Highlight business impact and recommendations
            - Use professional, concise language appropriate for executives
            - Keep the summary between {self.min_summary_length} and {max_length} words
            - Structure with clear sections if appropriate
            - Prioritize information relevant to business decision-making
            """,
            max_length=max_length
        )
        
        response = await self.generate_text(
            prompt=prompt,
            max_tokens=int(max_length * 1.5),
            temperature=temperature
        )
        
        return response.get("text", "")
    
    async def _generate_technical_summary(
        self, 
        text: str, 
        context: Optional[str] = None,
        max_length: int = 1000,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a technical summary of the text.
        
        Args:
            text: Text to summarize
            context: Optional context or query
            max_length: Maximum length in words
            temperature: Temperature for generation
            
        Returns:
            Technical summary
        """
        prompt = self._create_summary_prompt(
            text=text,
            context=context,
            instructions=f"""
            Create a technical summary of the following text.
            
            Guidelines:
            - Preserve technical accuracy and important details
            - Use precise technical terminology appropriate to the domain
            - Include key methodologies, specifications, or technical findings
            - Organize into logical sections (e.g., Background, Methods, Results, Implications)
            - Maintain technical rigor while being concise
            - Keep the summary between {self.min_summary_length} and {max_length} words
            - Include critical technical data points, measurements, or specifications
            - Use appropriate technical format (e.g., citations, references to equations)
            """,
            max_length=max_length
        )
        
        response = await self.generate_text(
            prompt=prompt,
            max_tokens=int(max_length * 1.5),
            temperature=temperature
        )
        
        return response.get("text", "")
    
    async def _generate_detailed_summary(
        self, 
        text: str, 
        context: Optional[str] = None,
        max_length: int = 1000,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a detailed summary of the text.
        
        Args:
            text: Text to summarize
            context: Optional context or query
            max_length: Maximum length in words
            temperature: Temperature for generation
            
        Returns:
            Detailed summary
        """
        prompt = self._create_summary_prompt(
            text=text,
            context=context,
            instructions=f"""
            Create a detailed summary of the following text.
            
            Guidelines:
            - Provide a comprehensive overview of the content
            - Include all major points, findings, and conclusions
            - Organize into clear sections with logical flow
            - Preserve important details, examples, and supporting evidence
            - Maintain nuance and complexity where necessary
            - Keep the summary under {max_length} words while being thorough
            - Include relevant context and background information
            - Use appropriate transitions between ideas and sections
            """,
            max_length=max_length
        )
        
        response = await self.generate_text(
            prompt=prompt,
            max_tokens=int(max_length * 1.5),
            temperature=temperature
        )
        
        return response.get("text", "")
    
    def _create_summary_prompt(
        self,
        text: str,
        instructions: str,
        max_length: int,
        context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for summarization.
        
        Args:
            text: Text to summarize
            instructions: Specific instructions for the summary
            max_length: Maximum length in words
            context: Optional context or query
            
        Returns:
            Formatted prompt
        """
        # Add context if provided
        context_section = ""
        if context:
            context_section = f"""
            Context/Query: {context}
            
            When summarizing, pay special attention to information relevant to the context or query above.
            """
        
        # Calculate appropriate chunk size for long texts
        text_length = len(text)
        if text_length > 20000:
            # Very long text - only use a portion
            # Use first 25%, middle 50%, and last 25%
            chunk_size = min(10000, text_length // 4)
            text_summary = (
                f"{text[:chunk_size]}\n\n"
                f"[... middle portion with {text_length - 2*chunk_size} characters omitted for length ...]\n\n"
                f"{text[-chunk_size:]}"
            )
        else:
            text_summary = text
        
        # Create the prompt
        prompt = f"""
        {instructions}
        
        {context_section}
        
        Text to summarize:
        {text_summary}
        
        Summary (maximum {max_length} words):
        """
        
        return prompt