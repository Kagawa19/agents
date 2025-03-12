"""
OpenAI Tool implementation.
Provides functions for interacting with OpenAI models.
"""

import logging
import time
from typing import Dict, Any, List, Optional

import openai

from app.monitoring.tracer import LangfuseTracer
from app.monitoring.metrics import track_llm_call


logger = logging.getLogger(__name__)


class OpenAITool:
    """
    Tool for interacting with OpenAI models.
    Provides functions for text generation, summarization, and analysis.
    """
    
    def __init__(self, api_key: str, tracer: LangfuseTracer):
        """
        Initialize the OpenAI tool.
        
        Args:
            api_key: OpenAI API key
            tracer: LangfuseTracer instance for monitoring
        """
        self.api_key = api_key
        self.tracer = tracer
        self.client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAITool initialized")
    
    def process(
        self, 
        prompt: str, 
        model: str = "gpt-4", 
        temperature: float = 0.7, 
        max_tokens: int = 1000,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a prompt using an OpenAI model.
        
        Args:
            prompt: Prompt to process
            model: OpenAI model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            system_message: Optional system message to set context
            
        Returns:
            Dictionary containing the generated response
        """
        with self.tracer.span("openai_process"):
            start_time = time.time()
            
            try:
                # Prepare messages
                messages = []
                
                # Add system message if provided
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                
                # Add user message
                messages.append({"role": "user", "content": prompt})
                
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract result
                result = {
                    "text": response.choices[0].message.content,
                    "model": model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "finish_reason": response.choices[0].finish_reason
                }
                
                # Calculate execution time
                execution_time = time.time() - start_time
                result["execution_time"] = execution_time
                
                # Track metrics
                track_llm_call(
                    provider="openai",
                    model=model,
                    tokens=response.usage.total_tokens,
                    call_type="chat",
                    latency=execution_time
                )
                
                # Log generation
                self.tracer.log_generation(
                    model=model,
                    prompt=prompt,
                    response=result["text"],
                    metadata={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "execution_time": execution_time,
                        "token_usage": result["usage"]
                    }
                )
                
                return result
            except Exception as e:
                logger.error(f"Error processing with OpenAI: {str(e)}")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_llm_call(
                    provider="openai",
                    model=model,
                    tokens=0,
                    call_type="error",
                    latency=execution_time
                )
                
                # Log event
                self.tracer.log_event(
                    event_type="openai_error",
                    event_data={
                        "error": str(e),
                        "model": model,
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error processing with OpenAI: {str(e)}",
                    "model": model,
                    "execution_time": execution_time
                }
    
    def summarize(
        self, 
        text: str, 
        max_length: int = 200, 
        model: str = "gpt-4",
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate a summary of the provided text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary in words
            model: OpenAI model to use
            temperature: Temperature parameter for generation
            
        Returns:
            Dictionary containing the summary
        """
        with self.tracer.span("openai_summarize"):
            # Construct prompt for summarization
            system_message = f"""
            You are an expert at creating concise, informative summaries. 
            Your task is to summarize the provided text clearly and accurately.
            Focus on the most important information and key insights.
            The summary should be no longer than {max_length} words.
            """
            
            prompt = f"""
            Please summarize the following text:
            
            {text}
            """
            
            # Process with OpenAI
            return self.process(
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_message=system_message,
                # Use a reasonable max_tokens based on max_length
                max_tokens=max(100, max_length * 2)
            )
    
    def analyze(
        self, 
        text: str, 
        question: Optional[str] = None, 
        model: str = "gpt-4",
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Analyze text to extract insights and information.
        
        Args:
            text: Text to analyze
            question: Specific question to answer about the text (optional)
            model: OpenAI model to use
            temperature: Temperature parameter for generation
            
        Returns:
            Dictionary containing the analysis
        """
        with self.tracer.span("openai_analyze"):
            # Construct system message
            system_message = """
            You are an expert analyst with deep knowledge across many domains.
            Your task is to analyze the provided text thoroughly and provide insightful observations.
            Extract key information, identify patterns, and highlight important insights.
            Structure your response clearly, using sections if appropriate.
            """
            
            # Construct prompt based on whether a question was provided
            if question:
                prompt = f"""
                Please analyze the following text and answer this specific question: {question}
                
                TEXT:
                {text}
                """
            else:
                prompt = f"""
                Please analyze the following text and extract key insights, main points, and important information.
                
                TEXT:
                {text}
                """
            
            # Process with OpenAI
            return self.process(
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_message=system_message,
                # Use a large max_tokens for analysis
                max_tokens=1500
            )
    
    def extract_entities(
        self, 
        text: str, 
        entity_types: Optional[List[str]] = None, 
        model: str = "gpt-4",
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to extract entities from
            entity_types: List of entity types to extract (optional)
            model: OpenAI model to use
            temperature: Temperature parameter for generation
            
        Returns:
            Dictionary containing extracted entities
        """
        with self.tracer.span("openai_extract_entities"):
            # Construct system message
            system_message = """
            You are an expert at entity extraction and information retrieval.
            Your task is to extract named entities from the provided text with high accuracy.
            Return the results as JSON with entity types as keys and lists of entities as values.
            """
            
            # Construct prompt based on entity types
            if entity_types:
                entity_types_str = ", ".join(entity_types)
                prompt = f"""
                Extract the following types of named entities from the text: {entity_types_str}.
                Format the output as a JSON object with entity types as keys and lists of entities as values.
                
                TEXT:
                {text}
                
                ENTITIES (JSON format):
                """
            else:
                prompt = f"""
                Extract all named entities from the text.
                Format the output as a JSON object with entity types as keys and lists of entities as values.
                Common entity types include: Person, Organization, Location, Date, Product, Event.
                
                TEXT:
                {text}
                
                ENTITIES (JSON format):
                """
            
            # Process with OpenAI
            return self.process(
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_message=system_message
            )