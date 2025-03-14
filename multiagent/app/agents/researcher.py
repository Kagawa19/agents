"""
Researcher agent implementation.
Responsible for gathering information from web sources.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from multiagent.app.agents.base import BaseAgent
from multiagent.app.monitoring.tracer import LangfuseTracer
from multiagent.app.tools.serper import SerperTool
from multiagent.app.tools.openai import OpenAITool


logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """
    Agent responsible for researching information from web sources.
    Uses Serper API to perform web searches and extract relevant information.
    """
    
    def __init__(
        self,
        agent_id: str,
        tracer: LangfuseTracer,
        serper_tool: SerperTool,
        openai_tool: Optional[OpenAITool] = None
    ):
        """
        Initialize the researcher agent.
        
        Args:
            agent_id: Unique identifier for the agent
            tracer: LangfuseTracer instance for monitoring
            serper_tool: SerperTool instance for web searching
            openai_tool: Optional OpenAITool for additional processing
        """
        super().__init__(agent_id, tracer)
        self.serper_tool = serper_tool
        self.openai_tool = openai_tool
        self.research_config = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the researcher agent with configuration parameters.
        
        Args:
            config: Configuration parameters including research settings
        """
        # Store research configuration
        self.research_config = {
            "temperature": config.get("temperature", 0.7),
            "model": config.get("model", "gpt-4"),
            "max_search_results": config.get("max_search_results", 5),
            "include_news": config.get("include_news", False),
            "prompt_template": config.get(
                "prompt_template", 
                "You are a research agent tasked with finding information about {query}. "
                "Synthesize and summarize information from multiple sources."
            )
        }
        
        # Mark as initialized
        self.initialized = True
        
        logger.info(f"ResearcherAgent {self.agent_id} initialized with config: {self.research_config}")
    
    def _process_search_results(self, results: Dict[str, Any]) -> str:
        """
        Process and format search results into a readable summary.
        
        Args:
            results: Search results from Serper API
            
        Returns:
            Formatted string of research results
        """
        # Check for errors
        if "error" in results:
            return f"Error in search: {results['error']}"
        
        # Extract processed results
        processed_results = results.get("results", {})
        output = []
        
        # Add answer box if available
        if processed_results.get("answer_box"):
            answer_box = processed_results["answer_box"]
            output.append("🔍 Highlighted Answer:")
            output.append(f"Title: {answer_box.get('title', 'N/A')}")
            output.append(f"Answer: {answer_box.get('answer', 'No direct answer')}")
            output.append(f"Snippet: {answer_box.get('snippet', 'No additional context')}")
            output.append("\n")
        
        # Add organic search results
        if processed_results.get("organic"):
            output.append("📚 Top Search Results:")
            for idx, result in enumerate(processed_results["organic"][:5], 1):
                output.append(f"{idx}. {result.get('title', 'Untitled')}")
                output.append(f"   Link: {result.get('link', 'No link')}")
                output.append(f"   Snippet: {result.get('snippet', 'No description')}")
                output.append("")
        
        # Add related searches
        related_searches = processed_results.get("related_searches", [])
        if related_searches:
            output.append("🔗 Related Searches:")
            output.append(", ".join(related_searches[:5]))
        
        return "\n".join(output)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research process.
        
        Args:
            input_data: Input data containing research query
            
        Returns:
            Dictionary with research results
        """
        # Extract query
        query = input_data.get("query")
        if not query:
            return {
                "error": "No research query provided",
                "status": "failed"
            }
        
        try:
            # Perform web search
            search_results = self.serper_tool.search(
                query, 
                num_results=self.research_config.get("max_search_results", 5)
            )
            
            # Process search results
            processed_results = self._process_search_results(search_results)
            
            # Optional: Use OpenAI to synthesize results if available
            final_output = processed_results
            if self.openai_tool:
                try:
                    synthesis_result = self.openai_tool.analyze(
                        text=processed_results,
                        question=f"Provide a concise and comprehensive summary of the research findings about: {query}",
                        model=self.research_config.get("model", "gpt-4"),
                        temperature=self.research_config.get("temperature", 0.7)
                    )
                    
                    # Use synthesized text if successful
                    if "text" in synthesis_result:
                        final_output = synthesis_result["text"]
                except Exception as synthesis_error:
                    logger.warning(f"Result synthesis failed: {synthesis_error}")
            
            # Optional: News search if configured
            news_results = None
            if self.research_config.get("include_news", False):
                try:
                    news_search = self.serper_tool.news_search(query)
                    if "results" in news_search:
                        news_results = news_search["results"]
                except Exception as news_error:
                    logger.warning(f"News search failed: {news_error}")
            
            return {
                "query": query,
                "research_results": final_output,
                "raw_search_results": search_results,
                "news_results": news_results,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Research execution error: {str(e)}")
            return {
                "error": f"Research failed: {str(e)}",
                "query": query,
                "status": "failed"
            }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for the research agent.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check if query is present and is a non-empty string
        return (
            isinstance(input_data, dict) and 
            "query" in input_data and 
            isinstance(input_data["query"], str) and 
            input_data["query"].strip() != ""
        )