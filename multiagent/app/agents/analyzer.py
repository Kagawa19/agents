"""
Streamlined Analyzer agent implementation.
Analyzes information using Jina, Bedrock, and web search.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from multiagent.app.agents.base import BaseAgent
from multiagent.app.monitoring.tracer import LangfuseTracer

# Import tools - simplified for clarity
from multiagent.app.tools import serper, scraper, jina, bedrock


logger = logging.getLogger(__name__)


class AnalyzerAgent(BaseAgent):
    """
    Agent for analyzing information and extracting insights.
    Uses Jina AI for semantic analysis and Bedrock (Claude) for insight generation.
    """
    
    def __init__(
        self,
        agent_id: str,
        tracer: LangfuseTracer,
        jina_tool: Any,
        openai_tool: Any,
        bedrock_tool: Optional[Any] = None,
        serper_tool: Optional[Any] = None,
        scraper_tool: Optional[Any] = None
    ):
        """Initialize the analyzer agent with required tools."""
        super().__init__(agent_id=agent_id, tracer=tracer)
        self.jina_tool = jina_tool
        self.openai_tool = openai_tool
        self.bedrock_tool = bedrock_tool
        self.serper_tool = serper_tool
        self.scraper_tool = scraper_tool
        self.config = {}
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration parameters."""
        self.config = config
        logger.info(f"Initialized {self.agent_id}")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analysis on provided information or search for data.
        
        Args:
            input_data: Dictionary containing:
                - query: The search/analysis query
                - information: Optional pre-gathered information
                - search_web: Boolean indicating whether to search the web
                
        Returns:
            Analysis results with insights and patterns
        """
        start_time = time.time()
        query = input_data.get("query", "")
        logger.info(f"Starting analysis for: {query[:50]}...")
        
        with self.tracer.span(name="analyzer_execute") as span:
            try:
                # Determine if we need to search the web
                if input_data.get("search_web", False) and not input_data.get("information"):
                    # Search and scrape if requested
                    information = await self._search_and_scrape(
                        query, 
                        num_results=input_data.get("num_results", 5)
                    )
                else:
                    # Use provided information
                    information = input_data.get("information", [])
                
                if not information:
                    return {
                        "status": "error",
                        "error": "No information provided or found for analysis",
                        "processing_time": time.time() - start_time
                    }
                
                # Process with Jina for semantic analysis
                embeddings, clusters = await self._process_with_jina(information)
                
                # Generate insights using Bedrock/Claude
                insights = await self._generate_insights(query, information, clusters)
                
                # Generate summary using Bedrock/Claude
                summary = await self._generate_summary(query, insights)
                
                # Format and return results
                result = {
                    "analysis_results": {
                        "summary": summary,
                        "insights": insights,
                        "clusters": clusters,
                        "source_count": len(information),
                        "analyzed_at": datetime.utcnow().isoformat()
                    },
                    "processing_time": time.time() - start_time,
                    "status": "completed"
                }
                
                logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
                return result
                
            except Exception as e:
                logger.error(f"Error in analyzer execution: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
    
    async def _search_and_scrape(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web and scrape content for the given query."""
        logger.info(f"Searching for: {query}")
        
        if not self.serper_tool or not self.scraper_tool:
            raise ValueError("Search tools required but not available")
        
        # Search web with SerperTool
        search_results = await self.serper_tool.search(query, num_results=num_results)
        
        # Scrape content from search results
        scraped_content = []
        for result in search_results:
            url = result.get("link")
            if url:
                try:
                    content = await self.scraper_tool.scrape(url)
                    if content:
                        scraped_content.append({
                            "content": content,
                            "metadata": {
                                "url": url,
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", "")
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error scraping {url}: {str(e)}")
        
        logger.info(f"Found {len(scraped_content)} documents")
        return scraped_content
    
    async def _process_with_jina(self, information: List[Dict[str, Any]]) -> tuple:
        """Process information with Jina for embeddings and clustering."""
        logger.info(f"Processing {len(information)} items with Jina")
        
        # Extract text content
        texts = [item.get("content", "") for item in information]
        
        # Get embeddings using Jina
        embeddings = await jina.jina_extract.get_embeddings(self.jina_tool, texts)
        
        # Perform clustering if we have enough documents
        clusters = []
        if len(embeddings) >= 2:
            try:
                # Cluster documents
                cluster_results = await jina.jina_search.cluster_embeddings(
                    self.jina_tool,
                    embeddings,
                    n_clusters=min(5, len(embeddings) // 2) if len(embeddings) > 4 else 2
                )
                
                # Format cluster information
                for cluster_id, indices in cluster_results.items():
                    cluster_texts = [texts[i][:200] for i in indices if i < len(texts)]
                    clusters.append({
                        "id": cluster_id,
                        "document_indices": indices,
                        "document_count": len(indices),
                        "sample_text": cluster_texts[0] if cluster_texts else ""
                    })
            except Exception as e:
                logger.warning(f"Error in clustering: {str(e)}")
        
        return embeddings, clusters
    
    async def _generate_insights(self, query: str, information: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights using Bedrock/Claude."""
        logger.info("Generating insights")
        
        # Combine text samples for analysis
        combined_text = "\n\n".join([item.get("content", "")[:500] for item in information[:5]])
        
        # Create prompt for insight generation
        prompt = f"""
        Analyze the following text related to the query: "{query}"
        
        Text samples:
        {combined_text[:3000]}
        
        Generate 3-5 key insights that:
        1. Provide non-obvious conclusions
        2. Connect multiple pieces of information
        3. Answer the original query directly or indirectly
        4. Include specific supporting evidence
        
        Format each insight with:
        - A clear insight title
        - A detailed explanation
        """
        
        # Use Bedrock/Claude to generate insights
        try:
            if self.bedrock_tool:
                response = await bedrock.claude_generate.generate_text(
                    client=self.bedrock_tool,
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0.4
                )
            else:
                # Fallback to OpenAI
                response = await self.openai_tool.generate_text(
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0.4
                )
            
            # Parse the response into a list of insights
            insights_text = response.get("text", "")
            
            # Simple parsing - split by insight numbers
            insights = []
            current_insight = {}
            current_text = ""
            
            for line in insights_text.split("\n"):
                line = line.strip()
                
                # Check for insight marker
                if line.startswith(("Insight 1", "Insight 2", "Insight 3", "Insight 4", "Insight 5")) or line.startswith(("1.", "2.", "3.", "4.", "5.")):
                    # Save previous insight if it exists
                    if current_insight and "title" in current_insight:
                        current_insight["explanation"] = current_text.strip()
                        insights.append(current_insight)
                    
                    # Start new insight
                    current_insight = {}
                    current_text = ""
                    
                    # Extract title
                    if ":" in line:
                        parts = line.split(":", 1)
                        current_insight["title"] = parts[1].strip()
                    else:
                        # For numbered list without "Insight" prefix
                        parts = line.split(" ", 1)
                        if len(parts) > 1:
                            current_insight["title"] = parts[1].strip()
                        else:
                            current_insight["title"] = line
                
                # Add content to current insight explanation
                elif current_insight and "title" in current_insight:
                    current_text += line + "\n"
            
            # Add the last insight
            if current_insight and "title" in current_insight:
                current_insight["explanation"] = current_text.strip()
                insights.append(current_insight)
            
            # Format insights with ids
            formatted_insights = []
            for i, insight in enumerate(insights):
                formatted_insights.append({
                    "id": f"insight_{i+1}",
                    "title": insight.get("title", f"Insight {i+1}"),
                    "explanation": insight.get("explanation", "")
                })
            
            return formatted_insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    async def _generate_summary(self, query: str, insights: List[Dict[str, Any]]) -> str:
        """Generate a summary of the analysis using Bedrock/Claude."""
        logger.info("Generating summary")
        
        # Extract top insights for the summary
        insight_points = "\n".join([
            f"- {insight.get('title', '')}: {insight.get('explanation', '')[:100]}..."
            for insight in insights[:3]
        ])
        
        # Create prompt for generating the summary
        prompt = f"""
        Write a concise analysis summary (200 words) for the query: "{query}"
        
        Key insights:
        {insight_points}
        
        The summary should:
        1. Directly address the original query
        2. Synthesize the most important findings
        3. Be written in a professional, analytical tone
        """
        
        try:
            # Generate summary using Bedrock/Claude
            if self.bedrock_tool:
                response = await bedrock.claude_generate.generate_text(
                    client=self.bedrock_tool,
                    prompt=prompt,
                    max_tokens=400,
                    temperature=0.4
                )
            else:
                # Fallback to OpenAI
                response = await self.openai_tool.generate_text(
                    prompt=prompt,
                    max_tokens=400,
                    temperature=0.4
                )
            
            return response.get("text", "Analysis summary unavailable")
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Analysis summary unavailable due to an error"