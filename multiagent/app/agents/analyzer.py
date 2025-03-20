"""
Enhanced Analyzer Agent Implementation with Detailed Logging
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from multiagent.app.agents.base import BaseAgent
from multiagent.app.monitoring.tracer import LangfuseTracer

# Import tools - simplified for clarity
from multiagent.app.tools import serper, scraper, jina, bedrock


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analyzer_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnalyzerAgent(BaseAgent):
    """
    Enhanced Agent for analyzing information with detailed logging.
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
        
        # Log tool availability
        self.jina_tool = jina_tool
        self.openai_tool = openai_tool
        self.bedrock_tool = bedrock_tool
        self.serper_tool = serper_tool
        self.scraper_tool = scraper_tool
        
        # Detailed tool logging
        print("🔍 Analyzer Agent Initialization:")
        print(f"  Jina Tool: {'Available' if jina_tool else 'Not Available'}")
        print(f"  OpenAI Tool: {'Available' if openai_tool else 'Not Available'}")
        print(f"  Bedrock Tool: {'Available' if bedrock_tool else 'Not Available'}")
        print(f"  Serper Tool: {'Available' if serper_tool else 'Not Available'}")
        print(f"  Scraper Tool: {'Available' if scraper_tool else 'Not Available'}")
        
        logger.info(f"Analyzer Agent initialized with tools: Jina={bool(jina_tool)}, "
                    f"OpenAI={bool(openai_tool)}, Bedrock={bool(bedrock_tool)}, "
                    f"Serper={bool(serper_tool)}, Scraper={bool(scraper_tool)}")
        
        self.config = {}
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration parameters."""
        self.config = config
        print(f"🚀 Initializing {self.agent_id} with config: {config}")
        logger.info(f"Initialized {self.agent_id} with configuration")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analysis on provided information or search for data.
        """
        start_time = time.time()
        query = input_data.get("query", "")
        
        print(f"🔬 Starting Analysis:")
        print(f"  Query: {query}")
        print(f"  Search Web: {input_data.get('search_web', False)}")
        logger.info(f"Starting analysis for query: {query}")
        
        with self.tracer.span(name="analyzer_execute") as span:
            try:
                # Determine information source
                if input_data.get("search_web", False) and not input_data.get("information"):
                    print("🌐 Performing web search...")
                    information = await self._search_and_scrape(
                        query, 
                        num_results=input_data.get("num_results", 5)
                    )
                else:
                    information = input_data.get("information", [])
                
                # Check if information is available
                if not information:
                    print("❌ No information found or provided!")
                    logger.warning("No information available for analysis")
                    return {
                        "status": "error",
                        "error": "No information provided or found for analysis",
                        "processing_time": time.time() - start_time
                    }
                
                print(f"📊 Found {len(information)} sources")
                
                # Process with Jina for semantic analysis
                print("🧠 Processing with Jina...")
                embeddings, clusters = await self._process_with_jina(information)
                
                # Determine insight generation tool
                print("💡 Generating Insights:")
                if self.bedrock_tool:
                    print("  Using Bedrock (Claude)")
                    logger.info("Generating insights with Bedrock")
                else:
                    print("  Falling back to OpenAI")
                    logger.warning("Bedrock unavailable, using OpenAI for insights")
                
                # Generate insights
                insights = await self._generate_insights(query, information, clusters)
                
                # Generate summary
                print("📝 Generating Summary:")
                summary = await self._generate_summary(query, insights)
                
                # Prepare results
                result = {
                    "summary": summary,
                    "insights": insights,
                    "clusters": clusters,
                    "source_count": len(information),
                    "analyzed_at": datetime.utcnow().isoformat()
                }
                
                execution_time = time.time() - start_time
                print(f"✅ Analysis Completed in {execution_time:.2f} seconds")
                logger.info(f"Analysis completed successfully in {execution_time:.2f} seconds")
                
                return {
                    "analysis_results": result,
                    "processing_time": execution_time,
                    "status": "completed"
                }
                
            except Exception as e:
                print(f"❌ Analysis Error: {str(e)}")
                logger.error(f"Error in analyzer execution: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
    
    async def _search_and_scrape(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web and scrape content for the given query."""
        print(f"🔍 Web Search:")
        print(f"  Query: {query}")
        print(f"  Results Requested: {num_results}")
        
        if not self.serper_tool or not self.scraper_tool:
            print("❌ Search tools not available!")
            logger.error("Search tools (Serper or Scraper) are not available")
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
                        print(f"  ✅ Scraped: {url}")
                except Exception as e:
                    print(f"  ❌ Scraping Error for {url}: {str(e)}")
                    logger.warning(f"Error scraping {url}: {str(e)}")
        
        print(f"🌐 Web Search Results: {len(scraped_content)} documents")
        logger.info(f"Web search found {len(scraped_content)} documents")
        return scraped_content
    
    async def _process_with_jina(self, information: List[Dict[str, Any]]) -> tuple:
        """Process information with Jina for embeddings and clustering."""
        print(f"🧠 Jina Processing:")
        print(f"  Documents to Process: {len(information)}")
        
        # Extract text content
        texts = [item.get("content", "") for item in information]
        
        # Get embeddings using Jina
        print("  Generating Embeddings...")
        embeddings = await jina.jina_extract.get_embeddings(self.jina_tool, texts)
        print(f"  Embeddings Generated: {len(embeddings)}")
        
        # Perform clustering if we have enough documents
        clusters = []
        if len(embeddings) >= 2:
            try:
                # Cluster documents
                print("  Clustering Documents...")
                cluster_count = min(5, len(embeddings) // 2) if len(embeddings) > 4 else 2
                cluster_results = await jina.jina_search.cluster_embeddings(
                    self.jina_tool,
                    embeddings,
                    n_clusters=cluster_count
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
                
                print(f"  Clusters Created: {len(clusters)}")
            except Exception as e:
                print(f"  ❌ Clustering Error: {str(e)}")
                logger.warning(f"Error in clustering: {str(e)}")
        
        return embeddings, clusters
    
    async def _generate_insights(self, query: str, information: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights using Bedrock/Claude or OpenAI."""
        print("💡 Generating Insights:")
        
        # Combine text samples for analysis
        combined_text = "\n\n".join([item.get("content", "")[:500] for item in information[:5]])
        
        # Select generation tool
        try:
            if self.bedrock_tool:
                print("  🤖 Using Bedrock (Claude)")
                response = await bedrock.claude_generate.generate_text(
                    client=self.bedrock_tool,
                    prompt=f"""
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
                    """,
                    max_tokens=1000,
                    temperature=0.4
                )
            else:
                # Fallback to OpenAI
                print("  📊 Falling back to OpenAI")
                response = await self.openai_tool.generate_text(
                    prompt=f"""
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
                    """,
                    max_tokens=1000,
                    temperature=0.4
                )
            
            insights_text = response.get("text", "")
            print("  📝 Insights Generated")
            
            # Parsing logic remains the same as previous implementation
            insights = []
            current_insight = {}
            current_text = ""
            
            for line in insights_text.split("\n"):
                line = line.strip()
                
                if line.startswith(("Insight 1", "Insight 2", "Insight 3", "Insight 4", "Insight 5")) or line.startswith(("1.", "2.", "3.", "4.", "5.")):
                    if current_insight and "title" in current_insight:
                        current_insight["explanation"] = current_text.strip()
                        insights.append(current_insight)
                    
                    current_insight = {}
                    current_text = ""
                    
                    if ":" in line:
                        parts = line.split(":", 1)
                        current_insight["title"] = parts[1].strip()
                    else:
                        parts = line.split(" ", 1)
                        if len(parts) > 1:
                            current_insight["title"] = parts[1].strip()
                        else:
                            current_insight["title"] = line
                
                elif current_insight and "title" in current_insight:
                    current_text += line + "\n"
            
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
            
            print(f"  ✅ Insights Formatted: {len(formatted_insights)}")
            return formatted_insights
            
        except Exception as e:
            print(f"  ❌ Insights Generation Error: {str(e)}")
            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
            return []
    
    async def _generate_summary(self, query: str, insights: List[Dict[str, Any]]) -> str:
        """Generate a summary of the analysis using Bedrock/Claude."""
        print("📝 Generating Summary:")
        
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
                print("  🤖 Using Bedrock (Claude)")
                logger.info("Generating summary with Bedrock")
                response = await bedrock.claude_generate.generate_text(
                    client=self.bedrock_tool,
                    prompt=prompt,
                    max_tokens=400,
                    temperature=0.4
                )
            else:
                # Fallback to OpenAI
                print("  📊 Falling back to OpenAI")
                logger.warning("Bedrock unavailable, using OpenAI for summary")
                response = await self.openai_tool.generate_text(
                    prompt=prompt,
                    max_tokens=400,
                    temperature=0.4
                )
            
            # Extract and return summary text
            summary_text = response.get("text", "Analysis summary unavailable")
            print("  ✅ Summary Generated Successfully")
            logger.info("Summary generation completed")
            
            return summary_text
            
        except Exception as e:
            print(f"  ❌ Summary Generation Error: {str(e)}")
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return "Analysis summary unavailable due to an error"

# Optional: Add method to validate and enhance tool availability


# You can add these methods to the AnalyzerAgent class to provide more comprehensive 
# logging, validation, and performance tracking    def validate_tools(self) -> Dict[str, bool]:
        """
        Validate the availability and functionality of tools.
        
        Returns:
            Dictionary of tool availability status
        """
        tool_status = {
            "jina": bool(self.jina_tool),
            "openai": bool(self.openai_tool),
            "bedrock": bool(self.bedrock_tool),
            "serper": bool(self.serper_tool),
            "scraper": bool(self.scraper_tool)
        }
        
        print("🔍 Tool Validation:")
        for tool, available in tool_status.items():
            print(f"  {tool.capitalize()} Tool: {'✅ Available' if available else '❌ Not Available'}")
        
        logger.info(f"Tool availability: {tool_status}")
        return tool_status

    # Add this method to the AnalyzerAgent class to provide more detailed tool checks
    def check_tool_capabilities(self) -> Dict[str, Any]:
        """
        Perform detailed checks on tool capabilities.
        
        Returns:
            Dictionary with detailed tool capability information
        """
        capabilities = {}
        
        # Jina Tool Check
        if self.jina_tool:
            try:
                # Check embedding generation
                capabilities['jina'] = {
                    'embeddings': True,
                    'clustering': True  # Assuming clustering is supported
                }
            except Exception as e:
                capabilities['jina'] = {
                    'error': str(e),
                    'available': False
                }
        
        # Bedrock Tool Check
        if self.bedrock_tool:
            try:
                # Check model availability
                capabilities['bedrock'] = {
                    'text_generation': True,
                    'insight_generation': True,
                    'summarization': True
                }
            except Exception as e:
                capabilities['bedrock'] = {
                    'error': str(e),
                    'available': False
                }
        
        # OpenAI Tool Check
        if self.openai_tool:
            try:
                # Check basic text generation
                capabilities['openai'] = {
                    'text_generation': True,
                    'fallback_available': True
                }
            except Exception as e:
                capabilities['openai'] = {
                    'error': str(e),
                    'available': False
                }
        
        print("🔬 Detailed Tool Capabilities:")
        for tool, capability in capabilities.items():
            print(f"  {tool.capitalize()} Tool:")
            for key, value in capability.items():
                print(f"    {key}: {value}")
        
        logger.info(f"Tool capabilities: {capabilities}")
        return capabilities

    # Optional: Add a method to log performance metrics
    def log_performance_metrics(self, 
                                start_time: float, 
                                information_count: int, 
                                insight_count: int) -> Dict[str, Any]:
        """
        Log performance metrics for the analysis process.
        
        Args:
            start_time: Start time of the analysis
            information_count: Number of information sources
            insight_count: Number of generated insights
        
        Returns:
            Dictionary with performance metrics
        """
        end_time = time.time()
        execution_time = end_time - start_time
        
        metrics = {
            'total_execution_time': execution_time,
            'information_sources': information_count,
            'generated_insights': insight_count,
            'avg_time_per_source': execution_time / max(information_count, 1)
        }
        
        print("📊 Performance Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value}")
        
        logger.info(f"Performance metrics: {metrics}")
        return metrics