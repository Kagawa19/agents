"""
Researcher agent implementation.
Responsible for searching and gathering information from various sources.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from multiagent.app.agents.base import BaseAgent
from multiagent.app.monitoring.tracer import LangfuseTracer
from multiagent.app.tools.serper import SerperTool


logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """
    Agent for researching information from various sources.
    Uses SerperTool for web search and content extraction.
    Can use LlamaIndexTool for document retrieval when available.
    """
    
    def __init__(
        self,
        agent_id: str,
        tracer: LangfuseTracer,
        serper_tool: SerperTool,
        llamaindex_tool: Optional[Any] = None
    ):
        """
        Initialize the researcher agent.
        
        Args:
            agent_id: Unique identifier for the agent
            tracer: LangfuseTracer instance for monitoring
            serper_tool: SerperTool for web searches
            llamaindex_tool: Optional LlamaIndexTool for document retrieval
        """
        super().__init__(agent_id=agent_id, tracer=tracer)
        self.serper_tool = serper_tool
        self.llamaindex_tool = llamaindex_tool
        self.config = {}
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the researcher with configuration parameters.
        
        Args:
            config: Configuration parameters for the researcher
        """
        self.config = config
        
        # Set up research parameters
        self.num_results = config.get("num_results", 5)
        self.search_depth = config.get("search_depth", 1)
        self.query_expansion = config.get("query_expansion", True)
        self.include_images = config.get("include_images", False)
        self.max_content_length = config.get("max_content_length", 5000)
        self.content_type_preferences = config.get("content_type_preferences", ["general", "news"])
        
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
        # Check for required query parameter
        if "query" not in input_data:
            logger.error("Missing required parameter 'query'")
            return False
            
        # Check if query is too short
        if len(input_data["query"].strip()) < 2:
            logger.error("Query is too short")
            return False
            
        return True
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand the query to improve search coverage.
        
        Args:
            query: The original query
            
        Returns:
            List of expanded queries
        """
        # This is a simple implementation - in a real system you might use
        # more sophisticated NLP techniques or an LLM to generate variations
        expanded_queries = [query]
        
        # Add variations with "how to" if appropriate
        if not query.lower().startswith("how to"):
            if query.endswith("?") and "how" not in query.lower():
                expanded_queries.append(f"How to {query}")
        
        # Add variations with "what is" if appropriate
        if not query.lower().startswith("what is"):
            if query.endswith("?") and "what" not in query.lower():
                expanded_queries.append(f"What is {query}")
        
        # Add "latest" variant if not already present
        if "latest" not in query.lower():
            expanded_queries.append(f"{query} latest")
        
        return expanded_queries
    
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research task.
        
        Args:
            input_data: Dictionary containing:
                - query: The search query
                - num_results: Number of results to fetch (optional)
                - search_depth: How deep to search (optional)
                - content_types: Types of content to search for (optional)
                
        Returns:
            Dictionary containing:
                - search_results: List of search results
                - processed_information: Processed and structured information
                - metadata: Information about the research process
                - processing_time: Time taken for research
        """
        start_time = time.time()
        logger.info(f"Starting research for query: {input_data.get('query', '')}")
        
        # Create span instead of using context manager
        span = self.tracer.span(name="researcher_execute")
        span.update(input=input_data)
        
        try:
            # Get query and parameters
            query = input_data["query"]
            num_results = input_data.get("num_results", self.num_results)
            search_depth = input_data.get("search_depth", self.search_depth)
            content_types = input_data.get("content_types", self.content_type_preferences)
            
            # Expand query if enabled
            expanded_queries = [query]
            if self.query_expansion:
                expanded_queries = self.expand_query(query)
            
            # Perform searches
            all_results = []
            for expanded_query in expanded_queries:
                # Perform web search
                search_results = self.perform_search(
                    expanded_query, 
                    num_results=max(1, num_results // len(expanded_queries)), 
                    content_types=content_types
                )
                all_results.extend(search_results)
            
            # Remove duplicate results based on URL
            unique_results = self._deduplicate_results(all_results)
            
            # Process and structure the information
            processed_information = self.process_information(unique_results, query)
            
            # Create metadata about the research process
            metadata = {
                "query": query,
                "expanded_queries": expanded_queries if expanded_queries != [query] else [],
                "num_results_requested": num_results,
                "num_results_found": len(unique_results),
                "search_depth": search_depth,
                "content_types": content_types,
                "researched_at": datetime.utcnow().isoformat()
            }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Build result
            result = {
                "search_results": unique_results,
                "processed_information": processed_information,
                "metadata": metadata,
                "processing_time": processing_time,
                "status": "completed"
            }
            
            logger.info(f"Research completed in {processing_time:.2f} seconds with {len(unique_results)} results")
            span.update(output=result)
            return result
                
        except Exception as e:
            error_msg = f"Error in researcher execution: {str(e)}"
            logger.error(error_msg)
            span.update(output={"status": "error", "error": error_msg})
            
            # Return error result
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
    
    def perform_search(
        self, 
        query: str, 
        num_results: int = 5,
        content_types: List[str] = ["general"]
    ) -> List[Dict[str, Any]]:
        """
        Perform web search using SerperTool.
        
        Args:
            query: The search query
            num_results: Number of results to fetch
            content_types: Types of content to search for
            
        Returns:
            List of search results
        """
        span = self.tracer.span(name="perform_search")
        span.update(input={"query": query, "num_results": num_results, "content_types": content_types})
        
        all_results = []
        
        # Check if SerperTool is available
        if self.serper_tool is None:
            logger.error("SerperTool not available")
            span.update(output={"result_count": 0, "error": "SerperTool not available"})
            return []
        
        try:
            # Perform regular web search
            if "general" in content_types:
                try:
                    # Convert the async call to a synchronous one
                    web_results = self._run_sync(self.serper_tool.search(query, num_results=num_results))
                    for result in web_results:
                        result["content_type"] = "general"
                    all_results.extend(web_results)
                except Exception as e:
                    logger.error(f"Error in web search: {str(e)}")
            
            # Perform news search if requested
            if "news" in content_types:
                try:
                    news_results = self._run_sync(self.serper_tool.search(
                        query, 
                        search_type="news", 
                        num_results=min(num_results, 3)
                    ))
                    for result in news_results:
                        result["content_type"] = "news"
                    all_results.extend(news_results)
                except Exception as e:
                    logger.error(f"Error in news search: {str(e)}")
            
            # Perform image search if requested
            if "images" in content_types and self.include_images:
                try:
                    image_results = self._run_sync(self.serper_tool.search(
                        query, 
                        search_type="images", 
                        num_results=min(num_results, 2)
                    ))
                    for result in image_results:
                        result["content_type"] = "image"
                    all_results.extend(image_results)
                except Exception as e:
                    logger.error(f"Error in image search: {str(e)}")
            
            # Try LlamaIndex for document search if available
            if "documents" in content_types and self.llamaindex_tool:
                try:
                    document_results = self._run_sync(self.llamaindex_tool.search(
                        query, 
                        num_results=min(num_results, 3)
                    ))
                    for result in document_results:
                        result["content_type"] = "document"
                    all_results.extend(document_results)
                except Exception as e:
                    logger.error(f"Error in document search: {str(e)}")
        except Exception as e:
            logger.error(f"Error during search operations: {str(e)}")
            span.update(output={"status": "error", "error": str(e)})
            return []
        
        span.update(output={"result_count": len(all_results)})
        return all_results
    def process_information(
        self, 
        search_results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Process and structure the search results into usable information.
        
        Args:
            search_results: List of search results
            query: The original query
            
        Returns:
            Processed and structured information
        """
        span = self.tracer.span(name="process_information")
        processed_items = []
        
        try:
            for result in search_results:
                try:
                    # Extract basic information
                    content_type = result.get("content_type", "general")
                    title = result.get("title", "")
                    url = result.get("link", "")
                    snippet = result.get("snippet", "")
                    
                    # Process content based on type
                    if content_type == "general":
                        processed_item = self._process_web_result(result, query)
                    elif content_type == "news":
                        processed_item = self._process_news_result(result, query)
                    elif content_type == "image":
                        processed_item = self._process_image_result(result, query)
                    elif content_type == "document":
                        processed_item = self._process_document_result(result, query)
                    else:
                        # Default processing
                        processed_item = {
                            "title": title,
                            "url": url,
                            "content": snippet,
                            "content_type": "unknown",
                            "relevance": self._calculate_relevance(title, snippet, query)
                        }
                    
                    # Add to processed items if valid
                    if processed_item and "title" in processed_item and processed_item["title"]:
                        processed_items.append(processed_item)
                        
                except Exception as e:
                    logger.warning(f"Error processing result: {str(e)}")
                    continue
            
            # Sort by relevance (highest first)
            processed_items = sorted(processed_items, key=lambda x: x.get("relevance", 0), reverse=True)
            
            span.update(output={"item_count": len(processed_items)})
            return processed_items
        except Exception as e:
            logger.error(f"Error in process_information: {str(e)}")
            span.update(output={"status": "error", "error": str(e)})
            return []
    
    def _process_web_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process a web search result.
        
        Args:
            result: Web search result
            query: The search query
            
        Returns:
            Processed information
        """
        title = result.get("title", "")
        url = result.get("link", "")
        snippet = result.get("snippet", "")
        
        # Calculate relevance
        relevance = self._calculate_relevance(title, snippet, query)
        
        # Truncate snippet if too long
        if len(snippet) > self.max_content_length:
            snippet = snippet[:self.max_content_length] + "..."
        
        return {
            "title": title,
            "url": url,
            "content": snippet,
            "content_type": "web",
            "relevance": relevance,
            "published_date": result.get("published_date", None),
            "domain": self._extract_domain(url)
        }
    
    def _process_news_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process a news search result.
        
        Args:
            result: News search result
            query: The search query
            
        Returns:
            Processed information
        """
        title = result.get("title", "")
        url = result.get("link", "")
        snippet = result.get("snippet", "")
        
        # News articles often have a published date
        published_date = result.get("date", result.get("published_date", None))
        
        # Calculate relevance (slightly boost news items for recency)
        relevance = self._calculate_relevance(title, snippet, query) * 1.1
        
        return {
            "title": title,
            "url": url,
            "content": snippet,
            "content_type": "news",
            "relevance": min(relevance, 1.0),  # Cap at 1.0
            "published_date": published_date,
            "source": result.get("source", self._extract_domain(url))
        }
    
    def _process_image_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process an image search result.
        
        Args:
            result: Image search result
            query: The search query
            
        Returns:
            Processed information
        """
        title = result.get("title", "")
        image_url = result.get("imageUrl", result.get("image_url", ""))
        source_url = result.get("link", result.get("source_url", ""))
        
        # Images may have alt text or a description
        description = result.get("snippet", result.get("description", title))
        
        return {
            "title": title,
            "url": source_url,
            "image_url": image_url,
            "content": description,
            "content_type": "image",
            "relevance": 0.7,  # Default relevance for images
            "source": self._extract_domain(source_url)
        }
    
    def _process_document_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process a document search result.
        
        Args:
            result: Document search result
            query: The search query
            
        Returns:
            Processed information
        """
        title = result.get("title", "")
        content = result.get("content", result.get("text", ""))
        
        # Calculate relevance
        relevance = self._calculate_relevance(title, content, query)
        
        # Truncate content if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        return {
            "title": title,
            "content": content,
            "content_type": "document",
            "relevance": relevance,
            "document_id": result.get("id", ""),
            "metadata": result.get("metadata", {})
        }
    
    def _calculate_relevance(self, title: str, content: str, query: str) -> float:
        """
        Calculate relevance score for a result based on keyword matching.
        
        Args:
            title: Result title
            content: Result content
            query: The search query
            
        Returns:
            Relevance score between 0 and 1
        """
        # This is a simple implementation - in a real system you might use
        # more sophisticated relevance calculation with vector embeddings
        query_terms = query.lower().split()
        
        # Remove very common words
        stop_words = {"the", "and", "or", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about"}
        query_terms = [term for term in query_terms if term not in stop_words]
        
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Exact match in title is a strong signal
        if query.lower() in title_lower:
            return 1.0
        
        # Count term occurrences
        title_matches = sum(1 for term in query_terms if term in title_lower)
        content_matches = sum(1 for term in query_terms if term in content_lower)
        
        # Calculate scores
        title_score = title_matches / max(1, len(query_terms)) * 0.7
        content_score = content_matches / max(1, len(query_terms)) * 0.3
        
        # Combine scores
        return min(title_score + content_score, 1.0)
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: The URL
            
        Returns:
            Domain name
        """
        try:
            if not url:
                return ""
                
            # Remove protocol
            domain = url.split("//")[-1]
            
            # Remove path
            domain = domain.split("/")[0]
            
            # Remove www prefix if present
            if domain.startswith("www."):
                domain = domain[4:]
                
            return domain
        except Exception:
            return url
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on URL.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated list of search results
        """
        unique_results = []
        seen_urls = set()
        
        for result in results:
            url = result.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results
        
    def _run_sync(self, coroutine):
        """
        Run a coroutine synchronously.
        
        Args:
            coroutine: The coroutine to run
            
        Returns:
            The result of the coroutine
        """
        import asyncio
        
        try:
            # Create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the coroutine
            result = loop.run_until_complete(coroutine)
            
            # Close the loop
            loop.close()
            
            return result
        except Exception as e:
            logger.error(f"Error running coroutine synchronously: {e}")
            # In case of error, return an empty result rather than propagating the exception
            return []