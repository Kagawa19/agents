"""
SerperTool for web search.
Integrates with the Serper.dev API to search the web.
"""

import logging
import json
import aiohttp
from typing import Dict, Any, List, Optional

from multiagent.app.monitoring.tracer import LangfuseTracer


logger = logging.getLogger(__name__)


class SerperTool:
    """
    Tool for searching the web using Serper.dev API.
    Provides methods for searching Google and extracting results.
    """
    
    def __init__(self, api_key: str, tracer: Optional[LangfuseTracer] = None):
        """
        Initialize the Serper tool.
        
        Args:
            api_key: Serper.dev API key
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.api_key = api_key
        self.tracer = tracer
        self.base_url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
    
    async def search(
        self,
        query: str,
        search_type: str = "search",
        num_results: int = 5,
        page: int = 1,
        country: str = "us",
        locale: str = "en",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the web with the given query.
        
        Args:
            query: Search query string
            search_type: Type of search (search, images, news, places)
            num_results: Number of results to return
            page: Page number for pagination
            country: Country code for localized results
            locale: Language code for localized results
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            
        Returns:
            List of search result items
        """
        # Create span for tracing if tracer is provided
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="serper_search",
                input={
                    "query": query,
                    "search_type": search_type,
                    "num_results": num_results
                }
            )
        
        try:
            logger.info(f"Searching with query: {query}")
            
            # Prepare payload
            payload = {
                "q": query,
                "gl": country,
                "hl": locale,
                "num": num_results,
                "page": page
            }
            
            # Add domain filters if provided
            if include_domains:
                include_filter = " OR ".join([f"site:{domain}" for domain in include_domains])
                payload["q"] = f"{payload['q']} ({include_filter})"
            
            if exclude_domains:
                exclude_filter = " ".join([f"-site:{domain}" for domain in exclude_domains])
                payload["q"] = f"{payload['q']} {exclude_filter}"
            
            # Select the right endpoint based on search type
            if search_type == "images":
                self.base_url = "https://google.serper.dev/images"
            elif search_type == "news":
                self.base_url = "https://google.serper.dev/news"
            elif search_type == "places":
                self.base_url = "https://google.serper.dev/places"
            else:
                self.base_url = "https://google.serper.dev/search"
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Serper API error ({response.status}): {error_text}")
                        if span:
                            span.update(output={"error": error_text, "status": response.status})
                        return []
                    
                    # Parse response
                    result = await response.json()
            
            # Extract search results
            search_results = self._extract_results(result, search_type)
            
            # Limit to requested number of results
            search_results = search_results[:num_results]
            
            # Update span with success output
            if span:
                span.update(output={"result_count": len(search_results)})
            
            logger.info(f"Found {len(search_results)} results for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in Serper search: {str(e)}")
            # Update span with error
            if span:
                span.update(output={"error": str(e)})
            return []
    
    def _extract_results(
        self,
        api_response: Dict[str, Any],
        search_type: str
    ) -> List[Dict[str, Any]]:
        """
        Extract and normalize search results from the API response.
        
        Args:
            api_response: API response from Serper
            search_type: Type of search performed
            
        Returns:
            List of normalized search results
        """
        results = []
        
        try:
            # Handle different result types
            if search_type == "search":
                # Organic search results
                if "organic" in api_response:
                    for item in api_response["organic"]:
                        result = {
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "position": item.get("position"),
                            "type": "organic"
                        }
                        
                        # Add additional fields if available
                        if "sitelinks" in item:
                            result["sitelinks"] = item["sitelinks"]
                        
                        results.append(result)
                
                # Knowledge graph results
                if "knowledgeGraph" in api_response:
                    kg = api_response["knowledgeGraph"]
                    result = {
                        "title": kg.get("title", ""),
                        "type": "knowledge_graph",
                        "description": kg.get("description", ""),
                        "attributes": kg.get("attributes", {}),
                        "thumbnailUrl": kg.get("thumbnailUrl", "")
                    }
                    results.append(result)
                
                # Answer box results
                if "answerBox" in api_response:
                    ab = api_response["answerBox"]
                    result = {
                        "title": ab.get("title", ""),
                        "type": "answer_box",
                        "answer": ab.get("answer", ""),
                        "snippet": ab.get("snippet", "")
                    }
                    results.append(result)
                
            elif search_type == "news":
                # News search results
                if "news" in api_response:
                    for item in api_response["news"]:
                        result = {
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "date": item.get("date", ""),
                            "source": item.get("source", ""),
                            "type": "news"
                        }
                        
                        # Add thumbnail if available
                        if "thumbnailUrl" in item:
                            result["thumbnailUrl"] = item["thumbnailUrl"]
                        
                        results.append(result)
                
            elif search_type == "images":
                # Image search results
                if "images" in api_response:
                    for item in api_response["images"]:
                        result = {
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "imageUrl": item.get("imageUrl", ""),
                            "source": item.get("source", ""),
                            "type": "image"
                        }
                        results.append(result)
                
            elif search_type == "places":
                # Places search results
                if "places" in api_response:
                    for item in api_response["places"]:
                        result = {
                            "title": item.get("title", ""),
                            "address": item.get("address", ""),
                            "rating": item.get("rating", ""),
                            "reviews": item.get("reviews", ""),
                            "type": "place"
                        }
                        
                        # Add thumbnail if available
                        if "thumbnailUrl" in item:
                            result["thumbnailUrl"] = item["thumbnailUrl"]
                        
                        results.append(result)
            
            # If API response includes searchParameters, add to all results
            if "searchParameters" in api_response:
                for result in results:
                    result["searchParameters"] = api_response["searchParameters"]
        
        except Exception as e:
            logger.error(f"Error extracting search results: {str(e)}")
        
        return results
    
    async def search_and_combine(
        self,
        query: str,
        num_results: int = 5,
        country: str = "us"
    ) -> Dict[str, Any]:
        """
        Search for a query and combine results from different search types.
        
        Args:
            query: Search query string
            num_results: Number of results per search type
            country: Country code for localized results
            
        Returns:
            Dictionary with combined search results
        """
        # Create span for tracing if tracer is provided
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="serper_combined_search",
                input={"query": query, "num_results": num_results}
            )
        
        try:
            # Perform searches
            web_results = await self.search(query, "search", num_results, country=country)
            news_results = await self.search(query, "news", min(3, num_results), country=country)
            
            # Combine results
            combined_results = {
                "query": query,
                "web": web_results,
                "news": news_results,
                "total_results": len(web_results) + len(news_results)
            }
            
            # Update span with success output
            if span:
                span.update(output={"total_results": combined_results["total_results"]})
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in combined search: {str(e)}")
            # Update span with error
            if span:
                span.update(output={"error": str(e)})
            return {
                "query": query,
                "web": [],
                "news": [],
                "total_results": 0,
                "error": str(e)
            }