"""
Serper API Tool implementation.
Provides web search capabilities using the Serper API.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional

import httpx

from multiagent.app.monitoring.tracer import LangfuseTracer
from multiagent.app.monitoring.metrics import track_vector_db_operation


logger = logging.getLogger(__name__)


class SerperTool:
    """
    Tool for performing web searches using the Serper API.
    Provides functions for general search, news search, and image search.
    """
    
    def __init__(self, api_key: str, tracer: LangfuseTracer):
        """
        Initialize the Serper tool.
        
        Args:
            api_key: Serper API key
            tracer: LangfuseTracer instance for monitoring
        """
        self.api_key = api_key
        self.tracer = tracer
        self.base_url = "https://serper.dev/api/v1"
        logger.info("SerperTool initialized")
    
    def search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Perform a general web search using Serper API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        with self.tracer.span("serper_search"):
            start_time = time.time()
            
            url = f"{self.base_url}/search"
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": num_results
            }
            
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    results = response.json()
                    
                    # Record metrics
                    execution_time = time.time() - start_time
                    track_vector_db_operation("search", "success", execution_time)
                    
                    # Log event
                    self.tracer.log_event(
                        event_type="serper_search",
                        event_data={
                            "query": query,
                            "num_results": num_results,
                            "result_count": len(results.get("organic", [])),
                            "execution_time": execution_time
                        }
                    )
                    
                    # Process results to extract useful information
                    processed_results = self._process_results(results)
                    
                    return {
                        "query": query,
                        "results": processed_results,
                        "raw_results": results if len(json.dumps(results)) < 10000 else {"note": "Raw results truncated due to size"},
                        "execution_time": execution_time
                    }
            except httpx.HTTPStatusError as e:
                logger.error(f"Serper API error: {e.response.status_code} - {e.response.text}")
                
                # Record metrics
                execution_time = time.time() - start_time
                track_vector_db_operation("search", "failure", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="serper_error",
                    event_data={
                        "error": str(e), 
                        "status_code": e.response.status_code,
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Serper API error: {str(e)}",
                    "query": query,
                    "execution_time": execution_time
                }
            except Exception as e:
                logger.error(f"Error performing search: {str(e)}")
                
                # Record metrics
                execution_time = time.time() - start_time
                track_vector_db_operation("search", "failure", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="serper_error",
                    event_data={
                        "error": str(e),
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error performing search: {str(e)}",
                    "query": query,
                    "execution_time": execution_time
                }
    
    def news_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Perform a news search using Serper API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary containing news search results
        """
        with self.tracer.span("serper_news_search"):
            start_time = time.time()
            
            url = f"{self.base_url}/news"
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": num_results
            }
            
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    results = response.json()
                    
                    # Record metrics
                    execution_time = time.time() - start_time
                    track_vector_db_operation("news_search", "success", execution_time)
                    
                    # Log event
                    self.tracer.log_event(
                        event_type="serper_news_search",
                        event_data={
                            "query": query,
                            "num_results": num_results,
                            "result_count": len(results.get("news", [])),
                            "execution_time": execution_time
                        }
                    )
                    
                    return {
                        "query": query,
                        "results": results.get("news", []),
                        "execution_time": execution_time
                    }
            except Exception as e:
                logger.error(f"Error performing news search: {str(e)}")
                
                # Record metrics
                execution_time = time.time() - start_time
                track_vector_db_operation("news_search", "failure", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="serper_error",
                    event_data={
                        "error": str(e),
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error performing news search: {str(e)}",
                    "query": query,
                    "execution_time": execution_time
                }
    
    def image_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Perform an image search using Serper API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary containing image search results
        """
        with self.tracer.span("serper_image_search"):
            start_time = time.time()
            
            url = f"{self.base_url}/images"
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": num_results
            }
            
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    results = response.json()
                    
                    # Record metrics
                    execution_time = time.time() - start_time
                    track_vector_db_operation("image_search", "success", execution_time)
                    
                    # Log event
                    self.tracer.log_event(
                        event_type="serper_image_search",
                        event_data={
                            "query": query,
                            "num_results": num_results,
                            "result_count": len(results.get("images", [])),
                            "execution_time": execution_time
                        }
                    )
                    
                    return {
                        "query": query,
                        "results": results.get("images", []),
                        "execution_time": execution_time
                    }
            except Exception as e:
                logger.error(f"Error performing image search: {str(e)}")
                
                # Record metrics
                execution_time = time.time() - start_time
                track_vector_db_operation("image_search", "failure", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="serper_error",
                    event_data={
                        "error": str(e),
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error performing image search: {str(e)}",
                    "query": query,
                    "execution_time": execution_time
                }
    
    def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process search results to extract useful information.
        
        Args:
            results: Raw search results from Serper API
            
        Returns:
            Processed search results
        """
        processed = {
            "organic": [],
            "answer_box": None,
            "knowledge_graph": None,
            "related_searches": results.get("relatedSearches", [])
        }
        
        # Process organic results
        for item in results.get("organic", []):
            processed["organic"].append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "position": item.get("position")
            })
        
        # Process answer box if present
        if "answerBox" in results:
            answer_box = results["answerBox"]
            processed["answer_box"] = {
                "title": answer_box.get("title", ""),
                "answer": answer_box.get("answer", ""),
                "snippet": answer_box.get("snippet", "")
            }
        
        # Process knowledge graph if present
        if "knowledgeGraph" in results:
            kg = results["knowledgeGraph"]
            processed["knowledge_graph"] = {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "description": kg.get("description", "")
            }
        
        return processed