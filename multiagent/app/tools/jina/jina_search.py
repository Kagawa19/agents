"""
Vector search for Jina.
Provides methods for vector search and similarity calculations.
"""

import logging
import aiohttp
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class JinaSearch:
    """
    Vector search utilities for Jina.
    Provides methods for searching and calculating similarities.
    """
    
    def __init__(
        self,
        api_key: str,
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the Jina Search utilities.
        
        Args:
            api_key: Jina AI API key
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.api_key = api_key
        self.tracer = tracer
        self.base_url = "https://api.jina.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def search(
        self,
        index_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in an index.
        
        Args:
            index_name: Name of the index
            query_vector: Query embedding
            limit: Number of results to return
            filter: Optional filter criteria
            
        Returns:
            List of search results
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_search",
                input={"index_name": index_name, "limit": limit}
            )
        
        try:
            # Prepare request
            url = f"{self.base_url}/indexes/{index_name}/search"
            payload = {
                "vector": query_vector,
                "limit": limit
            }
            
            if filter:
                payload["filter"] = filter
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Jina API error ({response.status}): {error_text}")
                        if span:
                            span.update(output={"error": error_text})
                        return []
                    
                    # Parse response
                    result = await response.json()
            
            # Extract matches
            matches = result.get("matches", [])
            
            if span:
                span.update(output={"match_count": len(matches)})
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def text_search(
        self,
        index_name: str,
        query_text: str,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (will be embedded first).
        
        Args:
            index_name: Name of the index
            query_text: Text query
            limit: Number of results to return
            filter: Optional filter criteria
            
        Returns:
            List of search results
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_text_search",
                input={"index_name": index_name, "query": query_text}
            )
        
        try:
            # Import jina_extract to get embeddings
            from multiagent.app.tools.jina.jina_extract import JinaExtract
            
            # Create extractor instance
            extractor = JinaExtract(api_key=self.api_key, tracer=self.tracer)
            
            # Get query embedding
            embeddings = await extractor.get_embeddings([query_text])
            if not embeddings:
                raise ValueError("Failed to generate embedding for query text")
            
            query_vector = embeddings[0]
            
            # Perform vector search
            results = await self.search(
                index_name=index_name,
                query_vector=query_vector,
                limit=limit,
                filter=filter
            )
            
            if span:
                span.update(output={"match_count": len(results)})
            
            return results
            
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def hybrid_search(
        self,
        index_name: str,
        query_text: str,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and text matching.
        
        Args:
            index_name: Name of the index
            query_text: Text query
            limit: Number of results to return
            filter: Optional filter criteria
            alpha: Weight between vector and text scores (1.0 = vector only)
            
        Returns:
            List of search results
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_hybrid_search",
                input={"index_name": index_name, "query": query_text, "alpha": alpha}
            )
        
        try:
            # Prepare request
            url = f"{self.base_url}/indexes/{index_name}/hybrid_search"
            payload = {
                "query": query_text,
                "limit": limit,
                "alpha": alpha
            }
            
            if filter:
                payload["filter"] = filter
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Jina API error ({response.status}): {error_text}")
                        
                        # Fall back to regular vector search if hybrid search is not available
                        if response.status == 404 and "not found" in (await response.text()).lower():
                            logger.warning("Hybrid search not available, falling back to text search")
                            if span:
                                span.update(output={"fallback": "text_search"})
                            return await self.text_search(index_name, query_text, limit, filter)
                        
                        if span:
                            span.update(output={"error": error_text})
                        return []
                    
                    # Parse response
                    result = await response.json()
            
            # Extract matches
            matches = result.get("matches", [])
            
            if span:
                span.update(output={"match_count": len(matches)})
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            # Fall back to text search
            try:
                logger.info("Falling back to text search after hybrid search error")
                return await self.text_search(index_name, query_text, limit, filter)
            except Exception as e2:
                logger.error(f"Error in fallback text search: {str(e2)}")
                return []
    
    async def calculate_similarities(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Calculate cosine similarities between query and document embeddings.
        
        Args:
            query_embedding: Query embedding
            document_embeddings: List of document embeddings
            
        Returns:
            List of similarity scores
        """
        try:
            # Convert to numpy arrays for easier computation
            query_np = np.array(query_embedding)
            docs_np = np.array(document_embeddings)
            
            # Normalize vectors
            query_norm = np.linalg.norm(query_np)
            docs_norm = np.linalg.norm(docs_np, axis=1)
            
            # Avoid division by zero
            query_norm = max(query_norm, 1e-10)
            docs_norm = np.maximum(docs_norm, 1e-10)
            
            # Calculate cosine similarity
            query_normalized = query_np / query_norm
            docs_normalized = docs_np / docs_norm[:, np.newaxis]
            similarities = np.dot(docs_normalized, query_normalized)
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {str(e)}")
            return [0.0] * len(document_embeddings)
    
    async def cluster_embeddings(
        self,
        embeddings: List[List[float]],
        n_clusters: int = 3
    ) -> Dict[int, List[int]]:
        """
        Cluster embeddings using K-means.
        
        Args:
            embeddings: List of embeddings to cluster
            n_clusters: Number of clusters
            
        Returns:
            Dictionary mapping cluster IDs to document indices
        """
        try:
            # Import scikit-learn
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                logger.error("scikit-learn not installed")
                return {0: list(range(len(embeddings)))}
            
            # Convert to numpy array
            embeddings_np = np.array(embeddings)
            
            # Ensure we don't request more clusters than documents
            n_clusters = min(n_clusters, len(embeddings))
            
            # Initialize and fit K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_np)
            
            # Group document indices by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[int(label)] = []
                clusters[int(label)].append(i)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering embeddings: {str(e)}")
            # Return a single cluster with all documents
            return {0: list(range(len(embeddings)))}
    
    async def get_similar_documents(
        self,
        index_name: str,
        doc_id: str,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to a reference document.
        
        Args:
            index_name: Name of the index
            doc_id: Reference document ID
            limit: Number of results to return
            filter: Optional filter criteria
            
        Returns:
            List of similar documents
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_similar_documents",
                input={"index_name": index_name, "doc_id": doc_id}
            )
        
        try:
            # Prepare request
            url = f"{self.base_url}/indexes/{index_name}/documents/{doc_id}/similar"
            payload = {
                "limit": limit
            }
            
            if filter:
                payload["filter"] = filter
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Jina API error ({response.status}): {error_text}")
                        if span:
                            span.update(output={"error": error_text})
                        return []
                    
                    # Parse response
                    result = await response.json()
            
            # Extract matches
            matches = result.get("matches", [])
            
            if span:
                span.update(output={"match_count": len(matches)})
            
            return matches
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []