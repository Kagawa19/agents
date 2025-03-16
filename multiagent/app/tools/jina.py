"""
Jina AI integration.
Provides a client for interacting with Jina AI services.
"""

import logging
import json
import aiohttp
from typing import Dict, Any, List, Optional, Union
import numpy as np
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class JinaTool:
    """
    Tool for interacting with Jina AI services.
    Provides methods for embeddings, search, and more.
    """
    
    def __init__(
        self,
        api_key: str,
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the Jina tool.
        
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
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: str = "jina-embeddings-v2-base-en"
    ) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to get embeddings for
            model: Embedding model to use
            
        Returns:
            List of embeddings
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_get_embeddings",
                input={"text_count": len(texts), "model": model}
            )
        
        try:
            # Check for empty inputs
            if not texts:
                return []
            
            # Prepare request
            url = f"{self.base_url}/embeddings"
            payload = {
                "model": model,
                "input": texts
            }
            
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
            
            # Extract embeddings
            embeddings = [item["embedding"] for item in result["data"]]
            
            if span:
                span.update(output={"embedding_count": len(embeddings)})
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def create_index(
        self,
        index_name: str,
        dimension: int = 768,
        metric: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Create a vector index.
        
        Args:
            index_name: Name of the index
            dimension: Dimension of embeddings
            metric: Distance metric (cosine, euclidean, dot)
            
        Returns:
            Index information
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_create_index",
                input={"index_name": index_name, "dimension": dimension}
            )
        
        try:
            # Prepare request
            url = f"{self.base_url}/indexes"
            payload = {
                "name": index_name,
                "dimension": dimension,
                "metric": metric
            }
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status not in [200, 201]:
                        error_text = await response.text()
                        logger.error(f"Jina API error ({response.status}): {error_text}")
                        if span:
                            span.update(output={"error": error_text})
                        return {"error": error_text}
                    
                    # Parse response
                    result = await response.json()
            
            if span:
                span.update(output={"status": "success"})
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def upsert_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Upsert documents to an index.
        
        Args:
            index_name: Name of the index
            documents: List of documents to upsert
            
        Returns:
            Upsert response
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_upsert_documents",
                input={"index_name": index_name, "document_count": len(documents)}
            )
        
        try:
            # Check for empty inputs
            if not documents:
                return {"status": "success", "count": 0}
            
            # Prepare request
            url = f"{self.base_url}/indexes/{index_name}/upsert"
            payload = {
                "documents": documents
            }
            
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
                        return {"error": error_text}
                    
                    # Parse response
                    result = await response.json()
            
            if span:
                span.update(output={"status": "success"})
            
            return result
            
        except Exception as e:
            logger.error(f"Error upserting documents: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def search(
        self,
        index_name: str,
        query_embedding: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in an index.
        
        Args:
            index_name: Name of the index
            query_embedding: Query embedding
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
                "vector": query_embedding,
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
    
    async def calculate_similarities(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]]
    ) -> List[float]:
        """
        Calculate cosine similarities between query and document embeddings.
        
        Args:
            query_embedding: Query embedding
            embeddings: List of document embeddings
            
        Returns:
            List of similarity scores
        """
        # Convert to numpy arrays for easier computation
        query_np = np.array(query_embedding)
        docs_np = np.array(embeddings)
        
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
            from sklearn.cluster import KMeans
            
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
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "jina-chat",
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text using Jina's LLM services.
        
        Args:
            prompt: Text prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            
        Returns:
            Generated text response
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_generate_text",
                input={"model": model, "max_tokens": max_tokens}
            )
        
        try:
            # Prepare request
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
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
                        return {"text": "", "error": error_text}
                    
                    # Parse response
                    result = await response.json()
            
            # Extract generated text
            choices = result.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
            else:
                text = ""
            
            if span:
                span.update(output={"status": "success"})
            
            return {"text": text, "model": model}
            
        except Exception as e:
            logger.error(f"Error generating text with Jina: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"text": "", "error": str(e)}