"""
Document retriever for LlamaIndex.
Provides different retrieval strategies for fetching relevant documents.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class Retriever:
    """
    Document retriever that implements various retrieval strategies.
    Provides methods for semantic search, hybrid search, and filtering.
    """
    
    def __init__(
        self,
        tracer: Optional[LangfuseTracer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            tracer: Optional LangfuseTracer instance for monitoring
            config: Configuration parameters
        """
        self.tracer = tracer
        self.config = config or {}
        self.top_k = self.config.get("top_k", 5)
        self.score_threshold = self.config.get("score_threshold", 0.7)
    
    async def semantic_search(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]],
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query_embedding: Query embedding
            document_embeddings: List of document embeddings
            documents: List of documents
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with scores
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="semantic_search",
                input={"document_count": len(documents)}
            )
        
        try:
            # Use default values if not provided
            if top_k is None:
                top_k = self.top_k
            
            if score_threshold is None:
                score_threshold = self.score_threshold
            
            # Ensure we have embeddings for all documents
            if len(document_embeddings) != len(documents):
                logger.warning(f"Embedding count ({len(document_embeddings)}) doesn't match document count ({len(documents)})")
                # Truncate to the smaller size
                min_size = min(len(document_embeddings), len(documents))
                document_embeddings = document_embeddings[:min_size]
                documents = documents[:min_size]
            
            # Calculate cosine similarity
            similarities = self._calculate_similarities(query_embedding, document_embeddings)
            
            # Sort by similarity score
            results = []
            for i, score in enumerate(similarities):
                if score >= score_threshold:
                    doc_with_score = documents[i].copy()
                    doc_with_score["score"] = float(score)
                    results.append(doc_with_score)
            
            # Sort by score in descending order
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            # Limit to top_k
            results = results[:top_k]
            
            if span:
                span.update(output={"result_count": len(results)})
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    def _calculate_similarities(
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
    
    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        documents: List[Dict[str, Any]],
        document_embeddings: List[List[float]],
        top_k: Optional[int] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Query string
            query_embedding: Query embedding
            documents: List of documents
            document_embeddings: List of document embeddings
            top_k: Number of documents to retrieve
            alpha: Weight between semantic (1.0) and keyword (0.0) scores
            
        Returns:
            List of relevant documents with scores
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="hybrid_search",
                input={"query": query, "document_count": len(documents), "alpha": alpha}
            )
        
        try:
            # Use default top_k if not provided
            if top_k is None:
                top_k = self.top_k
            
            # Calculate semantic similarity scores
            semantic_scores = self._calculate_similarities(query_embedding, document_embeddings)
            
            # Calculate keyword matching scores
            keyword_scores = []
            query_terms = set(query.lower().split())
            
            for doc in documents:
                content = doc.get("content", "").lower()
                # Count matching terms
                match_count = sum(1 for term in query_terms if term in content)
                # Normalize by query term count
                score = match_count / len(query_terms) if query_terms else 0
                keyword_scores.append(score)
            
            # Combine scores with alpha weighting
            combined_scores = []
            for i in range(len(documents)):
                if i < len(semantic_scores) and i < len(keyword_scores):
                    combined_score = (alpha * semantic_scores[i]) + ((1 - alpha) * keyword_scores[i])
                    combined_scores.append(combined_score)
            
            # Create results with combined scores
            results = []
            for i, score in enumerate(combined_scores):
                if i < len(documents):
                    doc_with_score = documents[i].copy()
                    doc_with_score["score"] = float(score)
                    doc_with_score["semantic_score"] = float(semantic_scores[i]) if i < len(semantic_scores) else 0
                    doc_with_score["keyword_score"] = float(keyword_scores[i]) if i < len(keyword_scores) else 0
                    results.append(doc_with_score)
            
            # Sort by combined score in descending order
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            # Limit to top_k
            results = results[:top_k]
            
            if span:
                span.update(output={"result_count": len(results)})
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def filter_by_metadata(
        self,
        documents: List[Dict[str, Any]],
        filter_criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter documents by metadata criteria.
        
        Args:
            documents: List of documents
            filter_criteria: Metadata filter criteria
            
        Returns:
            Filtered documents
        """
        try:
            filtered_docs = []
            for doc in documents:
                metadata = doc.get("metadata", {})
                if self._matches_criteria(metadata, filter_criteria):
                    filtered_docs.append(doc)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error filtering documents: {str(e)}")
            return []
    
    def _matches_criteria(
        self,
        metadata: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches filter criteria.
        
        Args:
            metadata: Document metadata
            criteria: Filter criteria
            
        Returns:
            True if metadata matches criteria, False otherwise
        """
        for key, value in criteria.items():
            # Handle nested keys with dot notation
            if "." in key:
                parts = key.split(".")
                current = metadata
                for part in parts[:-1]:
                    if part not in current:
                        return False
                    current = current[part]
                
                # Check final key
                if parts[-1] not in current or current[parts[-1]] != value:
                    return False
            
            # Handle special operators
            elif key.endswith("__gt"):
                base_key = key[:-4]
                if base_key not in metadata or metadata[base_key] <= value:
                    return False
            elif key.endswith("__lt"):
                base_key = key[:-4]
                if base_key not in metadata or metadata[base_key] >= value:
                    return False
            elif key.endswith("__in"):
                base_key = key[:-4]
                if base_key not in metadata or metadata[base_key] not in value:
                    return False
            
            # Handle simple keys
            elif key not in metadata or metadata[key] != value:
                return False
        
        return True
    
    async def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        model: str = "cross-encoder"
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using a cross-encoder or other reranking model.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            model: Reranking model to use
            
        Returns:
            Reranked documents
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="rerank_documents",
                input={"document_count": len(documents), "model": model}
            )
        
        try:
            # Check if we have sentence-transformers installed
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                logger.warning("sentence-transformers not installed, skipping reranking")
                if span:
                    span.update(output={"error": "sentence-transformers not installed"})
                return documents
            
            # Initialize cross-encoder
            cross_encoder = CrossEncoder(model)
            
            # Prepare document-query pairs
            pairs = []
            for doc in documents:
                content = doc.get("content", "")
                pairs.append([query, content])
            
            # Calculate reranking scores
            scores = cross_encoder.predict(pairs)
            
            # Update document scores
            reranked_docs = []
            for i, doc in enumerate(documents):
                if i < len(scores):
                    doc_copy = doc.copy()
                    # Save original score if present
                    if "score" in doc_copy:
                        doc_copy["original_score"] = doc_copy["score"]
                    doc_copy["score"] = float(scores[i])
                    reranked_docs.append(doc_copy)
                else:
                    reranked_docs.append(doc)
            
            # Sort by new scores
            reranked_docs = sorted(reranked_docs, key=lambda x: x.get("score", 0), reverse=True)
            
            if span:
                span.update(output={"status": "success"})
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return documents
    
    async def mmr_reranking(
        self,
        query_embedding: List[float],
        documents: List[Dict[str, Any]],
        document_embeddings: List[List[float]],
        top_k: Optional[int] = None,
        lambda_param: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using Maximum Marginal Relevance (MMR) for diversity.
        
        Args:
            query_embedding: Query embedding
            documents: List of documents
            document_embeddings: List of document embeddings
            top_k: Number of documents to retrieve
            lambda_param: Balance between relevance and diversity (0-1)
            
        Returns:
            Reranked documents with diversity
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="mmr_reranking",
                input={"document_count": len(documents), "lambda": lambda_param}
            )
        
        try:
            # Use default top_k if not provided
            if top_k is None:
                top_k = self.top_k
            
            # Calculate similarity scores
            similarities = self._calculate_similarities(query_embedding, document_embeddings)
            
            # Create result list with documents and scores
            scored_docs = []
            for i, score in enumerate(similarities):
                if i < len(documents):
                    doc_with_score = documents[i].copy()
                    doc_with_score["score"] = float(score)
                    scored_docs.append(doc_with_score)
            
            # Apply MMR reranking
            selected_indices = []
            remaining_indices = list(range(len(scored_docs)))
            
            # Select documents iteratively
            for _ in range(min(top_k, len(scored_docs))):
                if not remaining_indices:
                    break
                
                # Calculate MMR scores
                mmr_scores = []
                for i in remaining_indices:
                    if not selected_indices:
                        # First document is selected based on relevance only
                        mmr_scores.append((i, scored_docs[i]["score"]))
                    else:
                        # Calculate diversity penalty
                        relevance = scored_docs[i]["score"]
                        diversity_score = min(
                            [1 - self._calculate_similarity(
                                document_embeddings[i], document_embeddings[j]
                            ) for j in selected_indices]
                        )
                        
                        # Combine relevance and diversity
                        mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity_score
                        mmr_scores.append((i, mmr_score))
                
                # Select document with highest MMR score
                if mmr_scores:
                    selected_index, _ = max(mmr_scores, key=lambda x: x[1])
                    selected_indices.append(selected_index)
                    remaining_indices.remove(selected_index)
            
            # Create reranked result list
            reranked_docs = [scored_docs[i] for i in selected_indices]
            
            if span:
                span.update(output={"result_count": len(reranked_docs)})
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in MMR reranking: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return documents
    
    def _calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        norm1 = max(norm1, 1e-10)
        norm2 = max(norm2, 1e-10)
        
        # Calculate cosine similarity
        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        similarity = np.dot(vec1_normalized, vec2_normalized)
        
        return float(similarity)