"""
Jina AI Tool implementation.
Provides functions for document processing and embedding generation using Jina AI.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union

import httpx
import numpy as np
from docarray import Document, DocumentArray

from multiagent.app.monitoring.tracer import LangfuseTracer
from multiagent.app.monitoring.metrics import track_vector_db_operation


logger = logging.getLogger(__name__)


class JinaTool:
    """
    Tool for interacting with Jina AI services.
    Provides functions for document processing and embedding generation.
    """
    
    def __init__(self, api_key: str, tracer: LangfuseTracer):
        """
        Initialize the Jina tool.
        
        Args:
            api_key: Jina AI API key
            tracer: LangfuseTracer instance for monitoring
        """
        self.api_key = api_key
        self.tracer = tracer
        
        # Set up Jina client
        try:
            from jina import Client
            self.client = Client(host="grpcs://api.jina.ai:443")
            self.client.kwargs = {"headers": {"Authorization": f"Bearer {api_key}"}}
            logger.info("Jina AI client initialized successfully")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Jina AI client: {e}")
            self.initialized = False
    
    def process_documents(self, text: Union[str, List[str], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process documents using Jina AI.
        Chunks documents and extracts information.
        
        Args:
            text: Text content to process (string, list of strings, or dict)
            
        Returns:
            Processed document structure
        """
        with self.tracer.span("jina_process_documents"):
            start_time = time.time()
            
            if not self.initialized:
                return {"error": "Jina AI client not initialized"}
            
            try:
                # Convert input to DocumentArray
                docs = self._prepare_documents(text)
                
                # Process with Jina
                response = self.client.post("/documents/process", docs)
                
                # Extract and format results
                results = []
                for doc in response:
                    chunks = []
                    for chunk in doc.chunks:
                        chunks.append({
                            "id": chunk.id,
                            "text": chunk.text,
                            "mime_type": chunk.mime_type,
                            "tags": dict(chunk.tags) if hasattr(chunk, "tags") else {}
                        })
                    
                    results.append({
                        "id": doc.id,
                        "text": doc.text,
                        "chunks": chunks,
                        "tags": dict(doc.tags) if hasattr(doc, "tags") else {}
                    })
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_vector_db_operation("process_documents", "success", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="jina_process_documents",
                    event_data={
                        "document_count": len(docs),
                        "chunk_count": sum(len(doc.get("chunks", [])) for doc in results),
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "documents": results,
                    "execution_time": execution_time
                }
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_vector_db_operation("process_documents", "failure", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="jina_error",
                    event_data={
                        "error": str(e),
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error processing documents: {str(e)}",
                    "execution_time": execution_time
                }
    
    def generate_embeddings(self, text: Union[str, List[str], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for text using Jina AI.
        
        Args:
            text: Text to generate embeddings for (string, list of strings, or processed document)
            
        Returns:
            Dictionary containing embeddings
        """
        with self.tracer.span("jina_generate_embeddings"):
            start_time = time.time()
            
            if not self.initialized:
                return {"error": "Jina AI client not initialized"}
            
            try:
                # Prepare documents
                docs = self._prepare_documents(text)
                
                # Generate embeddings
                response = self.client.post("/embeddings", docs)
                
                # Extract embeddings
                embeddings = []
                for doc in response:
                    # Skip documents without embeddings
                    if not hasattr(doc, "embedding") or doc.embedding is None:
                        continue
                    
                    # Convert embedding to list and append to results
                    embedding_list = doc.embedding.tolist() if hasattr(doc.embedding, "tolist") else list(doc.embedding)
                    
                    embeddings.append({
                        "id": doc.id,
                        "text": doc.text[:100] + "..." if len(doc.text) > 100 else doc.text,
                        "embedding": embedding_list
                    })
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_vector_db_operation("generate_embeddings", "success", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="jina_generate_embeddings",
                    event_data={
                        "document_count": len(docs),
                        "embedding_count": len(embeddings),
                        "embedding_dim": len(embeddings[0]["embedding"]) if embeddings else 0,
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "embeddings": embeddings,
                    "execution_time": execution_time
                }
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_vector_db_operation("generate_embeddings", "failure", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="jina_error",
                    event_data={
                        "error": str(e),
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error generating embeddings: {str(e)}",
                    "execution_time": execution_time
                }
    
    def search_similar(self, text: str, corpus: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents given a query and corpus.
        
        Args:
            text: Query text
            corpus: List of documents with embeddings
            top_k: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        with self.tracer.span("jina_search_similar"):
            start_time = time.time()
            
            if not self.initialized:
                return {"error": "Jina AI client not initialized"}
            
            try:
                # Generate query embedding
                query_result = self.generate_embeddings(text)
                if "error" in query_result:
                    return query_result
                
                query_embedding = query_result["embeddings"][0]["embedding"]
                
                # Prepare corpus
                corpus_docs = []
                for item in corpus:
                    if "embedding" not in item:
                        continue
                    
                    corpus_docs.append({
                        "id": item.get("id", ""),
                        "text": item.get("text", ""),
                        "embedding": item["embedding"]
                    })
                
                # Perform similarity search
                results = self._vector_search(query_embedding, corpus_docs, top_k)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_vector_db_operation("search_similar", "success", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="jina_search_similar",
                    event_data={
                        "query": text,
                        "corpus_size": len(corpus),
                        "result_count": len(results),
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "query": text,
                    "results": results,
                    "execution_time": execution_time
                }
            except Exception as e:
                logger.error(f"Error searching similar documents: {str(e)}")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Track metrics
                track_vector_db_operation("search_similar", "failure", execution_time)
                
                # Log event
                self.tracer.log_event(
                    event_type="jina_error",
                    event_data={
                        "error": str(e),
                        "execution_time": execution_time
                    }
                )
                
                return {
                    "error": f"Error searching similar documents: {str(e)}",
                    "execution_time": execution_time
                }
    
    def _prepare_documents(self, text: Union[str, List[str], Dict[str, Any]]) -> DocumentArray:
        """
        Convert input text to DocumentArray for processing.
        
        Args:
            text: Input text in various formats
            
        Returns:
            DocumentArray ready for processing
        """
        # Import here to avoid issues if jina is not installed
        from docarray import Document, DocumentArray
        
        # Case 1: Already processed documents dict
        if isinstance(text, dict) and "documents" in text:
            docs = DocumentArray()
            for doc_data in text["documents"]:
                doc = Document(text=doc_data.get("text", ""))
                if "id" in doc_data:
                    doc.id = doc_data["id"]
                if "tags" in doc_data:
                    doc.tags = doc_data["tags"]
                docs.append(doc)
            return docs
        
        # Case 2: String
        elif isinstance(text, str):
            return DocumentArray([Document(text=text)])
        
        # Case 3: List of strings
        elif isinstance(text, list):
            return DocumentArray([Document(text=t) for t in text if isinstance(t, str)])
        
        # Case 4: Unknown format
        else:
            raise ValueError("Text must be a string, list of strings, or processed document")
    
    def _vector_search(self, query_embedding: List[float], corpus: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query embedding
            corpus: List of documents with embeddings
            top_k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        
        # Calculate cosine similarity for each document
        results_with_scores = []
        for doc in corpus:
            doc_vec = np.array(doc["embedding"])
            
            # Calculate cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            
            results_with_scores.append({
                "id": doc.get("id", ""),
                "text": doc.get("text", ""),
                "score": float(similarity)
            })
        
        # Sort by score (descending) and take top_k
        results_with_scores.sort(key=lambda x: x["score"], reverse=True)
        return results_with_scores[:top_k]