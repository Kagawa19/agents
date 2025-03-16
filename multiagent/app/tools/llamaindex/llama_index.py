"""
LlamaIndex main tool integration.
Provides a unified interface for RAG operations using LlamaIndex.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

# Import component modules
from multiagent.app.tools.llamaindex.document_store import DocumentStore
from multiagent.app.tools.llamaindex.retriever import Retriever
from multiagent.app.tools.llamaindex.query_router import QueryRouter

logger = logging.getLogger(__name__)

class LlamaIndexTool:
    """
    Main tool for LlamaIndex integration.
    Provides a unified interface for document indexing, retrieval, and querying.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the LlamaIndex tool.
        
        Args:
            config: Configuration parameters
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.config = config
        self.tracer = tracer
        
        # Set up paths
        self.index_store_path = config.get("index_store_path", "./storage/indices")
        self.document_store_path = config.get("document_store_path", "./storage/documents")
        
        # Create directories
        os.makedirs(self.index_store_path, exist_ok=True)
        os.makedirs(self.document_store_path, exist_ok=True)
        
        # Initialize components
        self.document_store = DocumentStore(self.document_store_path)
        self.retriever = Retriever(tracer=tracer, config=config.get("retriever_config", {}))
        self.query_router = QueryRouter(tracer=tracer, config=config.get("router_config", {}))
        
        # Track indices
        self.indices = {}
        self._load_indices()
    
    def _load_indices(self) -> None:
        """Load available indices from the index store."""
        try:
            # Check if index registry exists
            registry_path = os.path.join(self.index_store_path, "index_registry.json")
            if os.path.exists(registry_path):
                with open(registry_path, "r", encoding="utf-8") as f:
                    self.indices = json.load(f)
                logger.info(f"Loaded {len(self.indices)} indices from registry")
            else:
                # Create empty registry
                self.indices = {}
                with open(registry_path, "w", encoding="utf-8") as f:
                    json.dump(self.indices, f, indent=2)
                logger.info("Created new index registry")
                
        except Exception as e:
            logger.error(f"Error loading index registry: {str(e)}")
            self.indices = {}
    
    def _save_indices(self) -> None:
        """Save index registry to disk."""
        try:
            registry_path = os.path.join(self.index_store_path, "index_registry.json")
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(self.indices, f, indent=2)
            logger.info("Saved index registry")
        except Exception as e:
            logger.error(f"Error saving index registry: {str(e)}")
    
    async def initialize(self) -> bool:
        """
        Initialize LlamaIndex components.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        span = None
        if self.tracer:
            span = self.tracer.span(name="llamaindex_initialize")
        
        try:
            # Try importing LlamaIndex
            from llama_index import ServiceContext, set_global_service_context
            
            # Configure LlamaIndex
            service_context = self._create_service_context()
            if service_context:
                set_global_service_context(service_context)
                logger.info("LlamaIndex components initialized successfully")
                
                if span:
                    span.update(output={"status": "success"})
                
                return True
            else:
                logger.error("Failed to create service context")
                
                if span:
                    span.update(output={"error": "Failed to create service context"})
                
                return False
            
        except ImportError as e:
            logger.error(f"Missing dependencies for LlamaIndex: {str(e)}")
            
            if span:
                span.update(output={"error": f"Missing dependencies: {str(e)}"})
            
            return False
            
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return False
    
    def _create_service_context(self) -> Any:
        """
        Create a LlamaIndex ServiceContext.
        
        Returns:
            LlamaIndex ServiceContext or None if creation fails
        """
        try:
            # Import ServiceContext
            from llama_index import ServiceContext
            
            # Get LLM provider config
            llm_provider = self.config.get("llm_provider", "openai")
            
            if llm_provider == "openai":
                from llama_index.llms import OpenAI
                from llama_index.embeddings import OpenAIEmbedding
                
                # Create OpenAI client
                llm = OpenAI(
                    model=self.config.get("openai_model", "gpt-3.5-turbo"),
                    api_key=self.config.get("openai_api_key")
                )
                
                # Create embedding model
                embed_model = OpenAIEmbedding(
                    model=self.config.get("openai_embedding_model", "text-embedding-ada-002"),
                    api_key=self.config.get("openai_api_key")
                )
                
            elif llm_provider == "huggingface":
                from llama_index.llms import HuggingFaceLLM
                from llama_index.embeddings import HuggingFaceEmbedding
                
                # Create HuggingFace client
                llm = HuggingFaceLLM(
                    model_name=self.config.get("hf_model", "google/flan-t5-small"),
                    tokenizer_name=self.config.get("hf_tokenizer", "google/flan-t5-small")
                )
                
                # Create embedding model
                embed_model = HuggingFaceEmbedding(
                    model_name=self.config.get("hf_embedding_model", "sentence-transformers/all-mpnet-base-v2")
                )
                
            else:
                logger.error(f"Unsupported LLM provider: {llm_provider}")
                return None
            
            # Create service context
            service_context = ServiceContext.from_defaults(
                llm=llm,
                embed_model=embed_model,
                chunk_size=self.config.get("chunk_size", 1024),
                chunk_overlap=self.config.get("chunk_overlap", 20)
            )
            
            return service_context
            
        except Exception as e:
            logger.error(f"Error creating service context: {str(e)}")
            return None
    
    async def create_index(
        self,
        documents: List[Dict[str, Any]],
        index_name: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create an index from documents.
        
        Args:
            documents: List of documents to index
            index_name: Name of the index
            description: Description of the index
            
        Returns:
            Index information
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="llamaindex_create_index",
                input={"index_name": index_name, "document_count": len(documents)}
            )
        
        try:
            # Initialize LlamaIndex if needed
            if not await self.initialize():
                raise ValueError("Failed to initialize LlamaIndex")
            
            # Import required LlamaIndex classes
            from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
            from llama_index.storage.storage_context import StorageContext
            from llama_index.vector_stores import SimpleVectorStore
            
            # Process documents
            processed_docs = []
            for doc in documents:
                # Extract content and metadata
                content = doc.get("content", "") or doc.get("text", "")
                metadata = doc.get("metadata", {})
                
                # Skip empty documents
                if not content:
                    continue
                
                # Create LlamaIndex document
                llama_doc = Document(
                    text=content,
                    metadata=metadata
                )
                processed_docs.append(llama_doc)
            
            # Create vector store and storage context
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index
            index = VectorStoreIndex.from_documents(
                documents=processed_docs,
                storage_context=storage_context,
                show_progress=True
            )
            
            # Save index
            index_path = os.path.join(self.index_store_path, index_name)
            os.makedirs(index_path, exist_ok=True)
            index.storage_context.persist(persist_dir=index_path)
            
            # Update index registry
            self.indices[index_name] = {
                "name": index_name,
                "description": description,
                "document_count": len(processed_docs),
                "created_at": __import__('datetime').datetime.utcnow().isoformat(),
                "path": index_path
            }
            self._save_indices()
            
            result = {
                "index_name": index_name,
                "document_count": len(processed_docs),
                "status": "success"
            }
            
            if span:
                span.update(output=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"status": "error", "error": str(e)}
    
    async def query_index(
        self,
        query: str,
        index_name: str,
        similarity_top_k: int = 5,
        reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Query an index with a natural language query.
        
        Args:
            query: Natural language query
            index_name: Name of the index to query
            similarity_top_k: Number of similar documents to retrieve
            reranking: Whether to rerank results
            
        Returns:
            Query response with retrieved context and answer
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="llamaindex_query_index",
                input={"index_name": index_name, "query": query}
            )
        
        try:
            # Check if index exists
            if index_name not in self.indices:
                raise ValueError(f"Index '{index_name}' not found")
            
            # Import required LlamaIndex classes
            from llama_index import StorageContext, load_index_from_storage
            
            # Load index
            index_path = self.indices[index_name]["path"]
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
            
            # Create query engine
            query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
            
            # Execute query
            response = query_engine.query(query)
            
            # Extract source nodes
            source_nodes = response.source_nodes
            sources = []
            for node in source_nodes:
                sources.append({
                    "text": node.node.text,
                    "metadata": node.node.metadata,
                    "score": node.score if hasattr(node, "score") else None
                })
            
            # Apply reranking if requested
            if reranking and len(sources) > 1:
                # Create a simpler representation for reranking
                documents = [
                    {
                        "content": source["text"],
                        "metadata": source["metadata"],
                        "score": source["score"]
                    }
                    for source in sources
                ]
                
                # Get query embedding
                from llama_index import ServiceContext
                service_context = ServiceContext.from_defaults()
                embed_model = service_context.embed_model
                query_embedding = embed_model.get_text_embedding(query)
                
                # Get document embeddings
                document_embeddings = [
                    embed_model.get_text_embedding(doc["content"])
                    for doc in documents
                ]
                
                # Rerank using MMR for diversity
                reranked_docs = await self.retriever.mmr_reranking(
                    query_embedding=query_embedding,
                    documents=documents,
                    document_embeddings=document_embeddings,
                    top_k=similarity_top_k,
                    lambda_param=0.7  # Balance between relevance and diversity
                )
                
                # Update sources with reranked documents
                sources = [
                    {
                        "text": doc["content"],
                        "metadata": doc["metadata"],
                        "score": doc["score"]
                    }
                    for doc in reranked_docs
                ]
            
            result = {
                "query": query,
                "answer": str(response),
                "sources": sources,
                "index_name": index_name
            }
            
            if span:
                span.update(output={"status": "success"})
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"query": query, "error": str(e)}
    
    async def retrieve_documents(
        self,
        query: str,
        index_name: str,
        retrieval_strategy: str = "semantic",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            index_name: Name of the index to search
            retrieval_strategy: Strategy for retrieval (semantic, hybrid, mmr)
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="llamaindex_retrieve_documents",
                input={"index_name": index_name, "query": query, "strategy": retrieval_strategy}
            )
        
        try:
            # Check if index exists
            if index_name not in self.indices:
                raise ValueError(f"Index '{index_name}' not found")
            
            # Import required LlamaIndex classes
            from llama_index import ServiceContext, StorageContext, load_index_from_storage
            
            # Load index
            index_path = self.indices[index_name]["path"]
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
            
            # Get query embedding
            service_context = ServiceContext.from_defaults()
            embed_model = service_context.embed_model
            query_embedding = embed_model.get_text_embedding(query)
            
            # Create retriever
            retriever = index.as_retriever(similarity_top_k=top_k)
            
            # Retrieve nodes
            nodes = retriever.retrieve(query)
            
            # Format documents
            documents = []
            for node in nodes:
                documents.append({
                    "content": node.node.text,
                    "metadata": node.node.metadata,
                    "score": node.score if hasattr(node, "score") else None
                })
            
            # Apply specific retrieval strategy
            if retrieval_strategy == "hybrid" and len(documents) > 0:
                # Get document embeddings
                document_embeddings = [
                    embed_model.get_text_embedding(doc["content"])
                    for doc in documents
                ]
                
                # Apply hybrid search
                documents = await self.retriever.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    documents=documents,
                    document_embeddings=document_embeddings,
                    top_k=top_k,
                    alpha=0.5  # Equal weight to semantic and keyword matching
                )
                
            elif retrieval_strategy == "mmr" and len(documents) > 0:
                # Get document embeddings
                document_embeddings = [
                    embed_model.get_text_embedding(doc["content"])
                    for doc in documents
                ]
                
                # Apply MMR reranking
                documents = await self.retriever.mmr_reranking(
                    query_embedding=query_embedding,
                    documents=documents,
                    document_embeddings=document_embeddings,
                    top_k=top_k,
                    lambda_param=0.7  # Balance between relevance and diversity
                )
            
            if span:
                span.update(output={"document_count": len(documents)})
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def router_query(
        self,
        query: str,
        llm_client: Any = None
    ) -> Dict[str, Any]:
        """
        Use the query router to select the best index and retrieval strategy.
        
        Args:
            query: Query text
            llm_client: Optional LLM client for query rewriting
            
        Returns:
            Query result with routing information
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="llamaindex_router_query",
                input={"query": query}
            )
        
        try:
            # Get available indices
            available_indices = list(self.indices.keys())
            if not available_indices:
                return {
                    "query": query,
                    "error": "No indices available"
                }
            
            # Rewrite query if LLM client is available
            rewritten_query = query
            if llm_client:
                rewritten_query = await self.query_router.rewrite_query(query, llm_client)
            
            # Route query
            routing = await self.query_router.route_query(
                query=rewritten_query,
                available_indexes=available_indices
            )
            
            # Execute query with routing information
            result = await self.query_index(
                query=rewritten_query,
                index_name=routing["selected_index"],
                similarity_top_k=routing["top_k"],
                reranking=routing["retrieval_strategy"] == "mmr"
            )
            
            # Add routing information to result
            result["routing"] = routing
            result["original_query"] = query
            if rewritten_query != query:
                result["rewritten_query"] = rewritten_query
            
            if span:
                span.update(output={"status": "success", "routing": routing})
            
            return result
            
        except Exception as e:
            logger.error(f"Error in router query: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {
                "query": query,
                "error": str(e)
            }
    
    def list_indices(self) -> List[Dict[str, Any]]:
        """
        List available indices.
        
        Returns:
            List of index information
        """
        return list(self.indices.values())
    
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Check if index exists
            if index_name not in self.indices:
                logger.warning(f"Index '{index_name}' not found")
                return False
            
            # Get index path
            index_path = self.indices[index_name]["path"]
            
            # Remove index directory
            import shutil
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            
            # Remove from registry
            del self.indices[index_name]
            self._save_indices()
            
            logger.info(f"Deleted index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            return False