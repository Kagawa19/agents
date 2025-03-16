# tools/llama_index/llama_index.py
from typing import Dict, List, Optional, Any, Union
import logging
from pydantic import BaseModel, Field

# Import LlamaIndex libraries
try:
    from llama_index import (
        VectorStoreIndex, 
        SimpleDirectoryReader, 
        Document as LlamaDocument, 
        ServiceContext, 
        StorageContext,
        load_index_from_storage
    )
    from llama_index.node_parser import SimpleNodeParser
    from llama_index.retrievers import VectorIndexRetriever
    from llama_index.query_engine import RetrieverQueryEngine
    from llama_index.indices.postprocessor import SimilarityPostprocessor
    from llama_index.llms import LangChainLLM
    from llama_index.response_synthesizers import get_response_synthesizer
    from llama_index.callbacks import CallbackManager
except ImportError:
    logging.warning("LlamaIndex libraries not installed. Install with 'pip install llama-index'")

logger = logging.getLogger(__name__)

class QueryResult(BaseModel):
    """Result model for LlamaIndex queries."""
    response: str = ""
    source_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    query_plan: Optional[Dict[str, Any]] = None

class LlamaIndexTool:
    """Tool for using LlamaIndex for advanced document indexing and retrieval."""
    
    def __init__(
        self,
        storage_dir: str = "./llama_index_storage",
        model_name: Optional[str] = None,
        embed_model: str = "default",
        callback_manager: Optional[Any] = None
    ):
        """
        Initialize the LlamaIndex tool.
        
        Args:
            storage_dir: Directory for storage
            model_name: Optional LLM model name
            embed_model: Embedding model to use
            callback_manager: Optional callback manager for logging
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.embed_model = embed_model
        self.callback_manager = callback_manager or CallbackManager([])
        
        # Create storage directory if it doesn't exist
        import os
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize service context
        self.service_context = self._create_service_context()
        
        # Store indices and retrievers
        self._indices = {}
        self._retrievers = {}
        
    def _create_service_context(self) -> "ServiceContext":
        """
        Create a ServiceContext for LlamaIndex.
        
        Returns:
            ServiceContext instance
        """
        # Create a service context with the specified model and embed model
        try:
            if self.model_name:
                from langchain.llms import OpenAI
                # You can replace this with any LLM provider, including Bedrock
                llm = LangChainLLM(llm=OpenAI(model_name=self.model_name))
            else:
                llm = None
                
            # Create service context
            service_context = ServiceContext.from_defaults(
                llm=llm,
                embed_model=self.embed_model,
                node_parser=SimpleNodeParser.from_defaults(),
                callback_manager=self.callback_manager
            )
            return service_context
            
        except Exception as e:
            logger.error(f"Error creating service context: {str(e)}")
            # Return default service context
            return ServiceContext.from_defaults()
            
    def _convert_to_llama_documents(self, documents: List[Dict[str, Any]]) -> List["LlamaDocument"]:
        """
        Convert dictionary documents to LlamaIndex Document objects.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of LlamaIndex Document objects
        """
        llama_docs = []
        
        for doc in documents:
            # Extract text and metadata
            text = doc.get("text", "") or doc.get("content", "")
            metadata = doc.get("metadata", {})
            doc_id = doc.get("id", None)
            
            # Create LlamaIndex document
            llama_doc = LlamaDocument(
                text=text,
                metadata=metadata,
                doc_id=doc_id
            )
            
            llama_docs.append(llama_doc)
            
        return llama_docs
        
    def _get_or_create_index(self, index_name: str, documents: Optional[List[Dict[str, Any]]] = None) -> "VectorStoreIndex":
        """
        Get an existing index or create a new one.
        
        Args:
            index_name: Name of the index
            documents: Optional documents to index
            
        Returns:
            VectorStoreIndex instance
        """
        # Check if index already exists
        if index_name in self._indices:
            return self._indices[index_name]
            
        # Check if index exists on disk
        index_dir = os.path.join(self.storage_dir, index_name)
        if os.path.exists(index_dir) and os.path.isdir(index_dir):
            try:
                # Load existing index
                storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                index = load_index_from_storage(storage_context)
                self._indices[index_name] = index
                return index
            except Exception as e:
                logger.error(f"Error loading index {index_name}: {str(e)}")
                # If loading fails, create a new index
        
        # Create a new index if documents are provided
        if documents:
            llama_docs = self._convert_to_llama_documents(documents)
            
            # Create index
            index = VectorStoreIndex.from_documents(
                documents=llama_docs,
                service_context=self.service_context
            )
            
            # Save index
            os.makedirs(index_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=index_dir)
            
            # Cache index
            self._indices[index_name] = index
            return index
        else:
            raise ValueError(f"Index {index_name} does not exist and no documents provided to create it")
            
    def _get_retriever(self, index: "VectorStoreIndex", similarity_top_k: int = 4) -> "VectorIndexRetriever":
        """
        Get a retriever for an index.
        
        Args:
            index: VectorStoreIndex instance
            similarity_top_k: Number of similar items to retrieve
            
        Returns:
            VectorIndexRetriever instance
        """
        return index.as_retriever(similarity_top_k=similarity_top_k)
        
    def process_query(
        self,
        query: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        index_name: str = "default",
        similarity_top_k: int = 4,
        rerank: bool = True,
        response_mode: str = "compact",
    ) -> Dict[str, Any]:
        """
        Process a query using LlamaIndex.
        
        Args:
            query: The query string
            documents: Optional documents to index or update
            index_name: Name of the index to use
            similarity_top_k: Number of similar items to retrieve
            rerank: Whether to rerank results by similarity
            response_mode: Response synthesis mode
            
        Returns:
            QueryResult containing response and supporting information
        """
        try:
            # Get or create index
            index = self._get_or_create_index(index_name, documents)
            
            # Get retriever
            retriever = self._get_retriever(index, similarity_top_k)
            
            # Create postprocessors
            postprocessors = []
            if rerank:
                postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.7))
                
            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode=response_mode,
                service_context=self.service_context
            )
            
            # Create query engine
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=postprocessors
            )
            
            # Execute query
            response = query_engine.query(query)
            
            # Extract source nodes and format
            source_nodes = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    source_nodes.append({
                        "text": node.node.text,
                        "metadata": node.node.metadata,
                        "score": node.score if hasattr(node, "score") else None,
                        "id": node.node.node_id
                    })
            
            # Convert retrieved documents to a common format
            formatted_docs = []
            for node in source_nodes:
                formatted_docs.append({
                    "text": node["text"],
                    "metadata": node["metadata"],
                    "score": node["score"],
                    "id": node["id"]
                })
                
            # Get query plan if available
            query_plan = None
            if hasattr(response, "metadata") and "query_plan" in response.metadata:
                query_plan = response.metadata["query_plan"]
                
            return {
                "response": str(response),
                "source_nodes": source_nodes,
                "documents": formatted_docs,
                "query_plan": query_plan
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "source_nodes": [],
                "documents": [],
                "query_plan": None
            }
            
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        index_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Index documents with LlamaIndex.
        
        Args:
            documents: List of document dictionaries
            index_name: Name of the index
            
        Returns:
            Indexing result
        """
        try:
            # Convert documents
            llama_docs = self._convert_to_llama_documents(documents)
            
            # Get index directory
            index_dir = os.path.join(self.storage_dir, index_name)
            os.makedirs(index_dir, exist_ok=True)
            
            # Check if index exists
            existing_index = None
            if os.path.exists(os.path.join(index_dir, "docstore.json")):
                # Load existing index
                try:
                    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                    existing_index = load_index_from_storage(storage_context)
                except Exception as e:
                    logger.warning(f"Could not load existing index, creating new one: {str(e)}")
            
            if existing_index:
                # Update existing index
                for doc in llama_docs:
                    existing_index.insert(doc)
                
                # Save updated index
                existing_index.storage_context.persist(persist_dir=index_dir)
                self._indices[index_name] = existing_index
            else:
                # Create new index
                index = VectorStoreIndex.from_documents(
                    documents=llama_docs,
                    service_context=self.service_context
                )
                
                # Save index
                index.storage_context.persist(persist_dir=index_dir)
                self._indices[index_name] = index
                
            return {
                "status": "success",
                "message": f"Indexed {len(documents)} documents in {index_name}",
                "index_name": index_name,
                "document_count": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return {
                "status": "error",
                "message": f"Error indexing documents: {str(e)}",
                "index_name": index_name
            }
            
    def list_indices(self) -> List[str]:
        """
        List all available indices.
        
        Returns:
            List of index names
        """
        indices = []
        
        # Check directory for indices
        if os.path.exists(self.storage_dir):
            for item in os.listdir(self.storage_dir):
                item_path = os.path.join(self.storage_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "docstore.json")):
                    indices.append(item)
                    
        return indices
        
    def delete_index(self, index_name: str) -> Dict[str, Any]:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            Deletion result
        """
        try:
            # Check if index exists
            index_dir = os.path.join(self.storage_dir, index_name)
            if not os.path.exists(index_dir):
                return {
                    "status": "error",
                    "message": f"Index {index_name} does not exist"
                }
                
            # Remove from cache
            if index_name in self._indices:
                del self._indices[index_name]
                
            # Delete directory
            import shutil
            shutil.rmtree(index_dir)
            
            return {
                "status": "success",
                "message": f"Index {index_name} deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            return {
                "status": "error",
                "message": f"Error deleting index: {str(e)}"
            }
        
    def generate_query_plan(self, query: str, index_name: str = "default") -> Dict[str, Any]:
        """
        Generate a query plan for complex queries.
        
        Args:
            query: The complex query
            index_name: Name of the index to use
            
        Returns:
            Query plan
        """
        try:
            # This is a simplified version - a real implementation would use
            # LlamaIndex's query planning capabilities
            
            # Analyze the query
            query_parts = self._analyze_query(query)
            
            # Generate a plan based on query parts
            plan = {
                "original_query": query,
                "steps": [],
                "index": index_name
            }
            
            # Add steps based on query parts
            for i, part in enumerate(query_parts):
                plan["steps"].append({
                    "step_id": i + 1,
                    "query": part,
                    "retrieval_type": "vector" if i < len(query_parts) - 1 else "hybrid",
                    "top_k": 3 if i < len(query_parts) - 1 else 5
                })
                
            return plan
            
        except Exception as e:
            logger.error(f"Error generating query plan: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating query plan: {str(e)}"
            }
            
    def _analyze_query(self, query: str) -> List[str]:
        """
        Simple query analysis to break complex queries into parts.
        
        Args:
            query: The complex query
            
        Returns:
            List of query parts
        """
        # This is a very simplified implementation
        # A real implementation would use more sophisticated techniques
        
        # Check if query contains multiple questions
        if "?" in query:
            parts = [p.strip() + "?" for p in query.split("?") if p.strip()]
            if len(parts) > 1:
                return parts
                
        # Check for "and" connectors
        if " and " in query.lower():
            parts = [p.strip() for p in query.lower().split(" and ") if p.strip()]
            if len(parts) > 1:
                return parts
                
        # Default to single query
        return [query]

# Example usage
if __name__ == "__main__":
    # Initialize the tool
    llama_index_tool = LlamaIndexTool()
    
    # Index some documents
    documents = [
        {"text": "LlamaIndex is a data framework for LLM applications to ingest, structure, and access private or domain-specific data.", "metadata": {"source": "docs", "topic": "llama_index"}},
        {"text": "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM outputs by retrieving relevant context from external sources.", "metadata": {"source": "docs", "topic": "rag"}}
    ]
    
    index_result = llama_index_tool.index_documents(documents, "test_index")
    print(f"Index result: {index_result}")
    
    # Query the index
    query_result = llama_index_tool.process_query("What is LlamaIndex?", index_name="test_index")
    print(f"Query result: {query_result}")