"""
Retrieval-Augmented Generation (RAG) pipeline.
Connects search, retrieval, and generation components.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

# Import necessary tools
from multiagent.app.tools import serper, scraper, jina, bedrock, llamaindex
from multiagent.app.tools.normalizer import normalizer

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Provides a complete workflow for RAG-based systems.
    """
    
    def __init__(
        self,
        tracer: Optional[LangfuseTracer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            tracer: Optional LangfuseTracer instance for monitoring
            config: Configuration parameters
        """
        self.tracer = tracer
        self.config = config or {}
        
        # Initialize components
        self.serper_tool = None
        self.scraper_tool = None
        self.jina_tool = None
        self.bedrock_tool = None
        self.llamaindex_tool = None
        
        # Load components based on configuration
        self._load_components()
    
    def _load_components(self) -> None:
        """Load and initialize pipeline components."""
        try:
            # Load SerperTool if configured
            if self.config.get("serper_api_key"):
                self.serper_tool = serper.SerperTool(
                    api_key=self.config["serper_api_key"],
                    tracer=self.tracer
                )
                logger.info("SerperTool initialized")
            
            # Load ScraperTool
            self.scraper_tool = scraper.WebScraper(
                user_agent=self.config.get("user_agent"),
                tracer=self.tracer
            )
            logger.info("ScraperTool initialized")
            
            # Load JinaTool if configured
            if self.config.get("jina_api_key"):
                self.jina_tool = jina.JinaTool(
                    api_key=self.config["jina_api_key"],
                    tracer=self.tracer
                )
                logger.info("JinaTool initialized")
            
            # Load BedrockTool if configured
            if self.config.get("aws_access_key_id") and self.config.get("aws_secret_access_key"):
                self.bedrock_tool = bedrock.BedrockTool(
                    config={
                        "aws_access_key_id": self.config["aws_access_key_id"],
                        "aws_secret_access_key": self.config["aws_secret_access_key"],
                        "aws_region": self.config.get("aws_region", "us-east-1")
                    },
                    tracer=self.tracer
                )
                logger.info("BedrockTool initialized")
            
            # Load LlamaIndexTool if configured
            if self.config.get("llamaindex_enabled", False):
                self.llamaindex_tool = llamaindex.LlamaIndexTool(
                    config=self.config.get("llamaindex_config", {}),
                    tracer=self.tracer
                )
                logger.info("LlamaIndexTool initialized")
        
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline components: {str(e)}")
    
    async def search_and_retrieve(
        self,
        query: str,
        num_results: int = 5,
        scrape: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for information and retrieve content.
        
        Args:
            query: Query string
            num_results: Number of search results to retrieve
            scrape: Whether to scrape content from search results
            
        Returns:
            List of retrieved documents
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="rag_search_retrieve",
                input={"query": query, "num_results": num_results}
            )
        
        try:
            # Check if search tool is available
            if not self.serper_tool:
                raise ValueError("Search tool not available")
            
            # Search for information
            search_results = await self.serper_tool.search(
                query=query,
                num_results=num_results
            )
            
            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                if span:
                    span.update(output={"error": "No search results found"})
                return []
            
            # Skip scraping if not requested
            if not scrape:
                return [
                    {
                        "content": result.get("snippet", ""),
                        "metadata": {
                            "title": result.get("title", ""),
                            "url": result.get("link", ""),
                            "source": "search"
                        }
                    }
                    for result in search_results
                ]
            
            # Scrape content if requested
            if not self.scraper_tool:
                raise ValueError("Scraper tool not available")
            
            # Extract URLs
            urls = [result.get("link") for result in search_results if result.get("link")]
            
            # Scrape content in parallel
            documents = []
            for i, url in enumerate(urls):
                try:
                    # Skip non-web URLs
                    if not url.startswith(("http://", "https://")):
                        continue
                    
                    # Scrape content
                    content = await self.scraper_tool.scrape(url)
                    if content:
                        # Get metadata from search result
                        result = search_results[i]
                        
                        # Create document
                        document = {
                            "content": content,
                            "metadata": {
                                "url": url,
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", ""),
                                "position": i + 1,
                                "source": "web"
                            }
                        }
                        
                        # Normalize document
                        document = normalizer.standardize_document(document)
                        documents.append(document)
                
                except Exception as e:
                    logger.warning(f"Error scraping URL {url}: {str(e)}")
            
            if not documents:
                logger.warning(f"No content scraped for query: {query}")
                if span:
                    span.update(output={"error": "No content scraped"})
                return []
            
            if span:
                span.update(output={"document_count": len(documents)})
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in search and retrieve: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def process_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process documents for embedding and indexing.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Processed documents with embeddings
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="rag_process_documents",
                input={"document_count": len(documents)}
            )
        
        try:
            # Check if Jina tool is available
            if not self.jina_tool:
                logger.warning("Jina tool not available for embedding documents")
                if span:
                    span.update(output={"error": "Jina tool not available"})
                return documents
            
            # Extract content for embedding
            contents = [doc.get("content", "") for doc in documents]
            
            # Get embeddings
            embeddings = await self.jina_tool.get_embeddings(contents)
            
            # Add embeddings to documents
            processed_docs = []
            for i, doc in enumerate(documents):
                if i < len(embeddings):
                    doc_copy = doc.copy()
                    doc_copy["embedding"] = embeddings[i]
                    processed_docs.append(doc_copy)
                else:
                    processed_docs.append(doc)
            
            if span:
                span.update(output={"document_count": len(processed_docs)})
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return documents
    
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        model: str = "claude"
    ) -> str:
        """
        Generate a response using context and query.
        
        Args:
            query: User query
            context: Context documents
            model: Model to use for generation
            
        Returns:
            Generated response
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="rag_generate_response",
                input={"query": query, "context_count": len(context)}
            )
        
        try:
            # Check if generation tool is available
            if not self.bedrock_tool:
                raise ValueError("Generation tool not available")
            
            # Prepare context
            context_text = ""
            for i, doc in enumerate(context):
                content = doc.get("content", "")
                if content:
                    # Truncate long content
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    
                    # Add metadata
                    meta = doc.get("metadata", {})
                    source = meta.get("title", "") or meta.get("url", "")
                    
                    context_text += f"\n\nDocument {i+1}"
                    if source:
                        context_text += f" (Source: {source})"
                    context_text += f":\n{content}"
            
            # Create prompt
            prompt = f"""
            Answer the following question based on the provided context. If the context doesn't contain relevant information, acknowledge this limitation.
            
            Question: {query}
            
            Context:
            {context_text}
            
            Answer:
            """
            
            # Generate response
            response = await self.bedrock_tool.generate_text(
                prompt=prompt,
                model=model,
                max_tokens=500,
                temperature=0.3
            )
            
            if span:
                span.update(output={"status": "success"})
            
            return response.get("text", "")
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return f"Error generating response: {str(e)}"
    
    async def run_rag_pipeline(
        self,
        query: str,
        search_results: int = 5,
        model: str = "claude"
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.
        
        Args:
            query: User query
            search_results: Number of search results to retrieve
            model: Model to use for generation
            
        Returns:
            Pipeline result with retrieved documents and generated answer
        """
        start_time = time.time()
        
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="rag_pipeline",
                input={"query": query, "search_results": search_results, "model": model}
            )
        
        try:
            # Search and retrieve documents
            documents = await self.search_and_retrieve(
                query=query,
                num_results=search_results,
                scrape=True
            )
            
            if not documents:
                return {
                    "query": query,
                    "answer": "No relevant information found. Please try reformulating your query.",
                    "documents": [],
                    "execution_time": time.time() - start_time
                }
            
            # Process documents
            processed_docs = await self.process_documents(documents)
            
            # Generate response
            answer = await self.generate_response(
                query=query,
                context=processed_docs,
                model=model
            )
            
            # Prepare result
            result = {
                "query": query,
                "answer": answer,
                "documents": [
                    {
                        "content": doc.get("content", "")[:200] + "...",
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in processed_docs
                ],
                "document_count": len(processed_docs),
                "execution_time": time.time() - start_time
            }
            
            if span:
                span.update(output={"status": "success", "document_count": len(processed_docs)})
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {
                "query": query,
                "error": str(e),
                "execution_time": time.time() - start_time
            }