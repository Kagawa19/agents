"""
Document indexing for Jina.
Manages document indexing and storage.
"""

import logging
import aiohttp
import json
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class JinaIndex:
    """
    Manages document indexing and storage in Jina.
    Provides methods for indexing documents and managing indexes.
    """
    
    def __init__(
        self,
        api_key: str,
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the Jina Index manager.
        
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
    
    async def create_index(
        self,
        index_name: str,
        dimension: int = 768,
        metric: str = "cosine",
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a vector index.
        
        Args:
            index_name: Name of the index
            dimension: Dimension of embeddings
            metric: Distance metric (cosine, euclidean, dot)
            description: Description of the index
            
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
                "metric": metric,
                "description": description
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
    
    async def delete_index(self, index_name: str) -> Dict[str, Any]:
        """
        Delete a vector index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Deletion status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_delete_index",
                input={"index_name": index_name}
            )
        
        try:
            # Prepare request
            url = f"{self.base_url}/indexes/{index_name}"
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    url,
                    headers=self.headers
                ) as response:
                    if response.status not in [200, 204]:
                        error_text = await response.text()
                        logger.error(f"Jina API error ({response.status}): {error_text}")
                        if span:
                            span.update(output={"error": error_text})
                        return {"error": error_text}
                    
                    # Parse response
                    try:
                        result = await response.json()
                    except:
                        result = {"status": "success"}
            
            if span:
                span.update(output={"status": "success"})
            
            return result
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def batch_delete_documents(
        self,
        index_name: str,
        doc_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Delete multiple documents from an index.
        
        Args:
            index_name: Name of the index
            doc_ids: List of document IDs
            
        Returns:
            Deletion status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_batch_delete_documents",
                input={"index_name": index_name, "doc_count": len(doc_ids)}
            )
        
        try:
            # Check for empty inputs
            if not doc_ids:
                return {"status": "success", "count": 0}
            
            # Prepare request
            url = f"{self.base_url}/indexes/{index_name}/documents/delete"
            payload = {
                "ids": doc_ids
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
            logger.error(f"Error batch deleting documents: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def build_index_from_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        dimension: int = 768,
        metric: str = "cosine",
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Build an index from documents in one operation.
        
        Args:
            index_name: Name of the index
            documents: List of documents
            dimension: Dimension of embeddings
            metric: Distance metric
            description: Description of the index
            
        Returns:
            Index creation and document upload status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_build_index",
                input={"index_name": index_name, "document_count": len(documents)}
            )
        
        try:
            # Create index
            create_result = await self.create_index(
                index_name=index_name,
                dimension=dimension,
                metric=metric,
                description=description
            )
            
            if "error" in create_result:
                if "already exists" in str(create_result["error"]).lower():
                    logger.info(f"Index '{index_name}' already exists, continuing with document upload")
                else:
                    if span:
                        span.update(output={"error": create_result["error"]})
                    return create_result
            
            # Upsert documents
            upsert_result = await self.upsert_documents(
                index_name=index_name,
                documents=documents
            )
            
            if "error" in upsert_result:
                if span:
                    span.update(output={"error": upsert_result["error"]})
                return upsert_result
            
            # Return combined result
            result = {
                "index_name": index_name,
                "index_created": "error" not in create_result,
                "documents_inserted": upsert_result.get("count", 0),
                "status": "success"
            }
            
            if span:
                span.update(output=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}

                        result = {"status": "success"}
            
            if span:
                span.update(output={"status": "success"})
            
            return result
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def list_indexes(self) -> List[Dict[str, Any]]:
        """
        List all indexes.
        
        Returns:
            List of indexes
        """
        span = None
        if self.tracer:
            span = self.tracer.span(name="jina_list_indexes")
        
        try:
            # Prepare request
            url = f"{self.base_url}/indexes"
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Jina API error ({response.status}): {error_text}")
                        if span:
                            span.update(output={"error": error_text})
                        return []
                    
                    # Parse response
                    result = await response.json()
            
            # Extract indexes
            indexes = result.get("indexes", [])
            
            if span:
                span.update(output={"index_count": len(indexes)})
            
            return indexes
            
        except Exception as e:
            logger.error(f"Error listing indexes: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def get_index(self, index_name: str) -> Dict[str, Any]:
        """
        Get information about an index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Index information
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_get_index",
                input={"index_name": index_name}
            )
        
        try:
            # Prepare request
            url = f"{self.base_url}/indexes/{index_name}"
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers
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
            logger.error(f"Error getting index: {str(e)}")
            
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
            
            # Format documents if needed
            formatted_docs = []
            for doc in documents:
                if "vector" not in doc and "embedding" in doc:
                    doc["vector"] = doc.pop("embedding")
                
                if "id" not in doc:
                    doc["id"] = str(__import__("uuid").uuid4())
                
                formatted_docs.append(doc)
            
            payload = {
                "documents": formatted_docs
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
    
    async def get_document(
        self,
        index_name: str,
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Get a document from an index.
        
        Args:
            index_name: Name of the index
            doc_id: Document ID
            
        Returns:
            Document information
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_get_document",
                input={"index_name": index_name, "doc_id": doc_id}
            )
        
        try:
            # Prepare request
            url = f"{self.base_url}/indexes/{index_name}/documents/{doc_id}"
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers
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
            logger.error(f"Error getting document: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def delete_document(
        self,
        index_name: str,
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Delete a document from an index.
        
        Args:
            index_name: Name of the index
            doc_id: Document ID
            
        Returns:
            Deletion status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_delete_document",
                input={"index_name": index_name, "doc_id": doc_id}
            )
        
        try:
            # Prepare request
            url = f"{self.base_url}/indexes/{index_name}/documents/{doc_id}"
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    url,
                    headers=self.headers
                ) as response:
                    if response.status not in [200, 204]:
                        error_text = await response.text()
                        logger.error(f"Jina API error ({response.status}): {error_text}")
                        if span:
                            span.update(output={"error": error_text})
                        return {"error": error_text}
                    
                    # Parse response
                    try:
                        result = await response.json()
                    except: