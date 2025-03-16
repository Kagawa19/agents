"""
Document store for LlamaIndex.
Provides persistent storage for documents and metadata.
"""

import logging
import os
import json
import uuid
import pickle
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentStore:
    """
    Store for managing documents and their metadata.
    Provides methods for adding, retrieving, and querying documents.
    """
    
    def __init__(self, store_path: str):
        """
        Initialize the document store.
        
        Args:
            store_path: Path to store documents
        """
        self.store_path = store_path
        self.documents_path = os.path.join(store_path, "documents")
        self.metadata_path = os.path.join(store_path, "metadata")
        self.index_path = os.path.join(store_path, "index")
        
        # Create directories if they don't exist
        os.makedirs(self.documents_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)
        
        # Initialize metadata index
        self.metadata_index = self._load_metadata_index()
    
    def _load_metadata_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load metadata index from disk.
        
        Returns:
            Metadata index mapping document IDs to metadata
        """
        index_file = os.path.join(self.index_path, "metadata_index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata index: {str(e)}")
                return {}
        return {}
    
    def _save_metadata_index(self) -> None:
        """Save metadata index to disk."""
        index_file = os.path.join(self.index_path, "metadata_index.json")
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata index: {str(e)}")
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the store.
        
        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID
            
        Returns:
            Document ID
        """
        # Generate document ID if not provided
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add system metadata
        metadata.update({
            "doc_id": doc_id,
            "added_at": datetime.utcnow().isoformat(),
            "content_length": len(content)
        })
        
        try:
            # Save document content
            doc_file = os.path.join(self.documents_path, f"{doc_id}.txt")
            with open(doc_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Save document metadata
            meta_file = os.path.join(self.metadata_path, f"{doc_id}.json")
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            # Update metadata index
            self.metadata_index[doc_id] = metadata
            self._save_metadata_index()
            
            logger.info(f"Added document with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document with content and metadata
        """
        try:
            # Check if document exists
            doc_file = os.path.join(self.documents_path, f"{doc_id}.txt")
            meta_file = os.path.join(self.metadata_path, f"{doc_id}.json")
            
            if not os.path.exists(doc_file) or not os.path.exists(meta_file):
                logger.warning(f"Document {doc_id} not found")
                return None
            
            # Load document content
            with open(doc_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Load document metadata
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            return {
                "doc_id": doc_id,
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {str(e)}")
            return None
    
    def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document in the store.
        
        Args:
            doc_id: Document ID
            content: New document content
            metadata: New document metadata
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Check if document exists
            doc = self.get_document(doc_id)
            if not doc:
                logger.warning(f"Document {doc_id} not found for update")
                return False
            
            # Update content if provided
            if content is not None:
                doc_file = os.path.join(self.documents_path, f"{doc_id}.txt")
                with open(doc_file, "w", encoding="utf-8") as f:
                    f.write(content)
                
                # Update content length in metadata
                if doc["metadata"]:
                    doc["metadata"]["content_length"] = len(content)
            
            # Update metadata if provided
            if metadata is not None:
                # Merge with existing metadata
                if doc["metadata"]:
                    doc["metadata"].update(metadata)
                else:
                    doc["metadata"] = metadata
                
                # Add update timestamp
                doc["metadata"]["updated_at"] = datetime.utcnow().isoformat()
                
                # Save updated metadata
                meta_file = os.path.join(self.metadata_path, f"{doc_id}.json")
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(doc["metadata"], f, indent=2)
                
                # Update metadata index
                self.metadata_index[doc_id] = doc["metadata"]
                self._save_metadata_index()
            
            logger.info(f"Updated document with ID: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Check if document exists
            doc_file = os.path.join(self.documents_path, f"{doc_id}.txt")
            meta_file = os.path.join(self.metadata_path, f"{doc_id}.json")
            
            # Delete document files
            if os.path.exists(doc_file):
                os.remove(doc_file)
            
            if os.path.exists(meta_file):
                os.remove(meta_file)
            
            # Remove from metadata index
            if doc_id in self.metadata_index:
                del self.metadata_index[doc_id]
                self._save_metadata_index()
            
            logger.info(f"Deleted document with ID: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        List documents in the store.
        
        Args:
            limit: Maximum number of documents to return
            offset: Offset for pagination
            filter_criteria: Criteria for filtering documents
            
        Returns:
            List of document metadata
        """
        try:
            # Get all document IDs
            doc_ids = list(self.metadata_index.keys())
            
            # Apply filtering if criteria provided
            if filter_criteria:
                filtered_ids = []
                for doc_id, metadata in self.metadata_index.items():
                    if self._matches_criteria(metadata, filter_criteria):
                        filtered_ids.append(doc_id)
                doc_ids = filtered_ids
            
            # Apply pagination
            paginated_ids = doc_ids[offset:offset + limit]
            
            # Get document metadata
            documents = []
            for doc_id in paginated_ids:
                metadata = self.metadata_index.get(doc_id, {})
                documents.append({
                    "doc_id": doc_id,
                    "metadata": metadata
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
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
            
            # Handle simple keys
            elif key not in metadata or metadata[key] != value:
                return False
        
        return True
    
    def search_by_metadata(
        self,
        criteria: Dict[str, Any],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata criteria.
        
        Args:
            criteria: Metadata criteria
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        try:
            # Find matching document IDs
            matching_docs = []
            for doc_id, metadata in self.metadata_index.items():
                if self._matches_criteria(metadata, criteria):
                    # Get full document
                    doc = self.get_document(doc_id)
                    if doc:
                        matching_docs.append(doc)
                
                # Apply limit
                if len(matching_docs) >= limit:
                    break
            
            return matching_docs
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the document store.
        
        Returns:
            Statistics dictionary
        """
        try:
            # Count documents
            doc_count = len(self.metadata_index)
            
            # Calculate total content size
            total_size = 0
            for doc_id in self.metadata_index:
                doc_file = os.path.join(self.documents_path, f"{doc_id}.txt")
                if os.path.exists(doc_file):
                    total_size += os.path.getsize(doc_file)
            
            return {
                "document_count": doc_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size > 0 else 0,
                "store_path": self.store_path
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {
                "document_count": 0,
                "error": str(e)
            }