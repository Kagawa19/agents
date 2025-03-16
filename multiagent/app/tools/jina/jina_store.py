"""
Vector storage for Jina.
Provides methods for managing vector storage and index persistence.
"""

import logging
import os
import json
import shutil
import aiohttp
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class JinaStore:
    """
    Vector storage utilities for Jina.
    Provides methods for managing vector storage and index persistence.
    """
    
    def __init__(
        self,
        api_key: str,
        storage_path: str = "./storage/jina",
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the Jina Store utilities.
        
        Args:
            api_key: Jina AI API key
            storage_path: Path for local storage
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.api_key = api_key
        self.tracer = tracer
        self.storage_path = storage_path
        self.base_url = "https://api.jina.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Create storage directories
        self.indexes_path = os.path.join(storage_path, "indexes")
        self.backup_path = os.path.join(storage_path, "backups")
        os.makedirs(self.indexes_path, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
        
        # Load index registry
        self.index_registry = self._load_index_registry()
    
    def _load_index_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Load index registry from disk.
        
        Returns:
            Dictionary mapping index names to metadata
        """
        registry_file = os.path.join(self.storage_path, "index_registry.json")
        
        if os.path.exists(registry_file):
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading index registry: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_index_registry(self) -> None:
        """Save index registry to disk."""
        registry_file = os.path.join(self.storage_path, "index_registry.json")
        
        try:
            with open(registry_file, "w", encoding="utf-8") as f:
                json.dump(self.index_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving index registry: {str(e)}")
    
    async def list_remote_indexes(self) -> List[Dict[str, Any]]:
        """
        List indexes from remote Jina API.
        
        Returns:
            List of remote indexes
        """
        span = None
        if self.tracer:
            span = self.tracer.span(name="jina_list_remote_indexes")
        
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
            logger.error(f"Error listing remote indexes: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    def list_local_indexes(self) -> List[Dict[str, Any]]:
        """
        List indexes from local storage.
        
        Returns:
            List of local indexes
        """
        try:
            # Convert registry to list
            indexes = []
            for name, metadata in self.index_registry.items():
                indexes.append({
                    "name": name,
                    **metadata,
                    "storage": "local"
                })
            
            return indexes
            
        except Exception as e:
            logger.error(f"Error listing local indexes: {str(e)}")
            return []
    
    async def sync_remote_indexes(self) -> Dict[str, int]:
        """
        Sync local registry with remote indexes.
        
        Returns:
            Dictionary with sync statistics
        """
        span = None
        if self.tracer:
            span = self.tracer.span(name="jina_sync_remote_indexes")
        
        try:
            # Get remote indexes
            remote_indexes = await self.list_remote_indexes()
            
            added = 0
            updated = 0
            
            # Update local registry with remote information
            for index in remote_indexes:
                name = index.get("name")
                if not name:
                    continue
                
                if name in self.index_registry:
                    # Update existing entry
                    self.index_registry[name].update({
                        "remote_id": index.get("id"),
                        "dimension": index.get("dimension"),
                        "metric": index.get("metric"),
                        "remote_status": index.get("status"),
                        "last_synced": __import__("datetime").datetime.utcnow().isoformat()
                    })
                    updated += 1
                else:
                    # Add new entry
                    self.index_registry[name] = {
                        "name": name,
                        "remote_id": index.get("id"),
                        "dimension": index.get("dimension"),
                        "metric": index.get("metric"),
                        "remote_status": index.get("status"),
                        "description": index.get("description", ""),
                        "created_at": index.get("created_at"),
                        "updated_at": index.get("updated_at"),
                        "storage": "remote",
                        "local_path": None,
                        "last_synced": __import__("datetime").datetime.utcnow().isoformat()
                    }
                    added += 1
            
            # Save updated registry
            self._save_index_registry()
            
            result = {
                "added": added,
                "updated": updated,
                "total": len(self.index_registry)
            }
            
            if span:
                span.update(output=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error syncing remote indexes: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def export_remote_index(
        self,
        index_name: str
    ) -> Dict[str, Any]:
        """
        Export a remote index to local storage.
        
        Args:
            index_name: Name of the index to export
            
        Returns:
            Export status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_export_remote_index",
                input={"index_name": index_name}
            )
        
        try:
            # Check if index exists in registry
            if index_name not in self.index_registry:
                # Try to sync first
                await self.sync_remote_indexes()
                if index_name not in self.index_registry:
                    raise ValueError(f"Index '{index_name}' not found in registry")
            
            index_info = self.index_registry[index_name]
            
            # Check if index has remote ID
            if not index_info.get("remote_id"):
                raise ValueError(f"Index '{index_name}' does not have a remote ID")
            
            # Create local path for index
            local_path = os.path.join(self.indexes_path, index_name)
            os.makedirs(local_path, exist_ok=True)
            
            # Export index data
            # Note: Jina API doesn't currently support direct index export
            # This is a placeholder for future API capabilities
            
            # For now, we'll just update the registry with local path
            index_info["local_path"] = local_path
            index_info["storage"] = "both"
            index_info["last_exported"] = __import__("datetime").datetime.utcnow().isoformat()
            self._save_index_registry()
            
            result = {
                "index_name": index_name,
                "status": "success",
                "local_path": local_path
            }
            
            if span:
                span.update(output=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting remote index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def backup_index(
        self,
        index_name: str
    ) -> Dict[str, Any]:
        """
        Create a backup of an index.
        
        Args:
            index_name: Name of the index to backup
            
        Returns:
            Backup status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_backup_index",
                input={"index_name": index_name}
            )
        
        try:
            # Check if index exists in registry
            if index_name not in self.index_registry:
                raise ValueError(f"Index '{index_name}' not found in registry")
            
            index_info = self.index_registry[index_name]
            
            # Check if index has local path
            if not index_info.get("local_path"):
                # Try to export first
                export_result = await self.export_remote_index(index_name)
                if "error" in export_result:
                    raise ValueError(f"Could not export index: {export_result['error']}")
                
                index_info = self.index_registry[index_name]
            
            # Create backup timestamp
            timestamp = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{index_name}_{timestamp}"
            backup_path = os.path.join(self.backup_path, backup_name)
            
            # Copy index directory to backup
            shutil.copytree(index_info["local_path"], backup_path)
            
            # Update registry with backup info
            if "backups" not in index_info:
                index_info["backups"] = []
            
            index_info["backups"].append({
                "timestamp": timestamp,
                "path": backup_path
            })
            
            index_info["last_backup"] = timestamp
            self._save_index_registry()
            
            result = {
                "index_name": index_name,
                "backup_name": backup_name,
                "backup_path": backup_path,
                "timestamp": timestamp,
                "status": "success"
            }
            
            if span:
                span.update(output=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error backing up index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def restore_index(
        self,
        index_name: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Restore an index from backup.
        
        Args:
            index_name: Name of the index to restore
            timestamp: Optional specific backup timestamp to restore
            
        Returns:
            Restore status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_restore_index",
                input={"index_name": index_name, "timestamp": timestamp}
            )
        
        try:
            # Check if index exists in registry
            if index_name not in self.index_registry:
                raise ValueError(f"Index '{index_name}' not found in registry")
            
            index_info = self.index_registry[index_name]
            
            # Check if index has backups
            if "backups" not in index_info or not index_info["backups"]:
                raise ValueError(f"No backups found for index '{index_name}'")
            
            # Find the backup to restore
            backup_to_restore = None
            
            if timestamp:
                # Find specific backup by timestamp
                for backup in index_info["backups"]:
                    if backup["timestamp"] == timestamp:
                        backup_to_restore = backup
                        break
                
                if not backup_to_restore:
                    raise ValueError(f"Backup with timestamp '{timestamp}' not found")
            else:
                # Use the most recent backup
                backup_to_restore = sorted(
                    index_info["backups"],
                    key=lambda b: b["timestamp"],
                    reverse=True
                )[0]
            
            # Create a backup of current state before restoring
            current_backup = None
            if index_info.get("local_path") and os.path.exists(index_info["local_path"]):
                current_timestamp = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                current_backup_name = f"{index_name}_pre_restore_{current_timestamp}"
                current_backup_path = os.path.join(self.backup_path, current_backup_name)
                
                # Copy current state to backup
                shutil.copytree(index_info["local_path"], current_backup_path)
                current_backup = {
                    "timestamp": current_timestamp,
                    "path": current_backup_path
                }
            
            # Restore from backup
            if os.path.exists(index_info["local_path"]):
                shutil.rmtree(index_info["local_path"])
            
            shutil.copytree(backup_to_restore["path"], index_info["local_path"])
            
            # Update registry
            index_info["last_restored"] = backup_to_restore["timestamp"]
            
            if current_backup:
                index_info["backups"].append(current_backup)
            
            self._save_index_registry()
            
            result = {
                "index_name": index_name,
                "restored_from": backup_to_restore["timestamp"],
                "backup_path": backup_to_restore["path"],
                "current_backup": current_backup["timestamp"] if current_backup else None,
                "status": "success"
            }
            
            if span:
                span.update(output=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error restoring index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    def get_index_stats(
        self,
        index_name: str
    ) -> Dict[str, Any]:
        """
        Get statistics for an index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Index statistics
        """
        try:
            # Check if index exists in registry
            if index_name not in self.index_registry:
                raise ValueError(f"Index '{index_name}' not found in registry")
            
            index_info = self.index_registry[index_name]
            
            # Collect basic stats
            stats = {
                "name": index_name,
                "dimension": index_info.get("dimension"),
                "metric": index_info.get("metric"),
                "storage_type": index_info.get("storage", "unknown"),
                "created_at": index_info.get("created_at"),
                "updated_at": index_info.get("updated_at"),
                "last_synced": index_info.get("last_synced"),
                "last_backup": index_info.get("last_backup"),
                "last_restored": index_info.get("last_restored"),
                "backup_count": len(index_info.get("backups", []))
            }
            
            # Add local storage stats if available
            if index_info.get("local_path") and os.path.exists(index_info["local_path"]):
                local_path = index_info["local_path"]
                
                # Get directory size
                dir_size = 0
                for path, dirs, files in os.walk(local_path):
                    for f in files:
                        fp = os.path.join(path, f)
                        dir_size += os.path.getsize(fp)
                
                # Add file count
                file_count = sum(len(files) for _, _, files in os.walk(local_path))
                
                stats.update({
                    "local_path": local_path,
                    "size_bytes": dir_size,
                    "size_mb": round(dir_size / (1024 * 1024), 2),
                    "file_count": file_count
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}
    
    async def delete_index(
        self,
        index_name: str,
        delete_remote: bool = False,
        keep_backups: bool = True
    ) -> Dict[str, Any]:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
            delete_remote: Whether to delete the remote index
            keep_backups: Whether to keep backups
            
        Returns:
            Deletion status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_delete_index",
                input={"index_name": index_name, "delete_remote": delete_remote}
            )
        
        try:
            # Check if index exists in registry
            if index_name not in self.index_registry:
                raise ValueError(f"Index '{index_name}' not found in registry")
            
            index_info = self.index_registry[index_name]
            result = {"status": "success", "deleted": []}
            
            # Delete local index if it exists
            if index_info.get("local_path") and os.path.exists(index_info["local_path"]):
                shutil.rmtree(index_info["local_path"])
                result["deleted"].append("local")
            
            # Delete remote index if requested
            if delete_remote and index_info.get("remote_id"):
                # Import JinaIndex to delete remote index
                from multiagent.app.tools.jina.jina_index import JinaIndex
                
                index_api = JinaIndex(api_key=self.api_key, tracer=self.tracer)
                delete_result = await index_api.delete_index(index_name)
                
                if "error" not in delete_result:
                    result["deleted"].append("remote")
                else:
                    result["remote_error"] = delete_result["error"]
            
            # Delete backups if requested
            if not keep_backups and "backups" in index_info:
                for backup in index_info["backups"]:
                    backup_path = backup["path"]
                    if os.path.exists(backup_path):
                        shutil.rmtree(backup_path)
                
                result["deleted"].append("backups")
            
            # Remove from registry
            del self.index_registry[index_name]
            self._save_index_registry()
            
            if span:
                span.update(output=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}
    
    async def create_local_index(
        self,
        index_name: str,
        dimension: int = 768,
        metric: str = "cosine",
        description: str = "",
        create_remote: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new local index.
        
        Args:
            index_name: Name for the new index
            dimension: Dimension of embedding vectors
            metric: Distance metric (cosine, euclidean, dot)
            description: Description of the index
            create_remote: Whether to also create a remote index
            
        Returns:
            Index creation status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_create_local_index",
                input={"index_name": index_name, "dimension": dimension}
            )
        
        try:
            # Check if index already exists
            if index_name in self.index_registry:
                raise ValueError(f"Index '{index_name}' already exists")
            
            # Create local path
            local_path = os.path.join(self.indexes_path, index_name)
            os.makedirs(local_path, exist_ok=True)
            
            # Create a simple metadata file to identify the index
            metadata = {
                "name": index_name,
                "dimension": dimension,
                "metric": metric,
                "description": description,
                "created_at": __import__("datetime").datetime.utcnow().isoformat()
            }
            
            with open(os.path.join(local_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            # Add to registry
            self.index_registry[index_name] = {
                "name": index_name,
                "dimension": dimension,
                "metric": metric,
                "description": description,
                "created_at": metadata["created_at"],
                "updated_at": metadata["created_at"],
                "storage": "local",
                "local_path": local_path
            }
            
            # Create remote index if requested
            remote_id = None
            if create_remote:
                # Import JinaIndex to create remote index
                from multiagent.app.tools.jina.jina_index import JinaIndex
                
                index_api = JinaIndex(api_key=self.api_key, tracer=self.tracer)
                create_result = await index_api.create_index(
                    index_name=index_name,
                    dimension=dimension,
                    metric=metric,
                    description=description
                )
                
                if "error" not in create_result:
                    remote_id = create_result.get("id")
                    self.index_registry[index_name]["remote_id"] = remote_id
                    self.index_registry[index_name]["storage"] = "both"
            
            # Save updated registry
            self._save_index_registry()
            
            result = {
                "index_name": index_name,
                "local_path": local_path,
                "dimension": dimension,
                "metric": metric,
                "remote_id": remote_id,
                "status": "created"
            }
            
            if span:
                span.update(output=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating local index: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {"error": str(e)}