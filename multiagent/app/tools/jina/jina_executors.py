"""
Hub executor factory for Jina.
Provides methods for creating and configuring Jina executors.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class JinaExecutors:
    """
    Factory for creating and configuring Jina executors.
    Provides methods for working with executors from the Jina Hub.
    """
    
    def __init__(
        self,
        api_key: str,
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the Jina Executors factory.
        
        Args:
            api_key: Jina AI API key
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.api_key = api_key
        self.tracer = tracer
        
        # Registry of available executors
        self.executors = {
            # Preprocessors
            "text_preprocessor": {
                "hub": "jinahub://Preprocessor",
                "params": {"traversal_paths": "@r"}
            },
            "image_preprocessor": {
                "hub": "jinahub://ImagePreprocessor",
                "params": {"target_size": 224}
            },
            
            # Encoders
            "clip_text_encoder": {
                "hub": "jinahub://CLIPTextEncoder",
                "params": {"traversal_paths": "@r"}
            },
            "clip_image_encoder": {
                "hub": "jinahub://CLIPImageEncoder",
                "params": {"traversal_paths": "@r"}
            },
            "sentence_encoder": {
                "hub": "jinahub://TransformerTorchEncoder",
                "params": {
                    "pretrained_model_name_or_path": "sentence-transformers/all-mpnet-base-v2",
                    "traversal_paths": "@r"
                }
            },
            
            # Indexers
            "vector_indexer": {
                "hub": "jinahub://SimpleIndexer",
                "params": {
                    "traversal_paths": "@r",
                    "match_args": {"metric": "cosine", "limit": 10}
                }
            },
            
            # Rankers
            "simple_ranker": {
                "hub": "jinahub://SimpleRanker",
                "params": {
                    "metric": "cosine",
                    "traversal_paths": "@r",
                    "top_k": 10
                }
            },
            
            # Evaluators
            "text_evaluator": {
                "hub": "jinahub://TextEvaluator",
                "params": {"traversal_paths": "@r"}
            }
        }
    
    def get_executor_config(
        self,
        executor_type: str,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get configuration for a specific executor type.
        
        Args:
            executor_type: Type of executor to configure
            custom_params: Optional custom parameters to override defaults
            
        Returns:
            Executor configuration
        """
        try:
            # Check if executor type exists
            if executor_type not in self.executors:
                raise ValueError(f"Executor type '{executor_type}' not found")
            
            # Get base configuration
            config = self.executors[executor_type].copy()
            
            # Override with custom parameters if provided
            if custom_params:
                config["params"] = {**config["params"], **custom_params}
            
            return config
            
        except Exception as e:
            logger.error(f"Error getting executor config: {str(e)}")
            return {}
    
    def create_flow_config(
        self,
        executors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a flow configuration from a list of executors.
        
        Args:
            executors: List of executor configurations
            
        Returns:
            Flow configuration
        """
        try:
            # Base flow configuration
            flow_config = {
                "executors": [],
                "port": 51000,  # Default port
                "cors": True,
                "protocol": "http",
                "no_debug_endpoints": True
            }
            
            # Add executors to flow
            for i, executor in enumerate(executors):
                executor_name = executor.get("name", f"exec_{i}")
                executor_type = executor.get("type")
                
                if not executor_type:
                    logger.warning(f"Executor {executor_name} missing type, skipping")
                    continue
                
                # Get executor configuration
                exec_config = self.get_executor_config(
                    executor_type,
                    executor.get("params")
                )
                
                if not exec_config:
                    logger.warning(f"Failed to get config for executor {executor_name}, skipping")
                    continue
                
                # Add to flow configuration
                flow_config["executors"].append({
                    "name": executor_name,
                    "uses": exec_config["hub"],
                    "uses_with": exec_config["params"],
                    "replicas": executor.get("replicas", 1)
                })
            
            return flow_config
            
        except Exception as e:
            logger.error(f"Error creating flow config: {str(e)}")
            return {"executors": []}
    
    def create_indexing_pipeline(
        self,
        embedding_dim: int = 768,
        index_name: str = "default_index",
        replicas: int = 1
    ) -> Dict[str, Any]:
        """
        Create a standard indexing pipeline.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_name: Name for the index
            replicas: Number of replicas for each executor
            
        Returns:
            Flow configuration for indexing pipeline
        """
        try:
            # Define pipeline executors
            executors = [
                {
                    "name": "preprocessor",
                    "type": "text_preprocessor",
                    "replicas": replicas
                },
                {
                    "name": "encoder",
                    "type": "sentence_encoder",
                    "replicas": replicas
                },
                {
                    "name": "indexer",
                    "type": "vector_indexer",
                    "params": {
                        "dim": embedding_dim,
                        "index_file_name": f"{index_name}.bin"
                    },
                    "replicas": 1  # Indexer typically uses single replica
                }
            ]
            
            # Create flow configuration
            flow_config = self.create_flow_config(executors)
            
            # Add volumes for index persistence
            flow_config["volumes"] = f"./indexes/{index_name}:/workdir"
            
            return flow_config
            
        except Exception as e:
            logger.error(f"Error creating indexing pipeline: {str(e)}")
            return {"executors": []}
    
    def create_search_pipeline(
        self,
        embedding_dim: int = 768,
        top_k: int = 10,
        metric: str = "cosine",
        replicas: int = 1
    ) -> Dict[str, Any]:
        """
        Create a standard search pipeline.
        
        Args:
            embedding_dim: Dimension of embeddings
            top_k: Number of results to return
            metric: Distance metric (cosine, euclidean, dot)
            replicas: Number of replicas for each executor
            
        Returns:
            Flow configuration for search pipeline
        """
        try:
            # Define pipeline executors
            executors = [
                {
                    "name": "preprocessor",
                    "type": "text_preprocessor",
                    "replicas": replicas
                },
                {
                    "name": "encoder",
                    "type": "sentence_encoder",
                    "replicas": replicas
                },
                {
                    "name": "ranker",
                    "type": "simple_ranker",
                    "params": {
                        "metric": metric,
                        "top_k": top_k
                    },
                    "replicas": replicas
                }
            ]
            
            # Create flow configuration
            flow_config = self.create_flow_config(executors)
            
            return flow_config
            
        except Exception as e:
            logger.error(f"Error creating search pipeline: {str(e)}")
            return {"executors": []}
    
    def create_multimodal_pipeline(
        self,
        embedding_dim: int = 768,
        top_k: int = 10,
        replicas: int = 1
    ) -> Dict[str, Any]:
        """
        Create a multimodal pipeline for text and image processing.
        
        Args:
            embedding_dim: Dimension of embeddings
            top_k: Number of results to return
            replicas: Number of replicas for each executor
            
        Returns:
            Flow configuration for multimodal pipeline
        """
        try:
            # Define pipeline executors
            executors = [
                # Text branch
                {
                    "name": "text_preprocessor",
                    "type": "text_preprocessor",
                    "replicas": replicas
                },
                {
                    "name": "text_encoder",
                    "type": "clip_text_encoder",
                    "replicas": replicas
                },
                
                # Image branch
                {
                    "name": "image_preprocessor",
                    "type": "image_preprocessor",
                    "replicas": replicas
                },
                {
                    "name": "image_encoder",
                    "type": "clip_image_encoder",
                    "replicas": replicas
                },
                
                # Common ranker
                {
                    "name": "ranker",
                    "type": "simple_ranker",
                    "params": {
                        "top_k": top_k
                    },
                    "replicas": replicas
                }
            ]
            
            # Create flow configuration
            flow_config = self.create_flow_config(executors)
            
            # Add specialized routing
            flow_config["env"] = {
                "JINA_MIME_TYPE": "true"
            }
            
            return flow_config
            
        except Exception as e:
            logger.error(f"Error creating multimodal pipeline: {str(e)}")
            return {"executors": []}