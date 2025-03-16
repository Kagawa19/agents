"""
Jina Flow builder.
Creates and manages Jina Flow pipelines.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class JinaFlow:
    """
    Creates and manages Jina Flow pipelines.
    Provides methods for building and running Flows for various tasks.
    """
    
    def __init__(
        self,
        api_key: str,
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the Jina Flow builder.
        
        Args:
            api_key: Jina AI API key
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.api_key = api_key
        self.tracer = tracer
        self.base_url = "https://api.jina.ai/v1"
        self.flows = {}
    
    async def create_flow(
        self,
        flow_name: str,
        flow_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new Jina Flow.
        
        Args:
            flow_name: Name of the flow
            flow_config: Flow configuration parameters
            
        Returns:
            Flow information
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_create_flow",
                input={"flow_name": flow_name}
            )
        
        try:
            # Check if Jina Flow package is available
            try:
                from jina import Flow
                from docarray import Document, DocumentArray
            except ImportError:
                logger.error("Jina packages not installed. Using API only mode.")
                # Store flow config for API mode
                self.flows[flow_name] = {
                    "name": flow_name,
                    "config": flow_config,
                    "mode": "api_only"
                }
                
                if span:
                    span.update(output={"status": "api_only"})
                
                return {
                    "flow_name": flow_name,
                    "status": "created",
                    "mode": "api_only"
                }
            
            # Create Jina Flow object
            flow = Flow(**flow_config)
            
            # Store flow for later use
            self.flows[flow_name] = {
                "name": flow_name,
                "config": flow_config,
                "flow": flow,
                "mode": "local"
            }
            
            if span:
                span.update(output={"status": "success"})
            
            return {
                "flow_name": flow_name,
                "status": "created",
                "mode": "local"
            }
            
        except Exception as e:
            logger.error(f"Error creating Jina Flow: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {
                "flow_name": flow_name,
                "status": "error",
                "error": str(e)
            }
    
    async def create_indexing_flow(
        self,
        flow_name: str = "indexing_flow",
        dim: int = 768,
        replicas: int = 1
    ) -> Dict[str, Any]:
        """
        Create a flow for document indexing.
        
        Args:
            flow_name: Name of the flow
            dim: Dimension of embeddings
            replicas: Number of replicas for each executor
            
        Returns:
            Flow information
        """
        # Configure indexing flow
        flow_config = {
            "port": 51000 + hash(flow_name) % 1000,  # Dynamic port to avoid conflicts
            "cors": True,
            "protocol": "http",
            "no_debug_endpoints": True
        }
        
        try:
            from jina import Flow
            
            # Create flow
            result = await self.create_flow(flow_name, flow_config)
            
            if result["status"] == "created":
                # Add executors if using local mode
                if result["mode"] == "local":
                    flow = self.flows[flow_name]["flow"]
                    
                    # Add preprocessing
                    flow = flow.add(
                        name="preprocessor",
                        uses="jinahub://Preprocessor",
                        uses_with={"traversal_paths": "@r"},
                        replicas=replicas
                    )
                    
                    # Add encoder
                    flow = flow.add(
                        name="encoder",
                        uses="jinahub://CLIPImageEncoder",
                        uses_with={"traversal_paths": "@r"},
                        replicas=replicas
                    )
                    
                    # Add indexer
                    flow = flow.add(
                        name="indexer",
                        uses="jinahub://SimpleIndexer",
                        uses_with={
                            "traversal_paths": "@r",
                            "dim": dim,
                            "index_file_name": f"{flow_name}.bin"
                        },
                        volumes=f"./indexes/{flow_name}:/workspace/indexes"
                    )
                    
                    # Update flow in storage
                    self.flows[flow_name]["flow"] = flow
            
            return result
            
        except ImportError:
            logger.warning("Jina packages not installed, using API mode")
            return await self.create_flow(flow_name, flow_config)
            
        except Exception as e:
            logger.error(f"Error creating indexing flow: {str(e)}")
            return {
                "flow_name": flow_name,
                "status": "error",
                "error": str(e)
            }
    
    async def create_search_flow(
        self,
        flow_name: str = "search_flow",
        dim: int = 768,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Create a flow for document search.
        
        Args:
            flow_name: Name of the flow
            dim: Dimension of embeddings
            top_k: Number of results to return
            
        Returns:
            Flow information
        """
        # Configure search flow
        flow_config = {
            "port": 52000 + hash(flow_name) % 1000,  # Dynamic port to avoid conflicts
            "cors": True,
            "protocol": "http",
            "no_debug_endpoints": True
        }
        
        try:
            from jina import Flow
            
            # Create flow
            result = await self.create_flow(flow_name, flow_config)
            
            if result["status"] == "created":
                # Add executors if using local mode
                if result["mode"] == "local":
                    flow = self.flows[flow_name]["flow"]
                    
                    # Add encoder
                    flow = flow.add(
                        name="encoder",
                        uses="jinahub://CLIPImageEncoder",
                        uses_with={"traversal_paths": "@r"},
                    )
                    
                    # Add ranker
                    flow = flow.add(
                        name="ranker",
                        uses="jinahub://SimpleRanker",
                        uses_with={
                            "traversal_paths": "@r",
                            "metric": "cosine",
                            "top_k": top_k
                        }
                    )
                    
                    # Update flow in storage
                    self.flows[flow_name]["flow"] = flow
            
            return result
            
        except ImportError:
            logger.warning("Jina packages not installed, using API mode")
            return await self.create_flow(flow_name, flow_config)
            
        except Exception as e:
            logger.error(f"Error creating search flow: {str(e)}")
            return {
                "flow_name": flow_name,
                "status": "error",
                "error": str(e)
            }
    
    async def start_flow(self, flow_name: str) -> Dict[str, Any]:
        """
        Start a Jina Flow.
        
        Args:
            flow_name: Name of the flow to start
            
        Returns:
            Flow status
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_start_flow",
                input={"flow_name": flow_name}
            )
        
        try:
            # Check if flow exists
            if flow_name not in self.flows:
                raise ValueError(f"Flow '{flow_name}' not found")
            
            # Get flow information
            flow_info = self.flows[flow_name]
            
            # Skip if using API mode
            if flow_info["mode"] == "api_only":
                logger.info(f"Flow '{flow_name}' is in API mode, no need to start")
                
                if span:
                    span.update(output={"status": "api_only"})
                
                return {
                    "flow_name": flow_name,
                    "status": "api_only"
                }
            
            # Start the flow
            flow_info["flow"].start()
            
            # Update flow status
            flow_info["status"] = "running"
            
            if span:
                span.update(output={"status": "running"})
            
            return {
                "flow_name": flow_name,
                "status": "running",
                "endpoint": f"http://localhost:{flow_info['flow'].port}"
            }
            
        except Exception as e:
            logger.error(f"Error starting flow '{flow_name}': {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return {
                "flow_name": flow_name,
                "status": "error",
                "error": str(e)
            }
    
    async def stop_flow(self, flow_name: str) -> Dict[str, Any]:
        """
        Stop a Jina Flow.
        
        Args:
            flow_name: Name of the flow to stop
            
        Returns:
            Flow status
        """
        try:
            # Check if flow exists
            if flow_name not in self.flows:
                raise ValueError(f"Flow '{flow_name}' not found")
            
            # Get flow information
            flow_info = self.flows[flow_name]
            
            # Skip if using API mode
            if flow_info["mode"] == "api_only":
                logger.info(f"Flow '{flow_name}' is in API mode, no need to stop")
                return {
                    "flow_name": flow_name,
                    "status": "api_only"
                }
            
            # Stop the flow
            flow_info["flow"].close()
            
            # Update flow status
            flow_info["status"] = "stopped"
            
            return {
                "flow_name": flow_name,
                "status": "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error stopping flow '{flow_name}': {str(e)}")
            return {
                "flow_name": flow_name,
                "status": "error",
                "error": str(e)
            }
    
    async def get_flow_status(self, flow_name: str) -> Dict[str, Any]:
        """
        Get status of a Jina Flow.
        
        Args:
            flow_name: Name of the flow
            
        Returns:
            Flow status
        """
        try:
            # Check if flow exists
            if flow_name not in self.flows:
                return {
                    "flow_name": flow_name,
                    "status": "not_found"
                }
            
            # Get flow information
            flow_info = self.flows[flow_name]
            
            # Build status response
            status = {
                "flow_name": flow_name,
                "mode": flow_info["mode"],
                "config": flow_info["config"]
            }
            
            # Add additional information for local flows
            if flow_info["mode"] == "local":
                status["status"] = flow_info.get("status", "created")
                if status["status"] == "running":
                    status["endpoint"] = f"http://localhost:{flow_info['flow'].port}"
            else:
                status["status"] = "api_only"
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting flow status: {str(e)}")
            return {
                "flow_name": flow_name,
                "status": "error",
                "error": str(e)
            }
    
    async def list_flows(self) -> List[Dict[str, Any]]:
        """
        List all Jina Flows.
        
        Returns:
            List of flow information
        """
        try:
            # Build list of flow information
            flow_list = []
            for name, info in self.flows.items():
                status = await self.get_flow_status(name)
                flow_list.append(status)
            
            return flow_list
            
        except Exception as e:
            logger.error(f"Error listing flows: {str(e)}")
            return []