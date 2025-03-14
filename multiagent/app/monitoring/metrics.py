"""
Prometheus metrics implementation.
Provides metrics collection for monitoring system performance.
"""

import logging
import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client.openmetrics.exposition import generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


logger = logging.getLogger(__name__)

# Define metrics
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)

AGENT_EXECUTIONS = Counter(
    "agent_executions_total",
    "Total number of agent executions",
    ["agent_id", "status"]
)

AGENT_EXECUTION_DURATION = Histogram(
    "agent_execution_duration_seconds",
    "Agent execution duration in seconds",
    ["agent_id"]
)

WORKFLOW_EXECUTIONS = Counter(
    "workflow_executions_total",
    "Total number of workflow executions",
    ["workflow_id", "status"]
)

WORKFLOW_EXECUTION_DURATION = Histogram(
    "workflow_execution_duration_seconds",
    "Workflow execution duration in seconds",
    ["workflow_id"]
)

LLM_API_CALLS = Counter(
    "llm_api_calls_total",
    "Total number of LLM API calls",
    ["provider", "model"]
)

LLM_TOKEN_USAGE = Counter(
    "llm_token_usage_total",
    "Total number of tokens used",
    ["provider", "model", "type"]
)

LLM_API_LATENCY = Histogram(
    "llm_api_latency_seconds",
    "LLM API latency in seconds",
    ["provider", "model"]
)

ACTIVE_TASKS = Gauge(
    "active_tasks",
    "Number of active tasks",
    ["type"]
)

VECTOR_DB_OPERATIONS = Counter(
    "vector_db_operations_total",
    "Total number of vector database operations",
    ["operation", "status"]
)

VECTOR_DB_LATENCY = Histogram(
    "vector_db_latency_seconds",
    "Vector database operation latency in seconds",
    ["operation"]
)

TASK_QUEUE_SIZE = Gauge(
    "task_queue_size",
    "Size of the task queue",
    ["queue"]
)

API_ERRORS = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["endpoint", "error_type"]
)

SYSTEM_INFO = Gauge(
    "system_info",
    "System information",
    ["name", "value"]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting Prometheus metrics.
    """
    
    def __init__(self, app: FastAPI):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        logger.info("Prometheus middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process a request and collect metrics.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            Response from the next middleware
        """
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Get method and endpoint for metrics
        method = request.method
        endpoint = request.url.path
        
        # Time the request
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Get status code
            status_code = response.status_code
            
            # Record metrics
            HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=status_code).inc()
            HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)
            
            return response
        except Exception as e:
            # Record error metrics
            HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=500).inc()
            HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)
            API_ERRORS.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
            
            # Re-raise the exception
            raise


def metrics_endpoint() -> StarletteResponse:
    """
    Endpoint for exposing Prometheus metrics.
    
    Returns:
        Response with metrics in Prometheus format
    """
    try:
        # Update active task counts
        try:
            from app.worker.queue import TaskQueue
            queue = TaskQueue()
            active_tasks = queue.get_active_tasks()
            
            # Reset counts
            ACTIVE_TASKS.labels(type="total").set(len(active_tasks))
            # Count by type
            task_types = {}
            for task in active_tasks:
                task_type = task.get("name", "unknown").split(".")[-1]
                task_types[task_type] = task_types.get(task_type, 0) + 1
            
            for task_type, count in task_types.items():
                ACTIVE_TASKS.labels(type=task_type).set(count)
                
            # Update queue sizes
            queue_sizes = queue.get_queue_sizes()
            for queue_name, size in queue_sizes.items():
                TASK_QUEUE_SIZE.labels(queue=queue_name).set(size)
        except Exception as e:
            logger.error(f"Failed to update active task metrics: {e}")
        
        # Generate latest metrics
        metrics_data = generate_latest()
        
        return StarletteResponse(
            content=metrics_data,
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return StarletteResponse(
            content=f"Error generating metrics: {e}",
            status_code=500
        )


def track_agent_execution(agent_id: str, status: str, duration: float) -> None:
    """
    Track an agent execution for metrics.
    
    Args:
        agent_id: ID of the agent
        status: Execution status (success, failure)
        duration: Execution duration in seconds
    """
    try:
        AGENT_EXECUTIONS.labels(agent_id=agent_id, status=status).inc()
        AGENT_EXECUTION_DURATION.labels(agent_id=agent_id).observe(duration)
    except Exception as e:
        logger.error(f"Failed to track agent execution metrics: {e}")


def track_workflow_execution(workflow_id: str, status: str, duration: float) -> None:
    """
    Track a workflow execution for metrics.
    
    Args:
        workflow_id: ID of the workflow
        status: Execution status (success, failure)
        duration: Execution duration in seconds
    """
    try:
        WORKFLOW_EXECUTIONS.labels(workflow_id=workflow_id, status=status).inc()
        WORKFLOW_EXECUTION_DURATION.labels(workflow_id=workflow_id).observe(duration)
    except Exception as e:
        logger.error(f"Failed to track workflow execution metrics: {e}")


def track_llm_call(provider: str, model: str, tokens: int, call_type: str, latency: float) -> None:
    """
    Track an LLM API call for metrics.
    
    Args:
        provider: LLM provider (openai, bedrock)
        model: Model name
        tokens: Number of tokens used
        call_type: Type of call (prompt, completion)
        latency: API latency in seconds
    """
    try:
        LLM_API_CALLS.labels(provider=provider, model=model).inc()
        LLM_TOKEN_USAGE.labels(provider=provider, model=model, type=call_type).inc(tokens)
        LLM_API_LATENCY.labels(provider=provider, model=model).observe(latency)
    except Exception as e:
        logger.error(f"Failed to track LLM call metrics: {e}")


def track_vector_db_operation(operation: str, status: str, latency: float) -> None:
    """
    Track a vector database operation for metrics.
    
    Args:
        operation: Operation type (upsert, query, delete)
        status: Operation status (success, failure)
        latency: Operation latency in seconds
    """
    try:
        VECTOR_DB_OPERATIONS.labels(operation=operation, status=status).inc()
        VECTOR_DB_LATENCY.labels(operation=operation).observe(latency)
    except Exception as e:
        logger.error(f"Failed to track vector DB operation metrics: {e}")


def setup_metrics(app: FastAPI) -> None:
    """
    Set up metrics collection for a FastAPI application.
    
    Args:
        app: FastAPI application
    """
    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)
    
    # Add metrics endpoint
    app.add_route("/metrics", metrics_endpoint)
    
    # Set basic system info
    from multiagent.app import __version__
    SYSTEM_INFO.labels(name="version", value=__version__).set(1)
    
    logger.info("Metrics setup complete")