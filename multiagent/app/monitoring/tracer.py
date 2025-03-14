"""
Langfuse tracer implementation.
Provides tracing and monitoring capabilities for agent operations.
"""

import logging
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request, Response
from langfuse import Langfuse
from langfuse.decorators import observe
from starlette.middleware.base import BaseHTTPMiddleware

from multiagent.app.core.config import settings


logger = logging.getLogger(__name__)

# Global tracer instance
_tracer = None


def get_tracer() -> "LangfuseTracer":
    """
    Get the global tracer instance.
    
    Returns:
        Global LangfuseTracer instance
    """
    global _tracer
    if _tracer is None:
        _tracer = setup_tracer()
    return _tracer


def setup_tracer() -> "LangfuseTracer":
    """
    Set up the global tracer instance.
    
    Returns:
        Configured LangfuseTracer instance
    """
    global _tracer
    _tracer = LangfuseTracer(
        secret_key=settings.LANGFUSE_SECRET_KEY,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        host=settings.LANGFUSE_HOST
    )
    return _tracer


class LangfuseTracer:
    """
    Tracer for monitoring agent operations using Langfuse.
    Provides functions for creating traces, spans, and logging events.
    """
    
    def __init__(self, secret_key: str, public_key: str, host: str):
        """
        Initialize the Langfuse tracer.
        
        Args:
            secret_key: Langfuse secret key
            public_key: Langfuse public key
            host: Langfuse API host
        """
        try:
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            self.enabled = True
            logger.info("Langfuse tracer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse tracer: {e}")
            self.enabled = False
    
    def create_trace(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Create a new trace for tracking a sequence of operations.
        
        Args:
            name: Name of the trace
            kwargs: Additional trace parameters
            
        Returns:
            Trace information
        """
        if not self.enabled:
            return {"id": f"mock-trace-{name}", "name": name}
        
        try:
            trace = self.langfuse.trace(name=name, **kwargs)
            return {"id": trace.id, "name": name}
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return {"id": f"error-trace-{name}", "name": name}
    
    def trace(self, name: str):
        """
        Context manager for creating a trace.
        
        Args:
            name: Name of the trace
            
        Returns:
            Trace context manager
        """
        if not self.enabled:
            return self
        
        try:
            return self.langfuse.trace(name=name)
        except Exception as e:
            logger.error(f"Failed to create trace context: {e}")
            return self
    
    def span(self, name: str = None, **kwargs):
        """
        Context manager for creating a span within a trace.
        
        Args:
            name: Name of the span
            kwargs: Additional span parameters
            
        Returns:
            Span context manager
        """
        if not self.enabled:
            return self
        
        try:
            return self.langfuse.span(name=name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create span context: {e}")
            return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    @observe()
    def trace_requests(self, request: Request, call_next) -> Response:
        """
        Middleware for tracing HTTP requests.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            Response from the next middleware
        """
        if not self.enabled:
            return call_next(request)
        
        response = call_next(request)
        
        # Log additional metadata about the request
        self.langfuse.generation(
            name="http_request",
            input=f"{request.method} {request.url.path}",
            output=str(response.status_code),
            metadata={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client": request.client.host if request.client else "unknown",
                "status_code": response.status_code
            }
        )
        
        return response
    
    def log_generation(
        self, 
        model: str, 
        prompt: str, 
        response: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an LLM generation event.
        
        Args:
            model: Name of the model used
            prompt: Prompt sent to the model
            response: Response from the model
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            # Truncate prompt and response if they're too long
            max_length = 5000  # Arbitrary limit to prevent issues with very large prompts
            prompt = (prompt[:max_length] + "... [truncated]") if len(prompt) > max_length else prompt
            response = (response[:max_length] + "... [truncated]") if len(response) > max_length else response
            
            self.langfuse.generation(
                name=model,
                input=prompt,
                output=response,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to log generation: {e}")
    
    def score(self, name: str, value: float, comment: Optional[str] = None) -> None:
        """
        Record a score for the current trace.
        
        Args:
            name: Name of the score
            value: Numeric value of the score
            comment: Optional comment explaining the score
        """
        if not self.enabled:
            return
        
        try:
            self.langfuse.score(
                name=name,
                value=value,
                comment=comment
            )
        except Exception as e:
            logger.error(f"Failed to record score: {e}")


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding tracing to FastAPI requests.
    """
    
    def __init__(self, app: FastAPI):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.tracer = get_tracer()
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process a request and add tracing.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            Response from the next middleware
        """
        return self.tracer.trace_requests(request, call_next)


def setup_tracing(app: FastAPI) -> None:
    """
    Set up tracing for a FastAPI application.
    
    Args:
        app: FastAPI application
    """
    # Initialize tracer
    tracer = get_tracer()
    
    # Add middleware if tracing is enabled
    if tracer.enabled:
        app.add_middleware(TracingMiddleware)
        logger.info("Tracing middleware added to FastAPI application")