"""
Langfuse tracer implementation.
Provides tracing and monitoring capabilities for agent operations.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

from fastapi import FastAPI, Request, Response
from langfuse.client import Langfuse
from langfuse.model import CreateTrace, CreateSpan, CreateGeneration, CreateScore
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings


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
    _tracer = LangfuseTracer(settings)
    return _tracer


class LangfuseTracer:
    """
    Tracer for monitoring agent operations using Langfuse.
    Provides functions for creating traces, spans, and logging events.
    """
    
    def __init__(self, settings):
        """
        Initialize the Langfuse tracer.
        
        Args:
            settings: Application settings containing Langfuse keys
        """
        try:
            self.langfuse = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST
            )
            self.current_trace = None
            self.current_span = None
            self.enabled = True
            logger.info("Langfuse tracer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse tracer: {e}")
            self.enabled = False
    
    def trace_requests(self, request: Request, call_next: Callable) -> Response:
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
        
        trace_id = self.create_trace(
            name=f"{request.method} {request.url.path}",
            metadata={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        start_time = time.time()
        
        try:
            response = call_next(request)
            
            # Record the response status
            self.log_event(
                event_type="http_response",
                event_data={
                    "status_code": response.status_code,
                    "duration_ms": (time.time() - start_time) * 1000
                }
            )
            
            return response
        except Exception as e:
            # Record the exception
            self.log_event(
                event_type="http_exception",
                event_data={
                    "error": str(e),
                    "duration_ms": (time.time() - start_time) * 1000
                }
            )
            raise
        finally:
            self.end_trace()
    
    def create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create a new trace.
        
        Args:
            name: Name of the trace
            metadata: Additional metadata for the trace
            
        Returns:
            ID of the created trace or None if failed
        """
        if not self.enabled:
            return None
        
        try:
            trace = self.langfuse.trace(
                CreateTrace(
                    name=name,
                    metadata=metadata
                )
            )
            self.current_trace = trace
            return trace.id
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None
    
    def end_trace(self) -> None:
        """End the current trace."""
        if not self.enabled or self.current_trace is None:
            return
        
        try:
            self.current_trace = None
        except Exception as e:
            logger.error(f"Failed to end trace: {e}")
    
    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
        """
        Context manager for creating a span within a trace.
        
        Args:
            name: Name of the span
            metadata: Additional metadata for the span
            
        Yields:
            None
        """
        if not self.enabled or self.current_trace is None:
            yield
            return
        
        parent_span = self.current_span
        
        try:
            span = self.current_trace.span(
                CreateSpan(
                    name=name,
                    metadata=metadata
                )
            )
            self.current_span = span
            
            yield
        except Exception as e:
            logger.error(f"Error in span {name}: {e}")
            # Still need to yield to ensure the context manager completes
            yield
        finally:
            if self.current_span:
                try:
                    self.current_span.end()
                except Exception as e:
                    logger.error(f"Failed to end span: {e}")
                
                self.current_span = parent_span
    
    @contextmanager
    def trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
        """
        Context manager for creating a trace.
        
        Args:
            name: Name of the trace
            metadata: Additional metadata for the trace
            
        Yields:
            None
        """
        if not self.enabled:
            yield
            return
        
        old_trace = self.current_trace
        
        try:
            trace_id = self.create_trace(name, metadata)
            
            yield
        except Exception as e:
            logger.error(f"Error in trace {name}: {e}")
            # Still need to yield to ensure the context manager completes
            yield
        finally:
            self.end_trace()
            self.current_trace = old_trace
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Log an event to the current trace or span.
        
        Args:
            event_type: Type of the event
            event_data: Data associated with the event
        """
        if not self.enabled:
            return
        
        try:
            if self.current_span:
                self.current_span.update(
                    metadata={
                        f"event_{event_type}": event_data
                    }
                )
            elif self.current_trace:
                self.current_trace.update(
                    metadata={
                        f"event_{event_type}": event_data
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
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
            if len(prompt) > max_length:
                prompt = prompt[:max_length] + "... [truncated]"
            
            if len(response) > max_length:
                response = response[:max_length] + "... [truncated]"
            
            generation_data = CreateGeneration(
                model=model,
                prompt=prompt,
                response=response,
                metadata=metadata
            )
            
            if self.current_span:
                self.current_span.generation(generation_data)
            elif self.current_trace:
                self.current_trace.generation(generation_data)
        except Exception as e:
            logger.error(f"Failed to log generation: {e}")
    
    def score(self, name: str, value: float, comment: Optional[str] = None) -> None:
        """
        Record a score for the current trace or span.
        
        Args:
            name: Name of the score
            value: Numeric value of the score
            comment: Optional comment explaining the score
        """
        if not self.enabled:
            return
        
        try:
            score_data = CreateScore(
                name=name,
                value=value,
                comment=comment
            )
            
            if self.current_span:
                self.current_span.score(score_data)
            elif self.current_trace:
                self.current_trace.score(score_data)
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
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
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