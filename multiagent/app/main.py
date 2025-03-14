
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from multiagent.app.api.endpoints import agents, auth, health, query
from multiagent.app.api.error_handlers import add_exception_handlers
from app.core.config import settings
from app.core.events import create_start_app_handler, create_stop_app_handler
from app.monitoring.logging import setup_logging
from app.monitoring.metrics import PrometheusMiddleware, setup_metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Handles application startup and shutdown events.
    """
    # Startup
    startup_handler = create_start_app_handler(app)
    await startup_handler()
    
    # Yield control to the application
    yield
    
    # Shutdown
    shutdown_handler = create_stop_app_handler(app)
    await shutdown_handler()


def create_application() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    Sets up middleware, routes, and exception handlers.
    
    Returns:
        FastAPI: The configured application instance
    """
    # Initialize logging
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.PROJECT_VERSION,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup monitoring
    app.add_middleware(PrometheusMiddleware)
    setup_metrics(app)
    
    # Add routes
    app.include_router(auth.router, prefix="/api", tags=["auth"])
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(query.router, prefix="/api", tags=["query"])
    app.include_router(agents.router, prefix="/api", tags=["agents"])
    
    # Add exception handlers
    add_exception_handlers(app)
    
    return app


def get_application() -> FastAPI:
    """
    Returns the configured FastAPI application instance.
    
    Returns:
        FastAPI: The application instance
    """
    return create_application()


app = get_application()
