
import logging
from typing import Callable

from fastapi import FastAPI

from multiagent.app.db.session import engine
from multiagent.app.db.models import Base
from multiagent.app.monitoring.tracer import setup_tracer


logger = logging.getLogger(__name__)

def create_start_app_handler(app: FastAPI) -> Callable:
    """
    Creates a function to run at application startup.
    
    Args:
        app: FastAPI application
        
    Returns:
        Function to run at startup
    """
    async def start_app() -> None:
        """
        Runs at application startup.
        Initializes database, connections, and services.
        """
        # Create database tables
        logger.info("Creating database tables")
        Base.metadata.create_all(bind=engine)
        
        # Set up tracer
        logger.info("Setting up tracer")
        setup_tracer()
        
        logger.info("Application startup complete")
    
    return start_app

def create_stop_app_handler(app: FastAPI) -> Callable:
    """
    Creates a function to run at application shutdown.
    
    Args:
        app: FastAPI application
        
    Returns:
        Function to run at shutdown
    """
    async def stop_app() -> None:
        """
        Runs at application shutdown.
        Closes connections and performs cleanup.
        """
        logger.info("Application shutdown complete")
    
    return stop_app
