import os
import logging
import asyncio
import uvicorn
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import API components
from multiagent.app.api.query import submit_query, get_query_status, get_query_result, QueryRequest
from multiagent.app.api.websocket import websocket_endpoint
from multiagent.app.api.response import StandardResponse

# Import database components
from multiagent.app.db.session import engine, get_db, init_db
from multiagent.app.db.base import Base
from sqlalchemy.orm import Session

# Import monitoring components
from multiagent.app.monitoring.metrics import PrometheusMiddleware
from multiagent.app.monitoring.logging import setup_logging

# Setup logging
logger = logging.getLogger(__name__)
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Multiagent LLM System",
    description="API for a multiagent LLM system with Jina, Bedrock, and LlamaIndex integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Create database tables
@app.on_event("startup")
async def startup_event():
    try:
        # Initialize database without dropping existing tables
        init_db(drop_existing=False)

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

# Initialize default providers
@app.post("/api/initialize-providers")
async def initialize_default_providers(db: Session = Depends(get_db)):
    """Initialize default providers if they don't exist in the database."""
    # Import models here to avoid circular imports
    from multiagent.app.db.models import ProviderConfig, ProviderCapabilities
    
    try:
        # Check if providers exist
        providers_count = db.query(ProviderConfig).count()
        if providers_count > 0:
            logger.info(f"Found {providers_count} existing providers")
            return {"message": "Providers already initialized"}

        # Create default providers
        providers = [
            # ... your existing providers list ...
        ]

        # Add providers to database
        for provider_data in providers:
            provider = ProviderConfig(
                provider_id=provider_data["provider_id"],
                config=provider_data["config"],
                is_active=True
            )
            db.add(provider)
            db.flush()

            # Add capabilities
            for cap in provider_data["capabilities"]:
                capability = ProviderCapabilities(
                    provider_id=provider.id,
                    capability_type=cap["capability_type"],
                    capability_value=cap["capability_value"],
                    additional_data={}
                )
                db.add(capability)

        db.commit()
        logger.info("Initialized default providers")
        return {"message": "Default providers initialized successfully"}

    except Exception as e:
        db.rollback()
        logger.error(f"Error initializing providers: {str(e)}")
        raise

# Define API routes
# ... (rest of the API routes remain unchanged)

# Run the application
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))

    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )