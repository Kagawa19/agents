# main.py
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
        # Initialize database
        # Use drop_existing=True during development to reset database
        init_db(drop_existing=True)

        logger.info("Database initialized successfully")

        # Initialize default providers if not exists
        await initialize_default_providers()

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

async def initialize_default_providers():
    """Initialize default providers if they don't exist in the database."""
    # Import models here to avoid circular imports
    from multiagent.app.db.models import ProviderConfig, ProviderCapabilities
    
    db = next(get_db())
    try:
        # Check if providers exist
        providers_count = db.query(ProviderConfig).count()
        if providers_count > 0:
            logger.info(f"Found {providers_count} existing providers")
            return

        # Create default providers
        providers = [
            {
                "provider_id": "bedrock",
                "config": {
                    "default_model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "models": [
                        "anthropic.claude-3-sonnet-20240229-v1:0",
                        "anthropic.claude-3-haiku-20240307-v1:0"
                    ]
                },
                "capabilities": [
                    {"capability_type": "text_generation", "capability_value": 0.9},
                    {"capability_type": "summarization", "capability_value": 0.85},
                    {"capability_type": "reasoning", "capability_value": 0.95}
                ]
            },
            {
                "provider_id": "jina",
                "config": {
                    "host": "localhost", 
                    "port": 8080,
                    "workspace_dir": "./workspace"
                },
                "capabilities": [
                    {"capability_type": "vector_search", "capability_value": 0.95},
                    {"capability_type": "document_indexing", "capability_value": 0.9}
                ]
            },
            {
                "provider_id": "serper",
                "config": {
                    "api_key": os.getenv("SERPER_API_KEY", ""),
                    "engine": "google"
                },
                "capabilities": [
                    {"capability_type": "web_search", "capability_value": 0.9}
                ]
            }
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
                    additional_data={}  # Changed from metadata to additional_data
                )
                db.add(capability)

        db.commit()
        logger.info("Initialized default providers")

    except Exception as e:
        db.rollback()
        logger.error(f"Error initializing providers: {str(e)}")
    finally:
        db.close()

# Define API routes
@app.post("/api/query", response_model=StandardResponse)
async def api_submit_query(query_data: QueryRequest, background_tasks: BackgroundTasks):
    """
    Submit a query for processing.

    Args:
        query_data: Query request data
        background_tasks: FastAPI background tasks

    Returns:
        StandardResponse with task ID and status
    """
    try:
        result = await submit_query(query_data.dict()) 
        return StandardResponse(
            status="success",
            data=result
        )
    except Exception as e:
        logger.error(f"Error submitting query: {str(e)}")
        return StandardResponse(
            status="error",
            error=str(e)
        )

@app.get("/api/query/{task_id}/status", response_model=StandardResponse)
async def api_get_query_status(task_id: str):
    """
    Get the status of a query.

    Args:
        task_id: The task ID

    Returns:
        StandardResponse with query status
    """
    try:
        result = await get_query_status(task_id)
        return StandardResponse(
            status="success",
            data=result
        )
    except Exception as e:
        logger.error(f"Error getting query status: {str(e)}")
        return StandardResponse(
            status="error",
            error=str(e)
        )

@app.get("/api/query/{task_id}/result", response_model=StandardResponse)
async def api_get_query_result(task_id: str):
    """
    Get the result of a query.

    Args:
        task_id: The task ID

    Returns:
        StandardResponse with query result
    """
    try:
        result = await get_query_result(task_id)
        return StandardResponse(
            status="success",
            data=result
        )
    except Exception as e:
        logger.error(f"Error getting query result: {str(e)}")
        return StandardResponse(
            status="error",
            error=str(e)
        )

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint_route(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Args:
        websocket: WebSocket connection
    """
    await websocket_endpoint(websocket)

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """
    Get Prometheus metrics.

    Returns:
        Metrics in Prometheus format
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Health check endpoint
@app.get("/health")  
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy", "version": "1.0.0"}

# Provider management endpoints
@app.get("/api/providers", response_model=StandardResponse)
async def get_providers(db: Session = Depends(get_db)):
    """
    Get all providers.

    Args:
        db: Database session

    Returns:
        StandardResponse with providers
    """
    # Import models here to avoid circular imports
    from multiagent.app.db.models import ProviderConfig, ProviderCapabilities
    
    providers = db.query(ProviderConfig).all()

    provider_list = []
    for provider in providers:
        # Get capabilities
        capabilities = db.query(ProviderCapabilities).filter(
            ProviderCapabilities.provider_id == provider.id
        ).all()

        capability_list = [
            {
                "type": cap.capability_type,
                "value": cap.capability_value,
                "metadata": cap.additional_data  # Changed from metadata to additional_data
            }
            for cap in capabilities
        ]

        provider_list.append({
            "id": provider.id,
            "provider_id": provider.provider_id,
            "config": provider.config,
            "is_active": provider.is_active,
            "created_at": provider.created_at.isoformat(),
            "updated_at": provider.updated_at.isoformat(),
            "capabilities": capability_list
        })

    return StandardResponse(
        status="success",
        data=provider_list
    )

@app.put("/api/providers/{provider_id}/toggle", response_model=StandardResponse)
async def toggle_provider(provider_id: str, db: Session = Depends(get_db)):
    """
    Toggle a provider's active state.

    Args:
        provider_id: Provider ID
        db: Database session

    Returns:
        StandardResponse with updated provider
    """
    # Import model here to avoid circular imports
    from multiagent.app.db.models import ProviderConfig
    
    provider = db.query(ProviderConfig).filter(ProviderConfig.provider_id == provider_id).first()

    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")

    # Toggle is_active
    provider.is_active = not provider.is_active
    db.commit()

    return StandardResponse(
        status="success",
        data={
            "provider_id": provider.provider_id,
            "is_active": provider.is_active
        },
        message=f"Provider {provider_id} {'activated' if provider.is_active else 'deactivated'}"
    )

@app.get("/api/provider-performance", response_model=StandardResponse)
async def get_provider_performance(
    provider_id: Optional[str] = None,
    task_type: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get provider performance metrics.

    Args:
        provider_id: Optional provider ID filter
        task_type: Optional task type filter
        limit: Maximum number of records
        db: Database session

    Returns:
        StandardResponse with performance metrics
    """
    # Import models here to avoid circular imports
    from multiagent.app.db.models import ProviderConfig, ProviderPerformance
    
    query = db.query(ProviderPerformance)

    # Apply filters
    if provider_id:
        provider = db.query(ProviderConfig).filter(ProviderConfig.provider_id == provider_id).first()
        if provider:
            query = query.filter(ProviderPerformance.provider_id == provider.id)

    if task_type:
        query = query.filter(ProviderPerformance.task_type == task_type)

    # Get records
    records = query.order_by(ProviderPerformance.recorded_at.desc()).limit(limit).all()

    # Format results
    performance_data = []
    for record in records:
        provider = db.query(ProviderConfig).filter(ProviderConfig.id == record.provider_id).first()

        performance_data.append({
            "id": record.id,
            "provider_id": provider.provider_id if provider else None,
            "model_id": record.model_id,
            "task_type": record.task_type,
            "latency": record.latency,
            "success_rate": record.success_rate,
            "cost": record.cost,
            "quality_score": record.quality_score,
            "tokens_input": record.tokens_input,
            "tokens_output": record.tokens_output,
            "recorded_at": record.recorded_at.isoformat(),
            "metadata": record.additional_data  # Changed from metadata to additional_data
        })

    return StandardResponse(
        status="success",
        data=performance_data
    )

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