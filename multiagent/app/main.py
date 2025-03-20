import os
import logging
import asyncio
import uvicorn
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, Depends, HTTPException, BackgroundTasks, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import API components
from multiagent.app.api.query import submit_query, get_query_status, get_query_result, QueryRequest
from multiagent.app.api.websocket import websocket_endpoint
from multiagent.app.api.response import StandardResponse

# Import database components
from multiagent.app.db.session import engine, get_db, init_db, run_migrations
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
        
        # Run any pending migrations
        run_migrations()
        
        logger.info("Database initialized and migrated successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

# API Health check
@app.get("/api/health", response_model=StandardResponse)
async def health_check():
    """Health check endpoint."""
    return StandardResponse(
        success=True,
        message="Service is healthy",
        data={"status": "ok"}
    )

@app.post("/api/query", response_model=StandardResponse)
async def api_submit_query(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Submit a query to the system."""
    try:
        # Convert QueryRequest to a dictionary to match the submit_query function signature
        query_data = query_request.dict()
        
        # Call submit_query with just the query data
        task_id = await submit_query(query_data)
        
        return StandardResponse(
            success=True,
            message="Query submitted successfully",
            data={"task_id": task_id}
        )
    except Exception as e:
        logger.error(f"Error submitting query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting query: {str(e)}")

# Query status endpoint
@app.get("/api/query/{task_id}/status", response_model=StandardResponse)
async def api_get_query_status(
    task_id: str = Path(..., description="The task ID to check"),
    db: Session = Depends(get_db)
):
    """Get the status of a submitted query."""
    try:
        status = await get_query_status(task_id, db)
        return StandardResponse(
            success=True,
            message="Query status retrieved successfully",
            data=status
        )
    except Exception as e:
        logger.error(f"Error retrieving query status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving query status: {str(e)}")

# Query result endpoint
@app.get("/api/query/{task_id}/result", response_model=StandardResponse)
async def api_get_query_result(
    task_id: str = Path(..., description="The task ID to get results for"),
    db: Session = Depends(get_db)
):
    """Get the result of a submitted query."""
    try:
        result = await get_query_result(task_id, db)
        return StandardResponse(
            success=True,
            message="Query result retrieved successfully",
            data=result
        )
    except Exception as e:
        logger.error(f"Error retrieving query result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving query result: {str(e)}")

# Websocket endpoint
@app.websocket("/api/ws/{client_id}")
async def api_websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication."""
    await websocket_endpoint(websocket, client_id)

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

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
            {
                "provider_id": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                    "timeout": 60
                },
                "capabilities": [
                    {"capability_type": "text_generation", "capability_value": 0.9},
                    {"capability_type": "reasoning", "capability_value": 0.9},
                    {"capability_type": "summarization", "capability_value": 0.85}
                ]
            },
            {
                "provider_id": "anthropic",
                "config": {
                    "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                    "model": "claude-3-opus-20240229",
                    "timeout": 60
                },
                "capabilities": [
                    {"capability_type": "text_generation", "capability_value": 0.95},
                    {"capability_type": "reasoning", "capability_value": 0.95},
                    {"capability_type": "summarization", "capability_value": 0.9}
                ]
            },
            {
                "provider_id": "bedrock",
                "config": {
                    "region": os.getenv("AWS_REGION", "us-east-1"),
                    "model": os.getenv("BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0"),
                    "timeout": 120
                },
                "capabilities": [
                    {"capability_type": "text_generation", "capability_value": 0.85},
                    {"capability_type": "reasoning", "capability_value": 0.85},
                    {"capability_type": "summarization", "capability_value": 0.8}
                ]
            },
            {
                "provider_id": "jina",
                "config": {
                    "api_key": os.getenv("JINA_API_KEY", ""),
                    "timeout": 30
                },
                "capabilities": [
                    {"capability_type": "embedding", "capability_value": 0.95},
                    {"capability_type": "search", "capability_value": 0.9}
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

# Get all providers
@app.get("/api/providers")
async def get_providers(db: Session = Depends(get_db)):
    """Get all configured providers."""
    from multiagent.app.db.models import ProviderConfig
    
    try:
        providers = db.query(ProviderConfig).filter(ProviderConfig.is_active == True).all()
        return {"providers": [{"id": p.id, "provider_id": p.provider_id} for p in providers]}
    except Exception as e:
        logger.error(f"Error retrieving providers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving providers: {str(e)}")

# Get provider by ID
@app.get("/api/providers/{provider_id}")
async def get_provider(provider_id: str, db: Session = Depends(get_db)):
    """Get provider by ID."""
    from multiagent.app.db.models import ProviderConfig
    
    try:
        provider = db.query(ProviderConfig).filter(
            ProviderConfig.provider_id == provider_id,
            ProviderConfig.is_active == True
        ).first()
        
        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")
        
        return {
            "id": provider.id,
            "provider_id": provider.provider_id,
            "config": provider.config,
            "created_at": provider.created_at,
            "updated_at": provider.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving provider: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving provider: {str(e)}")

# Get provider capabilities
@app.get("/api/providers/{provider_id}/capabilities")
async def get_provider_capabilities(provider_id: str, db: Session = Depends(get_db)):
    """Get capabilities for a provider."""
    from multiagent.app.db.models import ProviderConfig, ProviderCapabilities
    
    try:
        provider = db.query(ProviderConfig).filter(
            ProviderConfig.provider_id == provider_id,
            ProviderConfig.is_active == True
        ).first()
        
        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")
        
        capabilities = db.query(ProviderCapabilities).filter(
            ProviderCapabilities.provider_id == provider.id
        ).all()
        
        return {
            "provider_id": provider_id,
            "capabilities": [
                {
                    "type": cap.capability_type,
                    "value": cap.capability_value,
                    "additional_data": cap.additional_data
                }
                for cap in capabilities
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving provider capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving provider capabilities: {str(e)}")

# Update provider configuration
@app.put("/api/providers/{provider_id}")
async def update_provider(
    provider_id: str,
    config: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """Update provider configuration."""
    from multiagent.app.db.models import ProviderConfig
    
    try:
        provider = db.query(ProviderConfig).filter(
            ProviderConfig.provider_id == provider_id
        ).first()
        
        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")
        
        provider.config = config
        db.commit()
        
        return {
            "message": f"Provider {provider_id} updated successfully",
            "provider_id": provider_id,
            "config": config
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating provider: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating provider: {str(e)}")

# Get all results
@app.get("/api/results")
async def get_results(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get query results with pagination."""
    from multiagent.app.db.models import Result
    from sqlalchemy import desc
    
    try:
        query = db.query(Result)
        
        if user_id:
            query = query.filter(Result.user_id == user_id)
        
        if status:
            query = query.filter(Result.status == status)
        
        total = query.count()
        
        results = query.order_by(desc(Result.created_at)).offset(offset).limit(limit).all()
        
        return {
            "results": [
                {
                    "id": r.id,
                    "task_id": r.task_id,
                    "query": r.query,
                    "status": r.status,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at
                }
                for r in results
            ],
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

# Run database migration manually
@app.post("/api/admin/migrate")
async def admin_run_migrations():
    """Run database migrations manually."""
    try:
        run_migrations()
        return {"message": "Database migrations completed successfully"}
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running migrations: {str(e)}")

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