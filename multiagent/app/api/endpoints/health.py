from fastapi import APIRouter, status

from multiagent.app.schemas.health import HealthCheck


router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthCheck, status_code=status.HTTP_200_OK)
async def health_check() -> HealthCheck:
    """
    Health check endpoint.
    Verifies that the API is operational.
    
    Returns:
        Health status
    """
    return {
        "status": "ok",
        "api": True,
        "database": True,
    }