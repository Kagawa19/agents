from typing import Dict, Optional

from pydantic import BaseModel, Field


class HealthCheck(BaseModel):
    """
    Schema for health check response.
    Indicates the status of various system components.
    """
    status: str = Field(..., description="Overall status (ok, warning, error)")
    api: bool = Field(..., description="API availability")
    database: bool = Field(..., description="Database connection status")
    redis: Optional[bool] = Field(None, description="Redis connection status")
    rabbitmq: Optional[bool] = Field(None, description="RabbitMQ connection status")
    agents: Optional[Dict[str, bool]] = Field(None, description="Status of individual agents")
    tools: Optional[Dict[str, bool]] = Field(None, description="Status of external tools")
    version: Optional[str] = Field(None, description="API version")
    uptime: Optional[float] = Field(None, description="API uptime in seconds")


class ComponentHealth(BaseModel):
    """
    Schema for detailed component health information.
    """
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status (ok, warning, error)")
    message: Optional[str] = Field(None, description="Additional status message")
    response_time: Optional[float] = Field(None, description="Response time in milliseconds")
    last_checked: str = Field(..., description="Timestamp of last check")