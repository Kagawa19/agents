from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """
    Schema for agent request.
    Contains the input data for direct agent execution.
    """
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")


class AgentResponse(BaseModel):
    """
    Schema for agent response.
    Contains the result of agent execution.
    """
    agent_id: str = Field(..., description="ID of the executed agent")
    result: Dict[str, Any] = Field(..., description="Result data")
    success: bool = Field(..., description="Whether the agent executed successfully")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class AgentList(BaseModel):
    """
    Schema for list of available agents.
    """
    agents: List[str] = Field(..., description="List of available agent IDs")


class AgentInfo(BaseModel):
    """
    Schema for detailed agent information.
    """
    agent_id: str = Field(..., description="Agent ID")
    description: str = Field(..., description="Agent description")
    capabilities: List[str] = Field(..., description="List of agent capabilities")
    input_schema: Dict[str, Any] = Field(..., description="Expected input schema")
    output_schema: Dict[str, Any] = Field(..., description="Expected output schema")


class AgentMetrics(BaseModel):
    """
    Schema for agent performance metrics.
    """
    agent_id: str = Field(..., description="Agent ID")
    avg_execution_time: float = Field(..., description="Average execution time in seconds")
    success_rate: float = Field(..., description="Success rate (0-1)")
    total_executions: int = Field(..., description="Total number of executions")
    last_execution: str = Field(..., description="Timestamp of last execution")