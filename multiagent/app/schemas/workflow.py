from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowStep(BaseModel):
    """
    Schema for a workflow step.
    """
    agent_id: str = Field(..., description="ID of the agent to execute")
    description: str = Field(..., description="Description of the step")
    order: int = Field(..., description="Order in the workflow sequence")
    input_mapping: Dict[str, str] = Field(..., description="Mapping of state to agent input")


class WorkflowDefinition(BaseModel):
    """
    Schema for workflow definition.
    """
    workflow_id: str = Field(..., description="Workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    input_schema: Dict[str, Any] = Field(..., description="Expected input schema")
    output_schema: Dict[str, Any] = Field(..., description="Expected output schema")


class WorkflowList(BaseModel):
    """
    Schema for list of available workflows.
    """
    workflows: List[Dict[str, str]] = Field(..., description="List of available workflows")


class WorkflowExecution(BaseModel):
    """
    Schema for workflow execution details.
    """
    execution_id: str = Field(..., description="Execution ID")
    workflow_id: str = Field(..., description="Workflow ID")
    start_time: str = Field(..., description="Start time")
    end_time: Optional[str] = Field(None, description="End time")
    status: str = Field(..., description="Execution status")
    steps_completed: int = Field(..., description="Number of completed steps")
    steps_total: int = Field(..., description="Total number of steps")
    current_step: Optional[str] = Field(None, description="Current step")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result")