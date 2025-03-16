# api/query.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import uuid
import logging
from datetime import datetime
from multiagent.app.worker.tasks import update_progress
# Import from other modules
from multiagent.app.worker.queue import TaskQueue
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """Validates user query requests."""
    query: str
    user_id: Optional[str] = None
    workflow_type: Optional[str] = "research"  # Default to research workflow
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority: Optional[int] = None
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query string."""
        if not v or len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        if len(v) > 1000:
            raise ValueError("Query must be less than 1000 characters")
        return v.strip()
    
    def validate(self) -> bool:
        """
        Validates query parameters.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            self.validate_query(self.query)
            return True
        except ValueError:
            return False

class QueryStatus(BaseModel):
    """Represents the status of a query."""
    task_id: str
    status: str
    progress: Optional[float] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
class QueryResponse(BaseModel):
    """Represents the response to a query."""
    task_id: str
    result: Dict[str, Any]
    status: str
    execution_time: Optional[float] = None
    created_at: Optional[datetime] = None

# Initialize task queue and tracer
task_queue = TaskQueue()
tracer = LangfuseTracer()

async def submit_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Receives user query, validates it, and submits task to the queue.
    
    Args:
        query_data: Query parameters including the query text
        
    Returns:
        Dict containing task_id and initial status
    """
    # Create QueryRequest for validation
    request = QueryRequest(**query_data)
    
    # Check if request is valid
    if not request.validate():
        raise ValueError("Invalid query parameters")
    
    # Generate task ID if not provided
    task_id = str(uuid.uuid4())
    
    # Create trace for monitoring
    trace_id = await tracer.create_trace(
        task_id=task_id, 
        trace_name="query_execution",
        metadata={
            "query": request.query,
            "workflow_type": request.workflow_type,
            "user_id": request.user_id
        }
    )
    
    logger.info(f"Submitting query: {request.query[:50]}... (task_id: {task_id})")
    
    # Add trace_id to task data
    task_data = {
        "task_id": task_id,
        "query": request.query,
        "workflow_type": request.workflow_type,
        "parameters": request.parameters,
        "user_id": request.user_id,
        "trace_id": trace_id
    }
    
    # Submit task to queue
    result = await task_queue.submit_task(task_data, priority=request.priority)
    
    return {
        "task_id": task_id,
        "status": "submitted",
        "submitted_at": datetime.utcnow().isoformat()
    }

def update_task_progress(
    self,
    task_id: str,
    status: str,
    progress: int,
    current_step: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> bool:
    """
    Update the progress of a task.
    
    Args:
        task_id: ID of the task
        status: Status of the task
        progress: Progress percentage (0-100)
        current_step: Current step being executed
        result: Task result (if completed)
        error: Error message (if failed)
    
    Returns:
        True if the update was successful, False otherwise
    """
    try:
        # Call the update_progress task
        update_progress.delay(
            task_id=task_id,
            status=status,
            progress=progress,
            current_step=current_step,
            result=result,
            error=error
        )
        logger.debug(f"Updated progress for task {task_id}: {status} ({progress}%)")
        return True
    except Exception as e:
        logger.error(f"Error updating progress for task {task_id}: {str(e)}")
        return False

async def get_query_status(task_id: str) -> Dict[str, Any]:
    """
    Checks the status of a query by task ID.
    
    Args:
        task_id: The task ID to check
        
    Returns:
        Dict containing status information
    """
    logger.debug(f"Checking status for task: {task_id}")
    
    # Get status from task queue
    status_data = await task_queue.get_task_status(task_id)
    
    if not status_data:
        raise ValueError(f"Task {task_id} not found")
    
    # Format as QueryStatus
    status = QueryStatus(
        task_id=task_id,
        status=status_data.get("status", "unknown"),
        progress=status_data.get("progress", 0.0),
        started_at=status_data.get("started_at"),
        updated_at=status_data.get("updated_at")
    )
    
    return status.dict()

async def get_query_result(task_id: str) -> Dict[str, Any]:
    """
    Retrieves the result of a completed query.
    
    Args:
        task_id: The task ID to retrieve
        
    Returns:
        Dict containing the query result
    """
    logger.debug(f"Retrieving result for task: {task_id}")
    
    # Get result from task queue
    result_data = await task_queue.get_task_result(task_id)
    
    if not result_data:
        raise ValueError(f"Result for task {task_id} not found")
    
    # Check if task is completed
    if result_data.get("status") != "completed":
        return {
            "task_id": task_id,
            "status": result_data.get("status", "pending"),
            "progress": result_data.get("progress", 0.0)
        }
    
    # Format as QueryResponse
    response = QueryResponse(
        task_id=task_id,
        result=result_data.get("result", {}),
        status="completed",
        execution_time=result_data.get("execution_time"),
        created_at=result_data.get("created_at")
    )
    
    return response.dict()