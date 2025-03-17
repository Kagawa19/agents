# api/query.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import uuid
import logging
from datetime import datetime
from multiagent.app.worker.tasks import update_progress, execute_workflow_task  # Import execute_workflow_task
# Import from other modules
from multiagent.app.monitoring.tracer import LangfuseTracer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """Validates user query requests."""
    query: str
    user_id: Optional[str] = None
    workflow_type: str = "research"  # Default to research workflow 
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

# Initialize tracer with environment variables
tracer = LangfuseTracer(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"), 
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://api.langfuse.com")
)

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
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create trace for monitoring
    trace_id_obj = tracer.create_trace(
        task_id=task_id,
        name="query_execution",
        metadata={
            "query": request.query, 
            "workflow_type": request.workflow_type,
            "user_id": request.user_id
        }
    )
    
    logger.info(f"Submitting query: {request.query[:50]}... (task_id: {task_id})")
    
    # Extract just the trace ID string instead of using the whole dictionary
    trace_id_str = trace_id_obj.get('id') if isinstance(trace_id_obj, dict) else str(trace_id_obj)
    
    # Prepare kwargs for task submission
    task_kwargs = {
        'workflow_id': request.workflow_type,
        'input_data': {
            'task_id': task_id,
            'query': request.query,
            'user_id': request.user_id,
            'parameters': request.parameters,
            'trace_id': trace_id_str
        }
    }
    
    # Only add priority if it's not None
    if request.priority is not None:
        task_kwargs['input_data']['priority'] = request.priority
    
    try:
        # Submit the execute_workflow_task
        result = execute_workflow_task.delay(**task_kwargs)
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "submitted_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        # Log detailed error information
        logger.error(f"Task submission failed: {e}") 
        logger.error(f"Task kwargs: {task_kwargs}")
        raise
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