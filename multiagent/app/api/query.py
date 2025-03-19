# api/query.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import uuid
import logging
from datetime import datetime
import traceback
import time
import os
from dotenv import load_dotenv

# Import task queue at the top level is fine
from multiagent.app.worker.queue import TaskQueue
from multiagent.app.monitoring.tracer import LangfuseTracer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize the task queue
task_queue = TaskQueue()

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
    # Import here to avoid circular imports
    from multiagent.app.worker.tasks import execute_workflow_task
    
    # Debug point 1: Log incoming query data
    logger.debug(f"Received query data: {query_data}")
    
    # Create QueryRequest for validation
    request = QueryRequest(**query_data)
    
    # Check if request is valid
    if not request.validate():
        logger.error(f"Invalid query parameters: {query_data}")
        raise ValueError("Invalid query parameters")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Debug point 2: Log task ID
    logger.debug(f"Generated task_id: {task_id}")
    
    # Create trace for monitoring
    try:
        trace_id_obj = tracer.create_trace(
            task_id=task_id,
            name="query_execution",
            metadata={
                "query": request.query, 
                "workflow_type": request.workflow_type,
                "user_id": request.user_id
            }
        )
        
        # Debug point 3: Log trace creation
        logger.debug(f"Created trace: {trace_id_obj}")
        
    except Exception as trace_error:
        logger.error(f"Error creating trace: {trace_error}")
        # Continue without trace if it fails
        trace_id_obj = {"id": "error-creating-trace"}
    
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
    
    # Debug point 4: Log task kwargs
    logger.debug(f"Task kwargs prepared: {task_kwargs}")
    
    # Only add priority if it's not None
    if request.priority is not None:
        task_kwargs['input_data']['priority'] = request.priority
        logger.debug(f"Added priority: {request.priority}")
    
    try:
        # Debug point 5: Log before task submission
        logger.info(f"About to submit task with: workflow_id={request.workflow_type}")
        
        # REMOVE THE SLEEP - this is causing delays
        # import time
        # time.sleep(10)  
        
        # Submit the execute_workflow_task with more detailed logging
        result = execute_workflow_task.delay(**task_kwargs)
        
        # Debug point 6: Log task submission result
        logger.info(f"Task submitted successfully. Task ID from result: {result.id}")
        logger.debug(f"Celery task result object: {result}")
        
        # Add additional debugging information in the response
        return {
            "task_id": task_id,
            "celery_task_id": result.id,
            "status": "submitted",
            "submitted_at": datetime.utcnow().isoformat(),
            "workflow_type": request.workflow_type
        }
    except Exception as e:
        # Log detailed error information with better formatting
        logger.error(f"Task submission failed: {str(e)}") 
        logger.error(f"Task kwargs: {task_kwargs}")
        # Include more details about the exception for debugging
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

async def update_task_progress(
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
    # Import here to avoid circular imports
    from multiagent.app.worker.tasks import update_progress
    
    try:
        # Debug logging progress update
        logger.debug(f"Updating progress for task {task_id}: status={status}, progress={progress}")
        
        # Call the update_progress task
        progress_task = update_progress.delay(
            task_id=task_id,
            status=status,
            progress=progress,
            current_step=current_step,
            result=result,
            error=error
        )
        
        logger.debug(f"Progress update submitted: {progress_task.id}")
        return True
    except Exception as e:
        logger.error(f"Error updating progress for task {task_id}: {str(e)}")
        logger.error(traceback.format_exc())
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
    
    try:
        # Get status from task queue
        status_data = await task_queue.get_task_status(task_id)
        
        logger.debug(f"Retrieved status data: {status_data}")
        
        if not status_data:
            logger.warning(f"Task {task_id} not found")
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
    except Exception as e:
        logger.error(f"Error getting query status: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def get_query_result(task_id: str) -> Dict[str, Any]:
    """
    Retrieves the result of a completed query.
    
    Args:
        task_id: The task ID to retrieve
        
    Returns:
        Dict containing the query result
    """
    logger.debug(f"Retrieving result for task: {task_id}")
    
    try:
        # Get result from task queue
        result_data = await task_queue.get_task_result(task_id)
        
        logger.debug(f"Retrieved result data: {result_data}")
        
        if not result_data:
            logger.warning(f"Result for task {task_id} not found")
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
    except Exception as e:
        logger.error(f"Error getting query result: {str(e)}")
        logger.error(traceback.format_exc())
        raise