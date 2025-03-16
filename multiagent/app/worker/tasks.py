"""
Task definitions for asynchronous processing.
Defines Celery tasks for executing workflows and agents.
"""

import logging
import time
from typing import Any, Dict, Optional
import traceback
from celery import states
from celery.exceptions import MaxRetriesExceededError, Retry

from multiagent.app.api.websocket import connection_manager
from multiagent.app.db.crud.results import crud_result
from multiagent.app.db.session import SessionLocal
from multiagent.app.worker.celery_app import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.worker.tasks.execute_workflow_task")
@celery_app.task(bind=True, name="app.worker.tasks.execute_workflow_task")
def execute_workflow_task(
    self,
    workflow_id: str,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a workflow asynchronously.
    
    Args:
        workflow_id: ID of the workflow to execute
        input_data: Input data for the workflow
    
    Returns:
        The workflow result
    """
    # Store the task ID in the input data
    input_data["task_id"] = self.request.id
    
    logger.info(f"Executing workflow {workflow_id} (Task ID: {self.request.id})")
    
    # Save task to database
    with SessionLocal() as db:
        crud_result.save_result(
            db=db,
            task_id=self.request.id,
            query=input_data.get("query", ""),
            workflow=workflow_id,
            user_id=input_data.get("user_id"),
            status="processing"
        )
    
    # Send initial progress update
    update_progress.delay(
        task_id=self.request.id,
        status="processing",
        progress=10,
        current_step="Initializing workflow execution"
    )
    
    try:
        # Initialize components
        try:
            # Initialize the workflow manager
            from multiagent.app.monitoring.tracer import get_tracer
            from multiagent.app.orchestrator.manager import AgentManager
            from multiagent.app.orchestrator.workflow import WorkflowManager
            from multiagent.app.core.config import settings
            
            # Send progress update
            update_progress.delay(
                task_id=self.request.id,
                status="processing",
                progress=20,
                current_step="Setting up execution environment"
            )
            
            tracer = get_tracer()
            agent_manager = AgentManager(settings, tracer)
            agent_manager.initialize()
            
            # Send progress update
            update_progress.delay(
                task_id=self.request.id,
                status="processing",
                progress=30,
                current_step="Initializing agent manager"
            )
            
            workflow_manager = WorkflowManager(agent_manager, tracer)
            
            # Send progress update
            update_progress.delay(
                task_id=self.request.id,
                status="processing",
                progress=40,
                current_step="Environment setup complete"
            )
            
        except Exception as setup_error:
            # Handle setup errors separately
            error_msg = f"Error setting up workflow environment: {str(setup_error)}"
            logger.error(error_msg)
            
            # Send specific error progress update
            update_progress.delay(
                task_id=self.request.id,
                status="failed",
                progress=0,
                error=error_msg
            )
            
            # Save specific error to database
            with SessionLocal() as db:
                crud_result.save_result(
                    db=db,
                    task_id=self.request.id,
                    query=input_data.get("query", ""),
                    workflow=workflow_id,
                    result={"error": error_msg},
                    user_id=input_data.get("user_id"),
                    status="failed"
                )
            
            # Re-raise with more specific message
            raise Exception(error_msg)
        
        # Execute the workflow with progress updates
        start_time = time.time()
        
        # Send progress update before execution
        update_progress.delay(
            task_id=self.request.id,
            status="processing",
            progress=50,
            current_step="Starting workflow execution"
        )
        
        result = workflow_manager.execute_workflow(workflow_id, input_data)
        execution_time = time.time() - start_time
        
        # Send progress update after execution
        update_progress.delay(
            task_id=self.request.id,
            status="processing",
            progress=90,
            current_step="Workflow execution completed, finalizing results"
        )
        
        # Save result to database
        with SessionLocal() as db:
            crud_result.save_result(
                db=db,
                task_id=self.request.id,
                query=input_data.get("query", ""),
                workflow=workflow_id,
                result=result,
                user_id=input_data.get("user_id"),
                status="completed"
            )
        
        # Add execution time to the result
        result["execution_time"] = execution_time
        
        # Send completion notification
        update_progress.delay(
            task_id=self.request.id,
            status="completed",
            progress=100,
            result=result
        )
        
        logger.info(f"Workflow {workflow_id} completed successfully (Task ID: {self.request.id})")
        return result
    
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Error executing workflow {workflow_id}: {error_msg}\n{stack_trace}")
        
        # Create structured error response
        error_details = {
            "error": error_msg,
            "workflow_id": workflow_id,
            "task_id": self.request.id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "trace": stack_trace if settings.DEBUG else "Enable DEBUG mode to see trace"
        }
        
        # Save error to database
        with SessionLocal() as db:
            crud_result.save_result(
                db=db,
                task_id=self.request.id,
                query=input_data.get("query", ""),
                workflow=workflow_id,
                result=error_details,
                user_id=input_data.get("user_id"),
                status="failed"
            )
        
        # Send error notification
        update_progress.delay(
            task_id=self.request.id,
            status="failed",
            progress=0,
            error=error_msg
        )
        
        # Re-raise the exception with structured error
        raise Exception(f"Workflow execution failed: {error_msg}")

@celery_app.task(bind=True, name="app.worker.tasks.execute_agent_task")
def execute_agent_task(
    self,
    agent_id: str,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute an agent asynchronously.
    
    Args:
        agent_id: ID of the agent to execute
        input_data: Input data for the agent
    
    Returns:
        The agent result
    """
    logger.info(f"Executing agent {agent_id} (Task ID: {self.request.id})")
    
    try:
        # Initialize the agent manager
        from multiagent.app.monitoring.tracer import get_tracer
        from multiagent.app.orchestrator.manager import AgentManager
        from multiagent.app.core.config import settings
        
        tracer = get_tracer()
        agent_manager = AgentManager(settings, tracer)
        agent_manager.initialize()
        
        # Execute the agent
        start_time = time.time()
        result = agent_manager.execute_agent(agent_id, input_data)
        execution_time = time.time() - start_time
        
        # Add execution time to the result
        result["execution_time"] = execution_time
        
        logger.info(f"Agent {agent_id} executed successfully (Task ID: {self.request.id})")
        return result
    
    except Exception as e:
        logger.error(f"Error executing agent {agent_id}: {str(e)}")
        # Re-raise the exception
        raise


@celery_app.task(name="app.worker.tasks.update_progress")
def update_progress(
    task_id: str,
    status: str,
    progress: int,
    current_step: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update the progress of a task and send a WebSocket notification.
    
    Args:
        task_id: ID of the task
        status: Status of the task
        progress: Progress percentage (0-100)
        current_step: Current step being executed
        result: Task result (if completed)
        error: Error message (if failed)
    
    Returns:
        The update message
    """
    # Create message
    message = {
        "type": "progress_update",
        "task_id": task_id,
        "status": status,
        "progress": progress
    }
    
    if current_step:
        message["current_step"] = current_step
    
    if result:
        message["result"] = result
    
    if error:
        message["error"] = error
    
    # Update database status
    with SessionLocal() as db:
        crud_result.update_status(db=db, task_id=task_id, status=status)
    
    # Send WebSocket notification
    # Since this is a background task, we need to get the user_id from the database
    with SessionLocal() as db:
        result_obj = crud_result.get_by_task_id(db=db, task_id=task_id)
        if result_obj and result_obj.user_id:
            # Send to the specific user
            connection_manager.broadcast(message, result_obj.user_id)
        else:
            # Broadcast to all - less secure but works when user_id is not available
            connection_manager.broadcast_all(message)
    
    logger.info(f"Progress update for task {task_id}: {status} ({progress}%)")
    return message