"""
Task definitions for asynchronous processing.
Defines Celery tasks for executing workflows and agents.
"""
import json
import logging
import time
from typing import Any, Dict, Optional
import traceback
from celery import states
from celery.exceptions import MaxRetriesExceededError, Retry

from multiagent.app.api.websocket import connection_manager
from multiagent.app.db.models import Result
from multiagent.app.db.results import crud_result
from multiagent.app.db.session import SessionLocal
from multiagent.app.worker.celery_app import celery_app


logger = logging.getLogger(__name__)




import asyncio
import time
import traceback
from typing import Dict, Any

@celery_app.task(bind=True, name="app.worker.tasks.execute_workflow_task")
def execute_workflow_task(
    self,
    workflow_id: str,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a workflow asynchronously with robust error handling and async support.
    
    Args:
        workflow_id: ID of the workflow to execute
        input_data: Input data for the workflow
    
    Returns:
        The workflow result
    """
    # Extract input data
    task_id = input_data.get("task_id")
    query = input_data.get("query")
    user_id = input_data.get("user_id")
    parameters = input_data.get("parameters", {})
    trace_id = input_data.get("trace_id")
    
    # Enhanced logging with more context
    logger.info(
        f"Starting workflow execution: workflow={workflow_id}, "
        f"task_id={task_id}, user_id={user_id}, "
        f"query={query[:100] if query else 'N/A'}"
    )
    
    # Detailed log for input parameters
    logger.debug(f"Workflow input parameters: {parameters}")
    
    # Save task to database
    try:
        with SessionLocal() as db:
            crud_result.save_result(
                db=db,
                task_id=task_id,
                query=query,
                workflow=workflow_id,
                user_id=user_id,
                status="processing"
            )
        logger.info(f"Task {task_id} saved to database")
    except Exception as save_error:
        logger.error(
            f"Failed to save task to database: {save_error}", 
            exc_info=True
        )
    
    # Send initial progress update
    update_progress.delay(
        task_id=task_id,
        status="processing",
        progress=10,
        current_step="Initializing workflow execution"
    )
    
    try:
        # Initialize components with detailed logging
        try:
            logger.info("Setting up execution environment")
            
            from multiagent.app.monitoring.tracer import get_tracer
            from multiagent.app.orchestrator.manager import AgentManager
            from multiagent.app.orchestrator.workflow import WorkflowManager
            from multiagent.app.core.config import settings
            
            # Progress update
            update_progress.delay(
                task_id=task_id,
                status="processing",
                progress=20,
                current_step="Setting up execution environment"
            )
            
            tracer = get_tracer()
            logger.info("Tracer initialized")
            
            agent_manager = AgentManager(settings, tracer)
            agent_manager.initialize()
            logger.info("Agent manager initialized")
            
            update_progress.delay(
                task_id=task_id,
                status="processing", 
                progress=30,
                current_step="Initializing agent manager"
            )
            
            workflow_manager = WorkflowManager(agent_manager, tracer)
            logger.info("Workflow manager created")
            
        except Exception as setup_error:
            # Detailed error logging for setup failures
            error_msg = f"Error setting up workflow environment: {str(setup_error)}"
            logger.error(error_msg, exc_info=True)
            
            # Send specific error progress update
            update_progress.delay(
                task_id=task_id,
                status="failed",
                progress=0,
                error=error_msg
            )
            
            # Save specific error to database
            with SessionLocal() as db:
                crud_result.save_result(
                    db=db,
                    task_id=task_id,
                    query=query,
                    workflow=workflow_id,
                    result={"error": error_msg},
                    user_id=user_id,
                    status="failed"
                )
            
            # Re-raise with more specific message
            raise Exception(error_msg)
        
        # Execute the workflow with progress updates
        start_time = time.time()
        
        # Log before workflow execution
        logger.info(f"Beginning workflow execution: {workflow_id}")
        
        update_progress.delay(
            task_id=task_id,
            status="processing",
            progress=50,
            current_step="Starting workflow execution"
        )
        
        # Robust async workflow execution
        async def safe_workflow_execute():
            try:
                # Use WorkflowManager to execute
                result = await workflow_manager.execute_workflow(workflow_id, input_data)
                
                # Ensure result is fully serializable
                try:
                    json.dumps(result)
                except TypeError:
                    result = str(result)
                
                return result
            
            except Exception as e:
                logger.error(f"Workflow execution internal error: {e}")
                raise
        
        # Run async workflow execution
        result = asyncio.run(safe_workflow_execute())
        
        execution_time = time.time() - start_time
        
        # Log successful execution
        logger.info(
            f"Workflow {workflow_id} completed successfully. "
            f"Execution time: {execution_time:.2f} seconds"
        )
        
        # Progress update
        update_progress.delay(
            task_id=task_id,
            status="processing",
            progress=90,
            current_step="Workflow execution completed, finalizing results"
        )
        
        # Save result to database  
        with SessionLocal() as db:
            crud_result.save_result(
                db=db,
                task_id=task_id,
                query=query,
                workflow=workflow_id,
                result=result,
                user_id=user_id,
                status="completed"
            )
        
        # Add execution time to the result
        result["execution_time"] = execution_time
        
        # Send completion notification
        update_progress.delay(
            task_id=task_id,
            status="completed",
            progress=100,
            result=result
        )
        
        logger.info(f"Workflow {workflow_id} fully processed (Task ID: {task_id})")
        return result
    
    except Exception as e:
        # Comprehensive error handling and logging
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        
        # Log the full error details
        logger.error(
            f"Workflow execution failed: {error_msg}\n"
            f"Workflow ID: {workflow_id}, Task ID: {task_id}",
            exc_info=True
        )
        
        # Create structured error response
        error_details = {
            "error": error_msg,
            "workflow_id": workflow_id,
            "task_id": task_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "trace": stack_trace if settings.DEBUG else "Enable DEBUG mode to see trace" 
        }
        
        # Save error to database
        with SessionLocal() as db:
            crud_result.save_result(
                db=db,
                task_id=task_id,
                query=query,
                workflow=workflow_id, 
                result=error_details,
                user_id=user_id,
                status="failed"
            )
        
        # Send error notification  
        update_progress.delay(
            task_id=task_id,
            status="failed",
            progress=0,
            error=error_msg  
        )
        
        # Log final error state
        logger.error(f"Workflow {workflow_id} failed with error: {error_msg}")
        
        # Re-raise the exception with structured error
        raise Exception(f"Workflow execution failed: {error_msg}")

@celery_app.task(name="your_app.tasks.update_progress")
def update_progress(
    task_id: str,
    status: str,
    progress: int,
    current_step: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update the progress of a task and send a notification.
    
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
    try:
        with SessionLocal() as db:
            db.query(Result).filter(Result.task_id == task_id).update({
                'status': status,
                'progress': progress
            })
            db.commit()
    except Exception as e:
        logger.error(f"Error updating task progress: {e}")
    
    # TODO: Implement WebSocket or other notification mechanism
    # You might want to add logic to broadcast the message to the appropriate client
    
    logger.info(f"Progress update for task {task_id}: {status} ({progress}%)")
    return message

@celery_app.task(name="your_app.tasks.execute_agent_task")
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
        # Initialize agent manager
        from multiagent.app.monitoring.tracer import get_tracer
        from multiagent.app.orchestrator.manager import AgentManager
        from multiagent.app.core.config import settings
        
        # Create tracer and agent manager
        tracer = get_tracer()
        agent_manager = AgentManager(settings, tracer)
        agent_manager.initialize()
        
        # Start timing
        start_time = time.time()
        
        # Execute the agent
        result = agent_manager.execute_agent(agent_id, input_data)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Ensure result is JSON serializable
        try:
            import json
            json.dumps(result)
        except TypeError:
            result = str(result)
        
        # Add execution metadata
        result = {
            "status": "success",
            "data": result,
            "execution_time": execution_time
        }
        
        logger.info(f"Agent {agent_id} executed successfully (Task ID: {self.request.id})")
        return result
    
    except Exception as e:
        # Log the error
        logger.error(f"Error executing agent {agent_id}: {str(e)}")
        
        # Create error response
        error_result = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        # Optionally save error to database or send notification
        try:
            with SessionLocal() as db:
                crud_result.save_result(
                    db=db,
                    task_id=self.request.id,
                    query=input_data.get('query', ''),
                    workflow='agent_execution',
                    result=error_result,
                    status="failed"
                )
        except Exception as save_error:
            logger.error(f"Failed to save error result: {save_error}")
        
        raise