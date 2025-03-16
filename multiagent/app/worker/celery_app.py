"""
Celery configuration for asynchronous task processing.
Sets up the Celery app and its configuration.
"""

import logging
import os
from celery import Celery
from celery.signals import task_postrun, task_prerun, worker_init, worker_ready
from datetime import datetime
from multiagent.app.api.websocket import connection_manager
from multiagent.app.db.session import SessionLocal
from multiagent.app.db.crud.results import crud_result

from multiagent.app.core.config import settings


logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "multiagent_llm",
    broker=settings.RABBITMQ_URI,
    backend=settings.REDIS_URI
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3000,  # 50 minutes
    worker_prefetch_multiplier=1,  # Disable prefetching
    worker_max_tasks_per_child=200,  # Restart worker after 200 tasks
    result_expires=86400,  # Results expire after 1 day
    # Use Redis for task results
    result_backend=settings.REDIS_URI,
    # Use Redis for task routing
    task_routes={
        "app.worker.tasks.execute_workflow_task": {"queue": "workflows"},
        "app.worker.tasks.execute_agent_task": {"queue": "agents"},
        "app.worker.tasks.update_progress": {"queue": "updates"},
    },
)

# Import tasks module to register tasks
import multiagent.app.worker.tasks


@worker_init.connect
def init_worker(**kwargs):
    """
    Run when the worker starts up.
    Initialize resources needed by the worker.
    """
    logger.info("Initializing Celery worker")
    # Initialize database connection if needed
    from app.db.session import init_db
    init_db()
    logger.info("Celery worker initialized")


@worker_ready.connect
def ready_worker(**kwargs):
    """
    Run when the worker is ready to accept tasks.
    """
    logger.info("Celery worker ready to accept tasks")


@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """
    Run before a task is executed.
    
    Args:
        task_id: ID of the task
        task: Task instance
    """
    logger.info(f"Starting task: {task.name} (ID: {task_id})")
    
    # Only handle workflow tasks
    if task.name == "app.worker.tasks.execute_workflow_task":
        try:
            # Get task arguments
            task_args = args[0] if args else []
            task_kwargs = kwargs.get('kwargs', {})
            
            # Extract workflow_id and user_id if available
            workflow_id = task_args[0] if len(task_args) > 0 else task_kwargs.get('workflow_id')
            input_data = task_args[1] if len(task_args) > 1 else task_kwargs.get('input_data', {})
            user_id = input_data.get('user_id')
            
            # Create message for clients
            message = {
                "type": "task_started",
                "task_id": task_id,
                "workflow_id": workflow_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update database status to "started"
            with SessionLocal() as db:
                crud_result.update_status(db=db, task_id=task_id, status="processing")
            
            # Send notification to specific user if available, otherwise broadcast to all
            from multiagent.app.api.websocket import connection_manager
            if user_id:
                connection_manager.broadcast(message, user_id)
            else:
                connection_manager.broadcast_all(message)
                
            logger.info(f"Sent start notification for task {task_id}")
                
        except Exception as e:
            logger.error(f"Error sending task start notification: {str(e)}")


@task_postrun.connect
def task_postrun_handler(task_id, task, retval, state, *args, **kwargs):
    """
    Run after a task is executed.
    
    Args:
        task_id: ID of the task
        task: Task instance
        retval: Return value of the task
        state: Task state
    """
    logger.info(f"Task completed: {task.name} (ID: {task_id}, State: {state})")
    
    # Only send notifications for workflow tasks
    if task.name == "app.worker.tasks.execute_workflow_task":
        try:
            # Create message for clients
            message = {
                "type": "task_completed",
                "task_id": task_id,
                "status": state,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add result to message if successful
            if state == "SUCCESS" and retval:
                message["result"] = retval
            
            # Get user_id for the task to direct the notification
            user_id = None
            with SessionLocal() as db:
                result_obj = crud_result.get_by_task_id(db=db, task_id=task_id)
                if result_obj:
                    user_id = result_obj.user_id
                    
                    # Update the result status in the database
                    if state == "SUCCESS":
                        crud_result.update_status(db=db, task_id=task_id, status="completed")
                    elif state in ["FAILURE", "REVOKED"]:
                        crud_result.update_status(db=db, task_id=task_id, status="failed")
            
            # Send notification to specific user if available, otherwise broadcast to all
            if user_id:
                connection_manager.broadcast(message, user_id)
            else:
                connection_manager.broadcast_all(message)
                
            logger.info(f"Sent completion notification for task {task_id}")
                
        except Exception as e:
            logger.error(f"Error sending task completion notification: {str(e)}")