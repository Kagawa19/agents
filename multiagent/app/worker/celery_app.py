"""
Celery configuration for asynchronous task processing.
Sets up the Celery app and its configuration.
"""

import logging
import os
from celery import Celery
from celery.signals import task_postrun, task_prerun, worker_init, worker_ready

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