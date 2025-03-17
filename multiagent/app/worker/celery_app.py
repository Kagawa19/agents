"""
Comprehensive Celery Application Configuration

This module sets up a robust Celery application with advanced configuration,
logging, and task management for a multi-agent system.
"""

import logging
import os
import socket
from typing import Optional, Dict, Any

from celery import Celery
from celery.signals import (
    task_postrun, 
    task_prerun, 
    worker_init, 
    worker_ready, 
    celeryd_after_setup
)
from kombu import Queue, Exchange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_broker_url() -> str:
    """
    Retrieve the broker URL with intelligent fallback and logging.
    
    Returns:
        str: Configured broker URL
    """
    # Prioritize environment variables and provide multiple fallback options
    broker_urls = [
        os.environ.get('BROKER_URI'),
        os.environ.get('REDIS_URI', 'redis://localhost:6379/0'),
        os.environ.get('RABBITMQ_URI', 'amqp://guest:guest@localhost:5672//'),
    ]
    
    # Filter out None values and take the first valid URL
    valid_urls = [url for url in broker_urls if url]
    
    if not valid_urls:
        raise ValueError("No valid broker URL found. Please configure BROKER_URI or REDIS_URI.")
    
    # Take the first valid URL
    broker_url = valid_urls[0]
    
    # Handle comma-separated URLs by taking the first
    if ',' in broker_url:
        broker_url = broker_url.split(',')[0]
    
    logger.info(f"Using broker URL: {broker_url}")
    return broker_url

def get_backend_url() -> str:
    """
    Retrieve the result backend URL with intelligent fallback.
    
    Returns:
        str: Configured backend URL
    """
    # Similar logic to get_broker_url
    backend_urls = [
        os.environ.get('BACKEND_URI'),
        os.environ.get('REDIS_URI', 'redis://localhost:6379/0'),
    ]
    
    valid_urls = [url for url in backend_urls if url]
    
    if not valid_urls:
        raise ValueError("No valid backend URL found. Please configure BACKEND_URI or REDIS_URI.")
    
    backend_url = valid_urls[0]
    
    # Handle comma-separated URLs
    if ',' in backend_url:
        backend_url = backend_url.split(',')[0]
    
    logger.info(f"Using backend URL: {backend_url}")
    return backend_url

# Define exchanges for task routing
workflows_exchange = Exchange('workflows', type='direct')
agents_exchange = Exchange('agents', type='direct')
updates_exchange = Exchange('updates', type='direct')

# Create Celery application
def create_celery_app() -> Celery:
    """
    Create and configure Celery application with comprehensive settings.
    
    Returns:
        Celery: Configured Celery application
    """
    # Create Celery app
    celery_app = Celery('multiagent_llm')
    
    # Configuration dictionary
    celery_app.conf.update(
        # Connection settings
        broker_url=get_broker_url(),
        result_backend=get_backend_url(),
        
        # Broker transport options
        broker_transport_options={
            'visibility_timeout': 3600,  # 1 hour
            'socket_timeout': 30,        # 30 seconds
            'socket_connect_timeout': 30,  # 30 seconds
            'max_retries': 10,            # Max retries for connection
        },
        
        # Serialization
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        
        # Time and tracking
        task_track_started=True,
        task_time_limit=3600,  # 1 hour maximum runtime
        task_soft_time_limit=3500,  # 58.3 minutes soft limit
        
        # Retry settings
        broker_connection_retry=True,
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=10,
        broker_connection_timeout=30,
        
        # Result handling
        result_expires=86400,  # Results expire after 1 day
        result_persistent=True,  # Store results reliably
        
        # Worker settings
        worker_concurrency=4,  # Number of concurrent tasks
        worker_prefetch_multiplier=1,  # One task per worker at a time
        worker_max_tasks_per_child=200,  # Restart worker after 200 tasks
        
        # Timezone
        enable_utc=True,
        timezone='UTC',
        
        # Task routing
        task_routes={
            'multiagent.app.worker.tasks.execute_workflow_task': {'queue': 'workflows'},
            'multiagent.app.worker.tasks.execute_agent_task': {'queue': 'agents'},
            'multiagent.app.worker.tasks.update_progress': {'queue': 'updates'},
        },
        
        # Queues
        task_queues=(
            Queue('workflows', workflows_exchange, routing_key='workflows'),
            Queue('agents', agents_exchange, routing_key='agents'),
            Queue('updates', updates_exchange, routing_key='updates'),
        ),
        
        # Default queue
        task_default_queue='workflows',
        task_default_exchange='workflows',
        task_default_routing_key='workflows',
        
        # Logging
        worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
        worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
    )
    
    # Autodiscover tasks
    celery_app.autodiscover_tasks(
        packages=['multiagent.app.worker'], 
        related_name='tasks'
    )
    
    return celery_app

# Create the Celery app instance
celery_app = create_celery_app()

# Celery signal handlers
@celeryd_after_setup.connect
def configure_worker(sender, instance, **kwargs):
    """
    Additional worker configuration after setup.
    
    Args:
        sender: Worker sender
        instance: Worker instance
        kwargs: Additional keyword arguments
    """
    logger.info(f"Worker {sender} ready. Configuring additional settings...")

@worker_init.connect
def init_worker(**kwargs):
    """
    Initialize worker with connection checks and additional setup.
    
    Args:
        kwargs: Additional keyword arguments
    """
    logger.info("Initializing Celery worker")
    
    # Log connection and system info
    logger.info(f"Broker URL: {celery_app.conf.broker_url}")
    logger.info(f"Result backend: {celery_app.conf.result_backend}")
    logger.info(f"Hostname: {socket.gethostname()}")
    
    # Additional initialization can be added here
    try:
        from multiagent.app.db.session import init_db
        init_db()
        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

@worker_ready.connect
def worker_ready(**kwargs):
    """
    Log when the worker is ready to accept tasks.
    
    Args:
        kwargs: Additional keyword arguments
    """
    logger.info("Celery worker is ready to accept tasks")

@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """
    Pre-task execution hook.
    
    Args:
        task_id: Unique task identifier
        task: Task instance
        args: Positional arguments
        kwargs: Keyword arguments
    """
    logger.info(f"Starting task: {task.name} (ID: {task_id})")

@task_postrun.connect
def task_postrun_handler(task_id, task, retval, state, *args, **kwargs):
    """
    Post-task execution hook.
    
    Args:
        task_id: Unique task identifier
        task: Task instance
        retval: Return value
        state: Task state
        args: Positional arguments
        kwargs: Keyword arguments
    """
    logger.info(f"Task completed: {task.name} (ID: {task_id}, State: {state})")

# Health check function (optional)
def test_celery_connection() -> bool:
    """
    Test Celery broker connection.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        # Ping the broker
        celery_app.control.ping(timeout=5)
        logger.info("Celery broker connection successful")
        return True
    except Exception as e:
        logger.error(f"Celery broker connection failed: {e}")
        return False

# Ensure the app is imported
__all__ = ['celery_app', 'test_celery_connection']