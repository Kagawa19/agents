"""
Celery configuration for asynchronous task processing.
Sets up the Celery app and its configuration with robust networking and connection handling.
"""

import logging
import os
import socket
from typing import Optional
from celery import Celery
from kombu import Queue, Exchange
from celery.signals import task_postrun, task_prerun, worker_init, worker_ready, celeryd_after_setup
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import settings after logging setup
try:
    from multiagent.app.core.config import settings
    logger.info("Settings imported successfully")
except ImportError:
    logger.error("Failed to import settings, using environment variables")
    # Fallback to environment variables
    from os import environ
    
    class Settings:
        REDIS_URI = environ.get('REDIS_URI', 'redis://redis:6379/0')
        RABBITMQ_URI = environ.get('RABBITMQ_URI', 'amqp://guest:guest@rabbitmq:5672//')
    
    settings = Settings()

def get_broker_url():
    """
    Get the broker URL with fallbacks
    """
    # Try to get from settings first
    broker_url = getattr(settings, 'REDIS_URI', None)
    
    # If not found in settings, try environment
    if not broker_url:
        broker_url = os.environ.get('REDIS_URI', 'redis://redis:6379/0')
    
    # Handle comma-separated URLs by taking only the first one
    if broker_url and ',' in broker_url:
        broker_url = broker_url.split(',')[0]
        logger.info(f"Found comma-separated URLs, using only the first: {broker_url}")
    
    logger.info(f"Using broker URL: {broker_url}")
    return broker_url

def get_backend_url():
    """
    Get the result backend URL with fallbacks
    """
    # Try to get from settings first
    backend_url = getattr(settings, 'REDIS_URI', None)
    
    # If not found in settings, try environment
    if not backend_url:
        backend_url = os.environ.get('REDIS_URI', 'redis://redis:6379/0')
    
    # Handle comma-separated URLs by taking only the first one
    if backend_url and ',' in backend_url:
        backend_url = backend_url.split(',')[0]
        logger.info(f"Found comma-separated URLs, using only the first: {backend_url}")
    
    logger.info(f"Using result backend URL: {backend_url}")
    return backend_url

# Create Celery app
celery_app = Celery('multiagent_llm')

# Define exchanges
workflows_exchange = Exchange('workflows', type='direct')
agents_exchange = Exchange('agents', type='direct')
updates_exchange = Exchange('updates', type='direct')

# Configuration based on best practices from Celery documentation
# https://docs.celeryq.dev/en/stable/userguide/configuration.html
celery_app.conf.update(
    # Connection settings with robust retry policy
    broker_url=get_broker_url(),
    result_backend=get_backend_url(),
    
    # Redis visibility timeout (in seconds)
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
    task_soft_time_limit=3500,  # 58.3 minutes soft limit (warning)
    
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
    
    # Task routing
    task_routes={
        'multiagent.app.worker.tasks.execute_workflow_task': {'queue': 'workflows'},
        'multiagent.app.worker.tasks.execute_agent_task': {'queue': 'agents'},
        'multiagent.app.worker.tasks.update_progress': {'queue': 'updates'},
    },
    
    # Queues
    task_queues=(
        Queue('workflows', exchange=workflows_exchange, routing_key='workflows'),
        Queue('agents', exchange=agents_exchange, routing_key='agents'),
        Queue('updates', exchange=updates_exchange, routing_key='updates'),
    ),
    
    # Default queue
    task_default_queue='workflows',
)

def test_redis_connection():
    """
    Test Redis connection with improved error handling
    """
    from redis import Redis
    from redis.exceptions import ConnectionError, TimeoutError
    
    try:
        redis_url = get_broker_url()
        logger.info(f"Testing Redis connection to {redis_url}")
        
        # Extract host and port from Redis URL
        # redis://hostname:port/db
        parts = redis_url.split('://')[-1].split(':')
        host = parts[0]
        port = int(parts[1].split('/')[0])
        
        # Try to connect directly
        client = Redis(host=host, port=port, socket_timeout=5, socket_connect_timeout=5)
        response = client.ping()
        
        if response:
            logger.info(f"Successfully connected to Redis at {host}:{port}")
            return True
        else:
            logger.error(f"Redis connection test failed: Ping returned {response}")
            return False
            
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Redis connection error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error testing Redis connection: {str(e)}")
        return False

@celeryd_after_setup.connect
def configure_worker(sender, instance, **kwargs):
    """
    Additional worker configuration after setup
    """
    logger.info(f"Worker {sender} ready. Configuring additional settings...")

@worker_init.connect
def init_worker(**kwargs):
    """
    Initialize worker with connection checks
    """
    logger.info("Initializing Celery worker")
    
    # Log connection info
    logger.info(f"Broker URL: {celery_app.conf.broker_url}")
    logger.info(f"Result backend: {celery_app.conf.result_backend}")
    
    # Log hostname
    logger.info(f"Hostname: {socket.gethostname()}")
    
    # Test Redis connection
    test_redis_connection()
    
    # Initialize database connection if needed
    try:
        from multiagent.app.db.session import init_db
        init_db()
        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

@worker_ready.connect
def worker_ready(**kwargs):
    """
    Logs when the worker is ready to accept tasks
    """
    logger.info("Celery worker is ready to accept tasks")

@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """
    Pre-task execution hook
    """
    logger.info(f"Starting task: {task.name} (ID: {task_id})")

@task_postrun.connect
def task_postrun_handler(task_id, task, retval, state, *args, **kwargs):
    """
    Post-task execution hook
    """
    logger.info(f"Task completed: {task.name} (ID: {task_id}, State: {state})")

# Ensure the app is imported
__all__ = ['celery_app']