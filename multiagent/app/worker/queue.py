"""
Interface for task queue management.
Provides a simplified interface for submitting tasks and checking their status.
"""

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Union
import uuid
from celery.result import AsyncResult

from multiagent.app.worker.celery_app import celery_app
from multiagent.app.worker.tasks import update_progress

logger = logging.getLogger(__name__)

# Configure more verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class TaskQueue:
    """
    Interface for interacting with the Celery task queue.
    Provides methods for submitting tasks and checking their status.
    """
    
    def __init__(self):
        """Initialize the task queue and verify connections."""
        logger.info("TaskQueue: Initializing task queue interface")
        
        # Verify Celery app
        if celery_app:
            logger.info(f"TaskQueue: Celery app initialized with broker: {celery_app.conf.broker_url}")
            logger.info(f"TaskQueue: Result backend: {celery_app.conf.result_backend}")
        else:
            logger.warning("TaskQueue: WARNING - Celery app not properly initialized")
            
        # Test connection on initialization
        self._test_celery_connection()

    def _test_celery_connection(self, max_retries=3, retry_delay=2):
        """
        Test the Celery connection to both broker and backend.
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
        """
        from celery.exceptions import OperationalError
        
        for attempt in range(max_retries):
            try:
                # Try to ping the broker
                logger.info(f"TaskQueue: Testing Celery broker connection (attempt {attempt+1}/{max_retries})")
                celery_app.control.ping(timeout=5)
                logger.info("TaskQueue: Successfully connected to Celery broker")
                return
            except OperationalError as e:
                logger.warning(f"TaskQueue: Failed to connect to Celery broker: {str(e)}")
            except Exception as e:
                logger.warning(f"TaskQueue: Unexpected error connecting to Celery broker: {str(e)}")
            
            # Wait before retry
            if attempt < max_retries - 1:
                logger.info(f"TaskQueue: Waiting {retry_delay}s before next connection attempt")
                time.sleep(retry_delay)
        
        logger.error(f"TaskQueue: Failed to connect to Celery broker after {max_retries} attempts")

    async def submit_task(self, task_name: str, *args, **kwargs):
        """
        Submit a task to Celery with retry logic.
        
        Args:
            task_name: Name of the Celery task
            *args: Positional arguments for the task
            **kwargs: Keyword arguments for the task
            
        Returns:
            AsyncResult object or None if submission failed
        """
        from celery.exceptions import OperationalError
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"TaskQueue: Submitting task {task_name} (attempt {attempt+1}/{max_retries})")
                
                # Use send_task as recommended in Celery docs for reliable task submission
                # https://docs.celeryq.dev/en/stable/reference/celery.app.task.html#celery.app.task.Task.send_task
                task = celery_app.send_task(
                    task_name,
                    args=args,
                    kwargs=kwargs,
                    retry=True,
                    retry_policy={
                        'max_retries': 3,
                        'interval_start': 0,
                        'interval_step': 0.5,
                        'interval_max': 3,
                    }
                )
                
                logger.info(f"TaskQueue: Task {task_name} submitted successfully. Task ID: {task.id}")
                return task
                
            except OperationalError as e:
                logger.warning(f"TaskQueue: OperationalError submitting task {task_name}: {str(e)}")
            except Exception as e:
                logger.error(f"TaskQueue: Error submitting task {task_name}: {str(e)}")
                logger.error(f"TaskQueue: Exception type: {type(e)}")
            
            # Wait before retry
            if attempt < max_retries - 1:
                logger.info(f"TaskQueue: Waiting {retry_delay}s before retry")
                time.sleep(retry_delay)
        
        logger.error(f"TaskQueue: Failed to submit task {task_name} after {max_retries} attempts")
        raise Exception(f"Failed to submit task {task_name} after {max_retries} attempts")

    async def get_celery_task_id(self, task_id: str) -> Optional[str]:
        """
        Get the Celery task ID for a given task UUID.
        
        Args:
            task_id: Your UUID for the task
            
        Returns:
            The Celery task ID or None if not found
        """
        # Import inside function to avoid circular imports
        from multiagent.app.db.session import SessionLocal
        from multiagent.app.db.models import Result
        
        try:
            with SessionLocal() as db:
                result = db.query(Result).filter(Result.task_id == task_id).first()
                if result and result.celery_task_id:
                    logger.debug(f"TaskQueue: Found Celery task ID {result.celery_task_id} for task {task_id}")
                    return result.celery_task_id
                else:
                    logger.warning(f"TaskQueue: No Celery task ID found for task {task_id}")
        except Exception as e:
            logger.error(f"TaskQueue: Error getting Celery task ID for {task_id}: {str(e)}")
        
        return None
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task with proper error handling.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Dictionary containing task status information
        """
        logger.info(f"TaskQueue: Getting status for task {task_id}")
        
        try:
            # Get the Celery task ID
            celery_task_id = await self.get_celery_task_id(task_id)
            
            if not celery_task_id:
                logger.warning(f"TaskQueue: No Celery task ID found for {task_id}")
                return {
                    "task_id": task_id,
                    "status": "unknown",
                    "error": "No Celery task ID mapping found",
                    "progress": 0,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                }
            
            # Now use the Celery task ID to get the status
            result = AsyncResult(celery_task_id, app=celery_app)
            
            # Map Celery status to application status
            status_mapping = {
                "PENDING": "pending",
                "STARTED": "processing",
                "SUCCESS": "completed",
                "FAILURE": "failed",
                "REVOKED": "cancelled",
                "RETRY": "retrying"
            }
            
            celery_status = result.state
            app_status = status_mapping.get(celery_status, "unknown")
            
            # Determine progress based on status
            progress = 0
            if celery_status == "PENDING":
                progress = 0
            elif celery_status == "STARTED":
                progress = 50
            elif celery_status == "SUCCESS":
                progress = 100
            elif celery_status == "RETRY":
                progress = 25
            
            # Get additional info if available
            info = {}
            if hasattr(result, 'info') and result.info:
                if isinstance(result.info, dict):
                    info = result.info
            
            # Construct status response
            status_data = {
                "task_id": task_id,
                "status": app_status,
                "celery_status": celery_status,
                "progress": progress,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }
            
            # Add info if available
            if info:
                status_data.update(info)
            
            logger.info(f"TaskQueue: Status for task {task_id} (Celery ID: {celery_task_id}): {app_status} ({progress}%)")
            return status_data
            
        except Exception as e:
            logger.error(f"TaskQueue: Error getting status for task {task_id}: {str(e)}")
            return {
                "task_id": task_id,
                "status": "unknown",
                "error": str(e),
                "progress": 0,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }
    
    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get the result of a task with proper error handling.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Dictionary containing task result or error information
        """
        logger.info(f"TaskQueue: Getting result for task {task_id}")
        
        try:
            # Get the Celery task ID
            celery_task_id = await self.get_celery_task_id(task_id)
            
            if not celery_task_id:
                logger.warning(f"TaskQueue: No Celery task ID found for {task_id}")
                return {
                    "task_id": task_id,
                    "status": "unknown",
                    "error": "No Celery task ID mapping found",
                    "progress": 0,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                }
            
            # Now use the Celery task ID to get the result
            result = AsyncResult(celery_task_id, app=celery_app)
            
            # Check if task is ready
            if not result.ready():
                logger.info(f"TaskQueue: Task {task_id} (Celery ID: {celery_task_id}) is not ready yet")
                status = await self.get_task_status(task_id)
                return status
            
            # Task is ready - get the result or exception
            if result.successful():
                logger.info(f"TaskQueue: Task {task_id} (Celery ID: {celery_task_id}) completed successfully")
                task_result = result.get()
                
                # Ensure result is a dictionary
                if not isinstance(task_result, dict):
                    task_result = {"value": task_result}
                
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": task_result,
                    "progress": 100,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                }
            else:
                # Task failed
                logger.warning(f"TaskQueue: Task {task_id} (Celery ID: {celery_task_id}) failed")
                error = str(result.get(propagate=False))
                
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": error,
                    "progress": 100,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                }
                
        except Exception as e:
            logger.error(f"TaskQueue: Error getting result for task {task_id}: {str(e)}")
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "progress": 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
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
            logger.info(f"TaskQueue: Updating progress for task {task_id}: {status} ({progress}%)")
            
            # Call the update_progress task with retry
            update_progress.apply_async(
                kwargs={
                    "task_id": task_id,
                    "status": status,
                    "progress": progress,
                    "current_step": current_step,
                    "result": result,
                    "error": error
                },
                retry=True,
                retry_policy={
                    'max_retries': 3,
                    'interval_start': 0,
                    'interval_step': 0.5,
                    'interval_max': 3,
                }
            )
            
            logger.debug(f"TaskQueue: Successfully updated progress for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"TaskQueue: Error updating progress for task {task_id}: {str(e)}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task
        
        Returns:
            True if the task was successfully revoked, False otherwise
        """
        try:
            logger.info(f"TaskQueue: Attempting to cancel task {task_id}")
            
            # Revoke the task and terminate it if it's running
            celery_app.control.revoke(task_id, terminate=True)
            
            logger.info(f"TaskQueue: Successfully revoked task {task_id}")
            return True
        except Exception as e:
            logger.error(f"TaskQueue: Error revoking task {task_id}: {str(e)}")
            return False