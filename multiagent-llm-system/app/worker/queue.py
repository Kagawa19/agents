"""
Interface for task queue management.
Provides a simplified interface for submitting tasks and checking their status.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from celery.result import AsyncResult

from app.worker.celery_app import celery_app


logger = logging.getLogger(__name__)


class TaskQueue:
    """
    Interface for interacting with the Celery task queue.
    Provides methods for submitting tasks and checking their status.
    """
    
    def submit_task(
        self,
        task_name: str,
        *args: Any,
        **kwargs: Any
    ) -> str:
        """
        Submit a task to the queue.
        
        Args:
            task_name: Name of the task to execute
            *args: Positional arguments for the task
            **kwargs: Keyword arguments for the task
        
        Returns:
            Task ID
        """
        # Submit the task
        task = celery_app.send_task(task_name, args=args, kwargs=kwargs)
        logger.info(f"Submitted task {task_name} (ID: {task.id})")
        return task.id
    
    def get_task_status(self, task_id: str) -> str:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Task status (PENDING, STARTED, SUCCESS, FAILURE, REVOKED)
        """
        result = AsyncResult(task_id, app=celery_app)
        return result.state
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get the result of a task.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Task result or None if not ready
        """
        result = AsyncResult(task_id, app=celery_app)
        
        if result.ready():
            try:
                return result.get()
            except Exception as e:
                logger.error(f"Error retrieving task result: {str(e)}")
                return None
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task
        
        Returns:
            True if the task was successfully revoked, False otherwise
        """
        try:
            celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"Revoked task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error revoking task {task_id}: {str(e)}")
            return False
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get a list of active tasks.
        
        Returns:
            List of active tasks with their status
        """
        try:
            # Get active tasks from all workers
            inspect = celery_app.control.inspect()
            active = inspect.active() or {}
            scheduled = inspect.scheduled() or {}
            reserved = inspect.reserved() or {}
            
            # Combine all tasks
            all_tasks = []
            
            # Add active tasks
            for worker, tasks in active.items():
                for task in tasks:
                    task_info = {
                        "id": task["id"],
                        "name": task["name"],
                        "args": task["args"],
                        "kwargs": task["kwargs"],
                        "state": "ACTIVE",
                        "worker": worker
                    }
                    all_tasks.append(task_info)
            
            # Add scheduled tasks
            for worker, tasks in scheduled.items():
                for task in tasks:
                    task_info = {
                        "id": task["request"]["id"],
                        "name": task["request"]["name"],
                        "args": task["request"]["args"],
                        "kwargs": task["request"]["kwargs"],
                        "state": "SCHEDULED",
                        "worker": worker
                    }
                    all_tasks.append(task_info)
            
            # Add reserved tasks
            for worker, tasks in reserved.items():
                for task in tasks:
                    task_info = {
                        "id": task["id"],
                        "name": task["name"],
                        "args": task["args"],
                        "kwargs": task["kwargs"],
                        "state": "RESERVED",
                        "worker": worker
                    }
                    all_tasks.append(task_info)
            
            return all_tasks
        
        except Exception as e:
            logger.error(f"Error getting active tasks: {str(e)}")
            return []
    
    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a task.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Task information
        """
        result = AsyncResult(task_id, app=celery_app)
        
        task_info = {
            "id": task_id,
            "state": result.state,
            "ready": result.ready()
        }
        
        # Add result if ready
        if result.ready():
            try:
                task_info["result"] = result.get()
            except Exception as e:
                task_info["error"] = str(e)
        
        # Add info if running
        if result.state == "STARTED":
            task_info["info"] = result.info
        
        return task_info
    
    def purge_queue(self) -> int:
        """
        Purge all pending tasks from the queue.
        
        Returns:
            Number of tasks purged
        """
        try:
            return celery_app.control.purge()
        except Exception as e:
            logger.error(f"Error purging task queue: {str(e)}")
            return 0