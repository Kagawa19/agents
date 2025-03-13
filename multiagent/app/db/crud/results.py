"""
CRUD operations for the Result model.
Contains functions for creating, reading, updating, and deleting query results.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.db.crud.base import CRUDBase
from app.db.models import Result, AgentExecution


logger = logging.getLogger(__name__)


class CRUDResult(CRUDBase[Result, dict, dict]):
    """
    CRUD operations for the Result model.
    Extends the base CRUD class with Result-specific operations.
    """
    
    def get_by_task_id(self, db: Session, *, task_id: str) -> Optional[Result]:
        """
        Get a result by task ID.
        
        Args:
            db: Database session
            task_id: Celery task ID
            
        Returns:
            The found result or None
        """
        return db.query(Result).filter(Result.task_id == task_id).first()
    
    def get_by_query(self, db: Session, *, query: str, limit: int = 10) -> List[Result]:
        """
        Get results by query text.
        
        Args:
            db: Database session
            query: Query text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching results
        """
        return db.query(Result).filter(Result.query.ilike(f"%{query}%")).limit(limit).all()
    
    def get_by_user(self, db: Session, *, user_id: str, skip: int = 0, limit: int = 100) -> List[Result]:
        """
        Get results for a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of user's results
        """
        return db.query(Result).filter(Result.user_id == user_id).offset(skip).limit(limit).all()
    
    def save_result(
        self, 
        db: Session, 
        *, 
        task_id: str, 
        query: str, 
        workflow: str, 
        result: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        status: str = "completed"
    ) -> Result:
        """
        Save a query result.
        Creates a new Result record or updates an existing one.
        
        Args:
            db: Database session
            task_id: Celery task ID
            query: Original query text
            workflow: Workflow ID that produced the result
            result: Result data (optional)
            user_id: User ID (optional)
            status: Result status
            
        Returns:
            The created or updated Result
        """
        # Check if result already exists
        db_result = self.get_by_task_id(db, task_id=task_id)
        
        if db_result:
            # Update existing result
            update_data = {
                "result": result,
                "status": status,
                "updated_at": datetime.utcnow()
            }
            if result is not None:
                update_data["result"] = result
            
            return self.update(db, db_obj=db_result, obj_in=update_data)
        else:
            # Create new result
            result_data = {
                "task_id": task_id,
                "query": query,
                "workflow": workflow,
                "status": status,
                "user_id": user_id
            }
            if result is not None:
                result_data["result"] = result
            
            db_result = Result(**result_data)
            db.add(db_result)
            db.commit()
            db.refresh(db_result)
            return db_result
    
    def update_status(self, db: Session, *, task_id: str, status: str) -> Optional[Result]:
        """
        Update the status of a result.
        
        Args:
            db: Database session
            task_id: Celery task ID
            status: New status
            
        Returns:
            The updated Result or None if not found
        """
        db_result = self.get_by_task_id(db, task_id=task_id)
        if not db_result:
            return None
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        return self.update(db, db_obj=db_result, obj_in=update_data)
    
    def get_recent_results(self, db: Session, *, limit: int = 10) -> List[Result]:
        """
        Get recent results.
        
        Args:
            db: Database session
            limit: Maximum number of results to return
            
        Returns:
            List of recent results
        """
        return db.query(Result).order_by(desc(Result.created_at)).limit(limit).all()
    
    def save_agent_execution(
        self,
        db: Session,
        *,
        result_id: int,
        agent_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        status: str = "completed",
        execution_time: Optional[float] = None
    ) -> AgentExecution:
        """
        Save an agent execution record.
        
        Args:
            db: Database session
            result_id: ID of the associated Result
            agent_id: ID of the executed agent
            input_data: Input data provided to the agent
            output_data: Output data produced by the agent
            error: Error message if execution failed
            status: Execution status
            execution_time: Execution time in seconds
            
        Returns:
            The created AgentExecution record
        """
        execution_data = {
            "result_id": result_id,
            "agent_id": agent_id,
            "status": status,
            "started_at": datetime.utcnow()
        }
        
        if input_data is not None:
            execution_data["input_data"] = input_data
        
        if output_data is not None:
            execution_data["output_data"] = output_data
        
        if error is not None:
            execution_data["error"] = error
        
        if execution_time is not None:
            execution_data["execution_time"] = execution_time
            execution_data["completed_at"] = datetime.utcnow()
        
        db_execution = AgentExecution(**execution_data)
        db.add(db_execution)
        db.commit()
        db.refresh(db_execution)
        return db_execution
    
    def get_statistics(self, db: Session) -> Dict[str, Any]:
        """
        Get statistics about results.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary of statistics
        """
        total_results = db.query(func.count(Result.id)).scalar() or 0
        completed_results = db.query(func.count(Result.id)).filter(Result.status == "completed").scalar() or 0
        failed_results = db.query(func.count(Result.id)).filter(Result.status == "failed").scalar() or 0
        
        return {
            "total_results": total_results,
            "completed_results": completed_results,
            "failed_results": failed_results,
            "success_rate": (completed_results / total_results) if total_results > 0 else 0
        }


# Create a CRUD instance for the Result model
crud_result = CRUDResult(Result)