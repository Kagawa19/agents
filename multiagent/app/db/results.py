"""
CRUD operations for the Result model.
Contains functions for creating, reading, updating, and deleting query results.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import json
from sqlalchemy import text

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from multiagent.app.db.base import CRUDBase
from multiagent.app.db.models import Result, AgentExecution


logger = logging.getLogger(__name__)


class CRUDResult(CRUDBase[Result, dict, dict]):
    """
    CRUD operations for the Result model.
    Extends the base CRUD class with Result-specific operations.
    """
    
    def get_by_task_id(self, db: Session, *, task_id: str) -> Optional[Result]:
        """
        Get a result by task ID with robust error handling.
        
        Args:
            db: SQLAlchemy database session
            task_id: Unique task identifier
        
        Returns:
            Result object or None if not found
        """
        try:
            logger.debug(f"Attempting to retrieve result for task_id: {task_id}")
            
            # Use a more robust query with explicit error handling
            query = db.query(Result).filter(Result.task_id == task_id)
            
            # Execute the query with explicit error handling
            try:
                result = query.first()
            except Exception as query_error:
                logger.error(f"Error querying result for task_id {task_id}: {query_error}")
                return None
            
            logger.debug(f"Result found for task_id {task_id}: {result is not None}")
            return result
        
        except Exception as e:
            logger.error(f"Unexpected error in get_by_task_id for task_id {task_id}: {e}")
            return None
    
    def get_by_query(self, db: Session, *, query: str, limit: int = 10) -> List[Result]:
        """
        Get results by query text.
        """
        return db.query(Result).filter(Result.query.ilike(f"%{query}%")).limit(limit).all()
    
    def get_by_user(self, db: Session, *, user_id: str, skip: int = 0, limit: int = 100) -> List[Result]:
        """
        Get results for a specific user.
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
        status: str = "completed",
        celery_task_id: Optional[str] = None
    ) -> Result:
        """
        Save a query result with enhanced error handling and verification.
        """
        try:
            # Set transaction isolation level to ensure changes are visible
            db.execute(text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED"))

            
            # Debug: Verify database connectivity
            db_check = db.execute(text("SELECT 1")).scalar()
            logger.debug(f"Database connectivity check: {db_check == 1}")
            
            # Check if a result with this task_id already exists
            db_result = self.get_by_task_id(db, task_id=task_id)
            
            # Serialize the result to ensure it's JSON-compatible
            serialized_result = None
            if result is not None:
                try:
                    # Test JSON serialization
                    json.dumps(result)
                    serialized_result = result
                except (TypeError, ValueError) as json_error:
                    logger.warning(f"Result not JSON serializable, converting to string: {json_error}")
                    serialized_result = {"data": str(result), "serialization_error": str(json_error)}
            
            current_time = datetime.utcnow()
            
            if db_result:
                logger.info(f"Updating existing result for task_id: {task_id}")
                
                # Prepare update data
                update_data = {
                    "status": status,
                    "updated_at": current_time
                }
                
                if serialized_result is not None:
                    update_data["result"] = serialized_result
                
                if celery_task_id is not None:
                    update_data["celery_task_id"] = celery_task_id
                
                # Update the existing record
                updated_obj = self.update(db, db_obj=db_result, obj_in=update_data)
                
                # Explicitly commit the changes
                db.commit()
                
                # Verify the update
                verification = db.query(Result).filter(Result.id == db_result.id).first()
                if verification:
                    logger.info(f"Result update verified for task_id {task_id}, status is now: {verification.status}")
                else:
                    logger.error(f"Failed to verify result update for task_id: {task_id}")
                
                return updated_obj
            else:
                logger.info(f"Creating new result for task_id: {task_id}")
                
                # Prepare new result data
                result_data = {
                    "task_id": task_id,
                    "query": query,
                    "workflow": workflow,
                    "status": status,
                    "user_id": user_id,
                    "created_at": current_time,
                    "updated_at": current_time,
                }
                
                if serialized_result is not None:
                    result_data["result"] = serialized_result
                
                if celery_task_id is not None:
                    result_data["celery_task_id"] = celery_task_id
                
                # Create a new result object
                db_result = Result(**result_data)
                
                # Add to session
                db.add(db_result)
                
                # Explicitly commit to ensure it's saved
                db.commit()
                db.refresh(db_result)
                
                # Verify the insert
                verification = db.query(Result).filter(Result.task_id == task_id).first()
                if verification:
                    logger.info(f"New result creation verified for task_id: {task_id}, id: {verification.id}")
                else:
                    logger.error(f"Failed to verify result creation for task_id: {task_id}")
                
                return db_result
                
        except Exception as e:
            # Log detailed error information
            logger.error(f"Error in save_result for task_id {task_id}: {str(e)}", exc_info=True)
            
            # Rollback transaction on error
            db.rollback()
            
            # Re-raise to ensure caller knows about the error
            raise
    
    def update_status(self, db: Session, *, task_id: str, status: str) -> Optional[Result]:
        """
        Update the status of a result with enhanced error handling.
        """
        try:
            logger.debug(f"Updating status for task_id {task_id} to {status}")
            
            db_result = self.get_by_task_id(db, task_id=task_id)
            if not db_result:
                logger.warning(f"Cannot update status: No result found for task_id {task_id}")
                return None
            
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            updated_obj = self.update(db, db_obj=db_result, obj_in=update_data)
            
            # Explicitly commit
            db.commit()
            
            # Verify update
            verification = db.query(Result).filter(Result.id == db_result.id).first()
            if verification and verification.status == status:
                logger.info(f"Status update verified for task_id {task_id}: {status}")
            else:
                logger.warning(f"Status update could not be verified for task_id {task_id}")
            
            return updated_obj
            
        except Exception as e:
            logger.error(f"Error updating status for task_id {task_id}: {str(e)}", exc_info=True)
            db.rollback()
            raise
    
    def get_recent_results(self, db: Session, *, limit: int = 10) -> List[Result]:
        """
        Get recent results.
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
        Save an agent execution record with enhanced error handling.
        """
        try:
            logger.debug(f"Saving agent execution for result_id: {result_id}, agent_id: {agent_id}")
            
            # Serialize input/output data
            serialized_input = None
            if input_data is not None:
                try:
                    json.dumps(input_data)
                    serialized_input = input_data
                except (TypeError, ValueError):
                    serialized_input = {"data": str(input_data)}
            
            serialized_output = None
            if output_data is not None:
                try:
                    json.dumps(output_data)
                    serialized_output = output_data
                except (TypeError, ValueError):
                    serialized_output = {"data": str(output_data)}
            
            # Prepare execution data
            execution_data = {
                "result_id": result_id,
                "agent_id": agent_id,
                "status": status,
                "started_at": datetime.utcnow()
            }
            
            if serialized_input is not None:
                execution_data["input_data"] = serialized_input
            
            if serialized_output is not None:
                execution_data["output_data"] = serialized_output
            
            if error is not None:
                execution_data["error"] = error
            
            if execution_time is not None:
                execution_data["execution_time"] = execution_time
                execution_data["completed_at"] = datetime.utcnow()
            
            # Create and save execution record
            db_execution = AgentExecution(**execution_data)
            db.add(db_execution)
            db.commit()
            db.refresh(db_execution)
            
            # Verify save
            verification = db.query(AgentExecution).filter(AgentExecution.id == db_execution.id).first()
            if verification:
                logger.info(f"Agent execution saved successfully, id: {verification.id}")
            else:
                logger.warning("Failed to verify agent execution save")
            
            return db_execution
            
        except Exception as e:
            logger.error(f"Error saving agent execution: {str(e)}", exc_info=True)
            db.rollback()
            raise
    
    def get_statistics(self, db: Session) -> Dict[str, Any]:
        """
        Get statistics about results.
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