# api/query.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import uuid
import logging
from datetime import datetime
import traceback
import time
from sqlalchemy import text
import os
import json
import pprint
from dotenv import load_dotenv

# Import task queue at the top level is fine
from multiagent.app.worker.queue import TaskQueue
from multiagent.app.monitoring.tracer import LangfuseTracer

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [query.py] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Initialize the task queue
task_queue = TaskQueue()

class QueryRequest(BaseModel):
    """Validates user query requests."""
    query: str
    user_id: Optional[str] = None
    workflow_type: str = "research"  # Default to research workflow 
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority: Optional[int] = None
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query string."""
        if not v or len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        if len(v) > 1000:
            raise ValueError("Query must be less than 1000 characters")
        return v.strip()
    
    def validate(self) -> bool:
        """
        Validates query parameters.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            self.validate_query(self.query)
            return True
        except ValueError:
            return False

class QueryStatus(BaseModel):
    """Represents the status of a query."""
    task_id: str
    status: str
    progress: Optional[float] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
class QueryResponse(BaseModel):
    """Represents the response to a query."""
    task_id: str
    result: Dict[str, Any]
    status: str
    execution_time: Optional[float] = None
    created_at: Optional[datetime] = None

# Initialize tracer with environment variables
tracer = LangfuseTracer(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"), 
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://api.langfuse.com")
)

async def submit_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Receives user query, validates it, and submits task to the queue.
    
    Args:
        query_data: Query parameters including the query text
        
    Returns:
        Dict containing task_id and initial status
    """
    # Import here to avoid circular imports
    from multiagent.app.worker.tasks import execute_workflow_task
    from multiagent.app.db.models import Result
    from multiagent.app.db.results import crud_result
    from sqlalchemy.orm import Session
    from multiagent.app.db.session import SessionLocal
    
    # Debug point 1: Log incoming query data
    logger.debug(f"🔍 Received query data: {json.dumps(query_data, indent=2)}")
    print(f"\n=== QUERY SUBMISSION STARTED ===\nQuery Data: {pprint.pformat(query_data)}\n")
    
    # Create QueryRequest for validation
    request = QueryRequest(**query_data)
    
    # Check if request is valid
    if not request.validate():
        logger.error(f"❌ Invalid query parameters: {query_data}")
        print(f"ERROR: Invalid query parameters: {query_data}")
        raise ValueError("Invalid query parameters")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Debug point 2: Log task ID
    logger.debug(f"🆔 Generated task_id: {task_id}")
    print(f"Generated task_id: {task_id}")
    
    # Create trace for monitoring
    try:
        trace_id_obj = tracer.create_trace(
            task_id=task_id,
            name="query_execution",
            metadata={
                "query": request.query, 
                "workflow_type": request.workflow_type,
                "user_id": request.user_id
            }
        )
        
        # Debug point 3: Log trace creation
        logger.debug(f"📊 Created trace: {trace_id_obj}")
        print(f"Created trace with ID: {trace_id_obj}")
        
    except Exception as trace_error:
        logger.error(f"❌ Error creating trace: {trace_error}")
        print(f"ERROR: Failed to create trace: {trace_error}")
        # Continue without trace if it fails
        trace_id_obj = {"id": "error-creating-trace"}
    
    logger.info(f"📝 Submitting query: {request.query[:50]}... (task_id: {task_id})")
    print(f"Submitting query: {request.query[:50]}... (task_id: {task_id})")
    
    # Extract just the trace ID string instead of using the whole dictionary
    trace_id_str = trace_id_obj.get('id') if isinstance(trace_id_obj, dict) else str(trace_id_obj)
    
    # Prepare kwargs for task submission
    task_kwargs = {
        'workflow_id': request.workflow_type,
        'input_data': {
            'task_id': task_id,
            'query': request.query,
            'user_id': request.user_id,
            'parameters': request.parameters,
            'trace_id': trace_id_str
        }
    }
    
    # Debug point 4: Log task kwargs
    logger.debug(f"🔧 Task kwargs prepared: {json.dumps(task_kwargs, indent=2)}")
    print(f"Task kwargs prepared: {pprint.pformat(task_kwargs)}")
    
    # Only add priority if it's not None
    if request.priority is not None:
        task_kwargs['input_data']['priority'] = request.priority
        logger.debug(f"⚡ Added priority: {request.priority}")
        print(f"Added priority: {request.priority}")
    
    # Create an initial database entry for the task
    try:
        with SessionLocal() as db:
            # Check if database connection is working
            db_check = db.execute(text("SELECT 1")).scalar()
            if db_check != 1:
                logger.error("❌ Database connection check failed!")
                print("ERROR: Database connection check failed!")
            else:
                logger.info("✅ Database connection verified before initial task save")
                print("Database connection verified")
                
            # Save initial task status to database
            from multiagent.app.db.results import crud_result
            initial_record = crud_result.save_result(
                db=db,
                task_id=task_id,
                query=request.query,
                workflow=request.workflow_type,
                user_id=request.user_id,
                status="submitted"
            )
            
            # Explicitly commit
            db.commit()
            
            # Log the record that was saved
            if initial_record:
                logger.info(f"💾 Initial database record created: {initial_record.id}")
                print(f"Initial database record:\nID: {initial_record.id}\nTask ID: {initial_record.task_id}\nStatus: {initial_record.status}")
            
            # Verify the save
            verification = crud_result.get_by_task_id(db=db, task_id=task_id)
            if verification:
                logger.info(f"✅ Initial task record verified in database with id: {verification.id}")
                print(f"Initial task record verified in database with id: {verification.id}")
                
                # Log the full record details
                record_dict = {
                    "id": verification.id,
                    "task_id": verification.task_id,
                    "query": verification.query[:50] + "..." if verification.query and len(verification.query) > 50 else verification.query,
                    "user_id": verification.user_id,
                    "workflow": verification.workflow,
                    "status": verification.status,
                    "created_at": verification.created_at.isoformat() if verification.created_at else None,
                    "updated_at": verification.updated_at.isoformat() if verification.updated_at else None
                }
                logger.debug(f"📋 Initial record details: {json.dumps(record_dict, indent=2)}")
                print(f"Initial record details: {pprint.pformat(record_dict)}")
            else:
                logger.warning("⚠️ Could not verify initial task record creation")
                print("WARNING: Could not verify initial task record creation")
                
    except Exception as db_error:
        logger.error(f"❌ Error creating initial database record: {db_error}", exc_info=True)
        print(f"ERROR: Failed to create initial database record: {db_error}")
        # Continue even if initial save fails
    
    try:
        # Debug point 5: Log before task submission
        logger.info(f"🚀 About to submit task with: workflow_id={request.workflow_type}")
        print(f"About to submit task with: workflow_id={request.workflow_type}")
        
        # Submit the execute_workflow_task with more detailed logging
        result = execute_workflow_task.delay(**task_kwargs)
        
        # Debug point 6: Log task submission result
        logger.info(f"✅ Task submitted successfully. Celery task ID: {result.id}")
        print(f"Task submitted successfully. Celery task ID: {result.id}")
        logger.debug(f"📋 Celery task result object: {result}")
        
        # Update the database record with the Celery task ID
        try:
            with SessionLocal() as db:
                existing_record = crud_result.get_by_task_id(db=db, task_id=task_id)
                
                if existing_record:
                    # Update with Celery task ID
                    logger.info(f"🔄 Updating record with Celery task ID: {result.id}")
                    print(f"Updating record with Celery task ID: {result.id}")
                    
                    crud_result.update(
                        db=db,
                        db_obj=existing_record,
                        obj_in={"celery_task_id": result.id}
                    )
                    db.commit()
                    logger.info(f"✅ Updated record with Celery task ID: {result.id}")
                    print(f"Updated record with Celery task ID: {result.id}")
                    
                    # Verify the update
                    updated_record = crud_result.get_by_task_id(db=db, task_id=task_id)
                    if updated_record and updated_record.celery_task_id == result.id:
                        logger.info(f"✅ Verified Celery task ID update: {updated_record.celery_task_id}")
                        print(f"Verified Celery task ID update: {updated_record.celery_task_id}")
                        
                        # Log the full updated record
                        updated_dict = {
                            "id": updated_record.id,
                            "task_id": updated_record.task_id,
                            "celery_task_id": updated_record.celery_task_id,
                            "query": updated_record.query[:50] + "..." if updated_record.query and len(updated_record.query) > 50 else updated_record.query,
                            "user_id": updated_record.user_id,
                            "workflow": updated_record.workflow,
                            "status": updated_record.status,
                            "created_at": updated_record.created_at.isoformat() if updated_record.created_at else None,
                            "updated_at": updated_record.updated_at.isoformat() if updated_record.updated_at else None
                        }
                        logger.debug(f"📋 Updated record details: {json.dumps(updated_dict, indent=2)}")
                        print(f"Updated record details: {pprint.pformat(updated_dict)}")
                    else:
                        logger.warning("⚠️ Could not verify Celery task ID update")
                        print("WARNING: Could not verify Celery task ID update")
                else:
                    # Create a new record if update failed
                    logger.warning("⚠️ No existing record found, creating new record as fallback")
                    print("WARNING: No existing record found, creating new record")
                    
                    new_record = crud_result.save_result(
                        db=db,
                        task_id=task_id,
                        query=request.query,
                        workflow=request.workflow_type,
                        status="submitted",
                        user_id=request.user_id,
                        celery_task_id=result.id
                    )
                    db.commit()
                    logger.info("✅ Created new record with Celery task ID as fallback")
                    print("Created new record with Celery task ID as fallback")
                    
                    # Log the full new record
                    if new_record:
                        new_dict = {
                            "id": new_record.id,
                            "task_id": new_record.task_id,
                            "celery_task_id": new_record.celery_task_id,
                            "query": new_record.query[:50] + "..." if new_record.query and len(new_record.query) > 50 else new_record.query,
                            "user_id": new_record.user_id,
                            "workflow": new_record.workflow,
                            "status": new_record.status,
                            "created_at": new_record.created_at.isoformat() if new_record.created_at else None,
                            "updated_at": new_record.updated_at.isoformat() if new_record.updated_at else None
                        }
                        logger.debug(f"📋 New record details: {json.dumps(new_dict, indent=2)}")
                        print(f"New record details: {pprint.pformat(new_dict)}")
        except Exception as update_error:
            logger.error(f"❌ Failed to update record with Celery task ID: {update_error}", exc_info=True)
            print(f"ERROR: Failed to update record with Celery task ID: {update_error}")
        
        # Add additional debugging information in the response
        response_data = {
            "task_id": task_id,
            "celery_task_id": result.id,
            "status": "submitted",
            "submitted_at": datetime.utcnow().isoformat(),
            "workflow_type": request.workflow_type
        }
        
        logger.info(f"📤 Returning response: {json.dumps(response_data, indent=2)}")
        print(f"\n=== QUERY SUBMISSION COMPLETED ===\nResponse: {pprint.pformat(response_data)}\n")
        
        return response_data
    except Exception as e:
        # Log detailed error information with better formatting
        logger.error(f"❌ Task submission failed: {str(e)}") 
        logger.error(f"🔧 Task kwargs: {json.dumps(task_kwargs, indent=2)}")
        # Include more details about the exception for debugging
        logger.error(f"🐞 Exception type: {type(e)}")
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        print(f"\nERROR: Task submission failed: {str(e)}")
        print(f"Exception type: {type(e)}")
        
        # Try to save error to database
        try:
            with SessionLocal() as db:
                error_record = crud_result.save_result(
                    db=db,
                    task_id=task_id,
                    query=request.query,
                    workflow=request.workflow_type,
                    user_id=request.user_id,
                    status="failed",
                    result={"error": str(e)}
                )
                db.commit()
                logger.info("💾 Saved submission error to database")
                print("Saved submission error to database")
                
                # Log the error record
                if error_record:
                    error_dict = {
                        "id": error_record.id,
                        "task_id": error_record.task_id,
                        "status": error_record.status,
                        "result": error_record.result
                    }
                    logger.debug(f"📋 Error record details: {json.dumps(error_dict, indent=2)}")
                    print(f"Error record details: {pprint.pformat(error_dict)}")
        except Exception as db_error:
            logger.error(f"❌ Failed to save submission error to database: {db_error}")
            print(f"ERROR: Failed to save submission error to database: {db_error}")
            
        print(f"\n=== QUERY SUBMISSION FAILED ===\n")
        raise

async def update_task_progress(
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
    # Import here to avoid circular imports
    from multiagent.app.worker.tasks import update_progress
    from multiagent.app.db.session import SessionLocal
    from multiagent.app.db.results import crud_result
    
    logger.debug(f"🔍 Retrieving result for task: {task_id}")
    print(f"\n=== RETRIEVING TASK RESULT ===\nTask ID: {task_id}")
    
    try:
        # First check the database for the result
        try:
            with SessionLocal() as db:
                db_record = crud_result.get_by_task_id(db=db, task_id=task_id)
                
                if db_record and db_record.status == "completed" and db_record.result:
                    logger.info(f"✅ Found completed result in database for task {task_id}")
                    print(f"Found completed result in database for task {task_id}")
                    
                    # Log the full record details
                    record_dict = {
                        "id": db_record.id,
                        "task_id": db_record.task_id,
                        "celery_task_id": db_record.celery_task_id,
                        "status": db_record.status,
                        "result": str(db_record.result)[:200] + "..." if db_record.result and len(str(db_record.result)) > 200 else db_record.result,
                        "created_at": db_record.created_at.isoformat() if db_record.created_at else None,
                        "updated_at": db_record.updated_at.isoformat() if db_record.updated_at else None
                    }
                    logger.debug(f"📋 Completed record details: {json.dumps(record_dict, indent=2)}")
                    print(f"Completed record details: {pprint.pformat(record_dict)}")
                    
                    # Calculate execution time if not present
                    execution_time = None
                    if db_record.result and isinstance(db_record.result, dict):
                        execution_time = db_record.result.get("execution_time")
                        logger.debug(f"⏱️ Execution time from result: {execution_time}")
                        print(f"Execution time from result: {execution_time}")
                    
                    # Format as QueryResponse
                    response_dict = QueryResponse(
                        task_id=task_id,
                        result=db_record.result,
                        status="completed",
                        execution_time=execution_time,
                        created_at=db_record.created_at
                    ).dict()
                    
                    logger.info(f"📤 Returning completed result from database")
                    print(f"Returning completed result from database")
                    print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                    return response_dict
                    
                elif db_record and db_record.status == "failed":
                    logger.info(f"⚠️ Task {task_id} failed, returning error result")
                    print(f"Task {task_id} failed, returning error result")
                    
                    # Log the error record
                    error_record = {
                        "id": db_record.id,
                        "task_id": db_record.task_id,
                        "status": db_record.status,
                        "error": db_record.result.get("error") if db_record.result and isinstance(db_record.result, dict) else "Unknown error"
                    }
                    logger.debug(f"📋 Failed task record: {json.dumps(error_record, indent=2)}")
                    print(f"Failed task record: {pprint.pformat(error_record)}")
                    
                    error_result = {
                        "task_id": task_id,
                        "status": "failed",
                        "error": "Task execution failed"
                    }
                    
                    if db_record.result and isinstance(db_record.result, dict):
                        error_result["error_details"] = db_record.result
                    
                    logger.info(f"📤 Returning error result from database")
                    print(f"Returning error result from database")
                    print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                    return error_result
                
                elif db_record:
                    logger.info(f"🔄 Task {task_id} is still in progress with status: {db_record.status}")
                    print(f"Task {task_id} is still in progress with status: {db_record.status}")
                    
                    # Log the in-progress record
                    progress_record = {
                        "id": db_record.id,
                        "task_id": db_record.task_id,
                        "status": db_record.status,
                        "created_at": db_record.created_at.isoformat() if db_record.created_at else None,
                        "updated_at": db_record.updated_at.isoformat() if db_record.updated_at else None
                    }
                    logger.debug(f"📋 In-progress task record: {json.dumps(progress_record, indent=2)}")
                    print(f"In-progress task record: {pprint.pformat(progress_record)}")
                    
                    progress_dict = {
                        "task_id": task_id,
                        "status": db_record.status,
                        "progress": 100 if db_record.status == "completed" else (
                            0 if db_record.status == "failed" else 50
                        )
                    }
                    
                    logger.info(f"📤 Returning in-progress status from database")
                    print(f"Returning in-progress status from database")
                    print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                    return progress_dict
        except Exception as db_error:
            logger.error(f"❌ Error getting result from database: {db_error}", exc_info=True)
            print(f"ERROR: Error getting result from database: {db_error}")
        
        # Get result from task queue as a fallback
        logger.debug(f"🔍 Checking task queue for result of task: {task_id}")
        print(f"Checking task queue for result of task: {task_id}")
        result_data = await task_queue.get_task_result(task_id)
        
        logger.debug(f"📊 Retrieved result data from queue: {json.dumps(str(result_data)[:200], indent=2) + '...' if result_data and len(str(result_data)) > 200 else result_data}")
        print(f"Retrieved result data from queue: {str(result_data)[:200] + '...' if result_data and len(str(result_data)) > 200 else result_data}")
        
        if not result_data:
            logger.warning(f"⚠️ Result for task {task_id} not found in queue")
            print(f"WARNING: Result for task {task_id} not found in queue")
            
            # Check database again in case we missed it
            try:
                with SessionLocal() as db:
                    db_record = crud_result.get_by_task_id(db=db, task_id=task_id)
                    
                    if db_record:
                        logger.info(f"✅ Found task {task_id} in database with status: {db_record.status}")
                        print(f"Found task {task_id} in database with status: {db_record.status}")
                        
                        if db_record.status == "completed" and db_record.result:
                            # Format as QueryResponse for completed tasks
                            fallback_response = QueryResponse(
                                task_id=task_id,
                                result=db_record.result,
                                status="completed",
                                execution_time=None,
                                created_at=db_record.created_at
                            ).dict()
                            
                            logger.info(f"📤 Returning completed result from database fallback")
                            print(f"Returning completed result from database fallback")
                            print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                            return fallback_response
                        else:
                            # Return status for in-progress tasks
                            fallback_status = {
                                "task_id": task_id,
                                "status": db_record.status,
                                "progress": 100 if db_record.status == "completed" else (
                                    0 if db_record.status == "failed" else 50
                                )
                            }
                            
                            logger.info(f"📤 Returning status from database fallback")
                            print(f"Returning status from database fallback")
                            print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                            return fallback_status
                    else:
                        logger.error(f"❌ Result for task {task_id} not found in database")
                        print(f"ERROR: Result for task {task_id} not found in database")
                        raise ValueError(f"Result for task {task_id} not found")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback to database failed: {fallback_error}", exc_info=True)
                print(f"ERROR: Fallback to database failed: {fallback_error}")
                raise ValueError(f"Result for task {task_id} not found")
        
        # Check if task is completed
        if result_data.get("status") != "completed":
            in_progress_dict = {
                "task_id": task_id,
                "status": result_data.get("status", "pending"),
                "progress": result_data.get("progress", 0.0)
            }
            
            logger.info(f"📤 Returning in-progress status from task queue")
            print(f"Returning in-progress status from task queue")
            print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
            return in_progress_dict
        
        # Format as QueryResponse
        response = QueryResponse(
            task_id=task_id,
            result=result_data.get("result", {}),
            status="completed",
            execution_time=result_data.get("execution_time"),
            created_at=result_data.get("created_at")
        )
        
        response_dict = response.dict()
        logger.info(f"📤 Returning completed result from task queue")
        print(f"Returning completed result from task queue")
        print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
        return response_dict
    except Exception as e:
        logger.error(f"❌ Error getting query result: {str(e)}")
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        print(f"ERROR: Error getting query result: {str(e)}")
        print(f"\n=== TASK RESULT RETRIEVAL FAILED ===\n")
        raise
    
   

async def get_query_status(task_id: str) -> Dict[str, Any]:
    """
    Checks the status of a query by task ID with enhanced error handling and database fallback.
    
    Args:
        task_id: The task ID to check
        
    Returns:
        Dict containing status information
    """
    from multiagent.app.db.session import SessionLocal
    from multiagent.app.db.results import crud_result
    
    logger.debug(f"🔍 Checking status for task: {task_id}")
    print(f"\n=== CHECKING TASK STATUS ===\nTask ID: {task_id}")
    
    try:
        # First try to get status from the database for reliability
        try:
            with SessionLocal() as db:
                db_record = crud_result.get_by_task_id(db=db, task_id=task_id)
                
                if db_record:
                    logger.info(f"✅ Found task {task_id} in database with status: {db_record.status}")
                    print(f"Found task {task_id} in database with status: {db_record.status}")
                    
                    # Log the full record details
                    record_dict = {
                        "id": db_record.id,
                        "task_id": db_record.task_id,
                        "celery_task_id": db_record.celery_task_id,
                        "status": db_record.status,
                        "created_at": db_record.created_at.isoformat() if db_record.created_at else None,
                        "updated_at": db_record.updated_at.isoformat() if db_record.updated_at else None
                    }
                    logger.debug(f"📋 Database record details: {json.dumps(record_dict, indent=2)}")
                    print(f"Database record details: {pprint.pformat(record_dict)}")
                    
                    # Format as QueryStatus
                    db_status = QueryStatus(
                        task_id=task_id,
                        status=db_record.status,
                        progress=100 if db_record.status == "completed" else (
                            0 if db_record.status == "failed" else 50
                        ),
                        started_at=db_record.created_at,
                        updated_at=db_record.updated_at
                    )
                    
                    # If the task is completed or failed, just return the database status
                    if db_record.status in ["completed", "failed"]:
                        status_dict = db_status.dict()
                        logger.info(f"📤 Returning final status from database: {json.dumps(status_dict, indent=2)}")
                        print(f"Returning final status from database: {pprint.pformat(status_dict)}")
                        print(f"\n=== TASK STATUS CHECK COMPLETED ===\n")
                        return status_dict
                    
                    # For in-progress tasks, try to get more detailed status from the task queue
                    logger.info("🔄 Task is in progress, checking task queue for more details")
                    print("Task is in progress, checking task queue for more details")
        except Exception as db_error:
            logger.error(f"❌ Error getting status from database: {db_error}", exc_info=True)
            print(f"ERROR: Error getting status from database: {db_error}")
        
        # Get status from task queue for more detailed progress info
        logger.debug(f"🔍 Checking task queue for status of task: {task_id}")
        print(f"Checking task queue for status of task: {task_id}")
        status_data = await task_queue.get_task_status(task_id)
        
        logger.debug(f"📊 Retrieved status data from queue: {json.dumps(status_data, indent=2) if status_data else None}")
        print(f"Retrieved status data from queue: {pprint.pformat(status_data) if status_data else None}")
        
        if not status_data:
            logger.warning(f"⚠️ Task {task_id} not found in queue")
            print(f"WARNING: Task {task_id} not found in queue")
            
            # If we have a database record but no queue status, use the database status
            try:
                with SessionLocal() as db:
                    db_record = crud_result.get_by_task_id(db=db, task_id=task_id)
                    
                    if db_record:
                        logger.info(f"✅ Using database status for task {task_id} as fallback")
                        print(f"Using database status for task {task_id} as fallback")
                        
                        status_dict = QueryStatus(
                            task_id=task_id,
                            status=db_record.status,
                            progress=100 if db_record.status == "completed" else (
                                0 if db_record.status == "failed" else 50
                            ),
                            started_at=db_record.created_at,
                            updated_at=db_record.updated_at
                        ).dict()
                        
                        logger.info(f"📤 Returning fallback status from database: {json.dumps(status_dict, indent=2)}")
                        print(f"Returning fallback status from database: {pprint.pformat(status_dict)}")
                        print(f"\n=== TASK STATUS CHECK COMPLETED ===\n")
                        return status_dict
                    else:
                        logger.error(f"❌ Task {task_id} not found in database")
                        print(f"ERROR: Task {task_id} not found in database")
                        raise ValueError(f"Task {task_id} not found")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback to database failed: {fallback_error}", exc_info=True)
                print(f"ERROR: Fallback to database failed: {fallback_error}")
                raise ValueError(f"Task {task_id} not found")
        
        # Format as QueryStatus
        status = QueryStatus(
            task_id=task_id,
            status=status_data.get("status", "unknown"),
            progress=status_data.get("progress", 0.0),
            started_at=status_data.get("started_at"),
            updated_at=status_data.get("updated_at")
        )
        
        status_dict = status.dict()
        logger.info(f"📤 Returning status from task queue: {json.dumps(status_dict, indent=2)}")
        print(f"Returning status from task queue: {pprint.pformat(status_dict)}")
        print(f"\n=== TASK STATUS CHECK COMPLETED ===\n")
        return status_dict
    except Exception as e:
        logger.error(f"❌ Error getting query status: {str(e)}")
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        print(f"ERROR: Error getting query status: {str(e)}")
        print(f"\n=== TASK STATUS CHECK FAILED ===\n")
        raise

async def get_query_result(task_id: str) -> Dict[str, Any]:
    """
    Retrieves the result of a completed query with enhanced error handling and database fallback.
    
    Args:
        task_id: The task ID to retrieve
        
    Returns:
        Dict containing the query result
    """
    from multiagent.app.db.session import SessionLocal
    from multiagent.app.db.results import crud_result
    
    logger.debug(f"🔍 Retrieving result for task: {task_id}")
    print(f"\n=== RETRIEVING TASK RESULT ===\nTask ID: {task_id}")
    
    try:
        # First check the database for the result
        try:
            with SessionLocal() as db:
                db_record = crud_result.get_by_task_id(db=db, task_id=task_id)
                
                if db_record and db_record.status == "completed" and db_record.result:
                    logger.info(f"✅ Found completed result in database for task {task_id}")
                    print(f"Found completed result in database for task {task_id}")
                    
                    # Log the full record details
                    record_dict = {
                        "id": db_record.id,
                        "task_id": db_record.task_id,
                        "celery_task_id": db_record.celery_task_id,
                        "status": db_record.status,
                        "result": str(db_record.result)[:200] + "..." if db_record.result and len(str(db_record.result)) > 200 else db_record.result,
                        "created_at": db_record.created_at.isoformat() if db_record.created_at else None,
                        "updated_at": db_record.updated_at.isoformat() if db_record.updated_at else None
                    }
                    logger.debug(f"📋 Completed record details: {json.dumps(record_dict, indent=2)}")
                    print(f"Completed record details: {pprint.pformat(record_dict)}")
                    
                    # Calculate execution time if not present
                    execution_time = None
                    if db_record.result and isinstance(db_record.result, dict):
                        execution_time = db_record.result.get("execution_time")
                        logger.debug(f"⏱️ Execution time from result: {execution_time}")
                        print(f"Execution time from result: {execution_time}")
                    
                    # Format as QueryResponse
                    response_dict = QueryResponse(
                        task_id=task_id,
                        result=db_record.result,
                        status="completed",
                        execution_time=execution_time,
                        created_at=db_record.created_at
                    ).dict()
                    
                    logger.info(f"📤 Returning completed result from database")
                    print(f"Returning completed result from database")
                    print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                    return response_dict
                    
                elif db_record and db_record.status == "failed":
                    logger.info(f"⚠️ Task {task_id} failed, returning error result")
                    print(f"Task {task_id} failed, returning error result")
                    
                    # Log the error record
                    error_record = {
                        "id": db_record.id,
                        "task_id": db_record.task_id,
                        "status": db_record.status,
                        "error": db_record.result.get("error") if db_record.result and isinstance(db_record.result, dict) else "Unknown error"
                    }
                    logger.debug(f"📋 Failed task record: {json.dumps(error_record, indent=2)}")
                    print(f"Failed task record: {pprint.pformat(error_record)}")
                    
                    error_result = {
                        "task_id": task_id,
                        "status": "failed",
                        "error": "Task execution failed"
                    }
                    
                    if db_record.result and isinstance(db_record.result, dict):
                        error_result["error_details"] = db_record.result
                    
                    logger.info(f"📤 Returning error result from database")
                    print(f"Returning error result from database")
                    print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                    return error_result
                
                elif db_record:
                    logger.info(f"🔄 Task {task_id} is still in progress with status: {db_record.status}")
                    print(f"Task {task_id} is still in progress with status: {db_record.status}")
                    
                    # Log the in-progress record
                    progress_record = {
                        "id": db_record.id,
                        "task_id": db_record.task_id,
                        "status": db_record.status,
                        "created_at": db_record.created_at.isoformat() if db_record.created_at else None,
                        "updated_at": db_record.updated_at.isoformat() if db_record.updated_at else None
                    }
                    logger.debug(f"📋 In-progress task record: {json.dumps(progress_record, indent=2)}")
                    print(f"In-progress task record: {pprint.pformat(progress_record)}")
                    
                    progress_dict = {
                        "task_id": task_id,
                        "status": db_record.status,
                        "progress": 100 if db_record.status == "completed" else (
                            0 if db_record.status == "failed" else 50
                        )
                    }
                    
                    logger.info(f"📤 Returning in-progress status from database")
                    print(f"Returning in-progress status from database")
                    print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                    return progress_dict
        except Exception as db_error:
            logger.error(f"❌ Error getting result from database: {db_error}", exc_info=True)
            print(f"ERROR: Error getting result from database: {db_error}")
        
        # Get result from task queue as a fallback
        logger.debug(f"🔍 Checking task queue for result of task: {task_id}")
        print(f"Checking task queue for result of task: {task_id}")
        result_data = await task_queue.get_task_result(task_id)
        
        logger.debug(f"📊 Retrieved result data from queue: {json.dumps(str(result_data)[:200], indent=2) + '...' if result_data and len(str(result_data)) > 200 else result_data}")
        print(f"Retrieved result data from queue: {str(result_data)[:200] + '...' if result_data and len(str(result_data)) > 200 else result_data}")
        
        if not result_data:
            logger.warning(f"⚠️ Result for task {task_id} not found in queue")
            print(f"WARNING: Result for task {task_id} not found in queue")
            
            # Check database again in case we missed it
            try:
                with SessionLocal() as db:
                    db_record = crud_result.get_by_task_id(db=db, task_id=task_id)
                    
                    if db_record:
                        logger.info(f"✅ Found task {task_id} in database with status: {db_record.status}")
                        print(f"Found task {task_id} in database with status: {db_record.status}")
                        
                        if db_record.status == "completed" and db_record.result:
                            # Format as QueryResponse for completed tasks
                            fallback_response = QueryResponse(
                                task_id=task_id,
                                result=db_record.result,
                                status="completed",
                                execution_time=None,
                                created_at=db_record.created_at
                            ).dict()
                            
                            logger.info(f"📤 Returning completed result from database fallback")
                            print(f"Returning completed result from database fallback")
                            print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                            return fallback_response
                        else:
                            # Return status for in-progress tasks
                            fallback_status = {
                                "task_id": task_id,
                                "status": db_record.status,
                                "progress": 100 if db_record.status == "completed" else (
                                    0 if db_record.status == "failed" else 50
                                )
                            }
                            
                            logger.info(f"📤 Returning status from database fallback")
                            print(f"Returning status from database fallback")
                            print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
                            return fallback_status
                    else:
                        logger.error(f"❌ Result for task {task_id} not found in database")
                        print(f"ERROR: Result for task {task_id} not found in database")
                        raise ValueError(f"Result for task {task_id} not found")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback to database failed: {fallback_error}", exc_info=True)
                print(f"ERROR: Fallback to database failed: {fallback_error}")
                raise ValueError(f"Result for task {task_id} not found")
        
        # Check if task is completed
        if result_data.get("status") != "completed":
            in_progress_dict = {
                "task_id": task_id,
                "status": result_data.get("status", "pending"),
                "progress": result_data.get("progress", 0.0)
            }
            
            logger.info(f"📤 Returning in-progress status from task queue")
            print(f"Returning in-progress status from task queue")
            print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
            return in_progress_dict
        
        # Format as QueryResponse
        response = QueryResponse(
            task_id=task_id,
            result=result_data.get("result", {}),
            status="completed",
            execution_time=result_data.get("execution_time"),
            created_at=result_data.get("created_at")
        )
        
        response_dict = response.dict()
        logger.info(f"📤 Returning completed result from task queue")
        print(f"Returning completed result from task queue")
        print(f"\n=== TASK RESULT RETRIEVAL COMPLETED ===\n")
        return response_dict
    except Exception as e:
        logger.error(f"❌ Error getting query result: {str(e)}")
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        print(f"ERROR: Error getting query result: {str(e)}")
        print(f"\n=== TASK RESULT RETRIEVAL FAILED ===\n")
        raise