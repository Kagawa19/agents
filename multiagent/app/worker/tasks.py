"""
Task definitions for asynchronous processing.
Defines Celery tasks for executing workflows and agents.
"""
import asyncio
from datetime import datetime
import json
import logging
import time
from sqlalchemy import text
import pprint
from typing import Any, Dict, Optional
from multiagent.app.worker.queue import TaskQueue

# Create a global task_queue instance

import traceback
from celery import states
from celery.exceptions import MaxRetriesExceededError, Retry
from sqlalchemy import text

from multiagent.app.api.websocket import connection_manager
from multiagent.app.db.base import Base  # Import Base from base.py
from multiagent.app.db.session import SessionLocal
from multiagent.app.worker.celery_app import celery_app

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [tasks.py] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

task_queue = TaskQueue()

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
    # Import inside the function to avoid circular imports
    from multiagent.app.db.models import Result
    from multiagent.app.db.results import crud_result
    from multiagent.app.core.config import settings
    
    # Debug Point 1: Entry point - verify task received with correct parameters
    logger.info(f"🚀 DEBUG POINT 1: Task started with workflow_id={workflow_id}, input_data keys={list(input_data.keys())}")
    print(f"\n=== WORKFLOW TASK EXECUTION STARTED ===\nWorkflow ID: {workflow_id}\nInput Data Keys: {list(input_data.keys())}")
    
    # Extract input data
    task_id = input_data.get("task_id")
    query = input_data.get("query")
    user_id = input_data.get("user_id")
    parameters = input_data.get("parameters", {})
    trace_id = input_data.get("trace_id")
    
    # Debug Point 2: After extracting input data
    logger.info(f"📋 DEBUG POINT 2: Extracted data - task_id={task_id}, user_id={user_id}, query_preview={query[:50] if query else 'None'}")
    print(f"Extracted data:\nTask ID: {task_id}\nUser ID: {user_id}\nQuery: {query[:50] if query else 'None'}")
    
    # Enhanced logging with more context
    logger.info(
        f"🔄 Starting workflow execution: workflow={workflow_id}, "
        f"task_id={task_id}, user_id={user_id}, "
        f"query={query[:100] if query else 'N/A'}"
    )
    
    # Detailed log for input parameters
    logger.debug(f"🔧 Workflow input parameters: {json.dumps(parameters, indent=2)}")
    print(f"Input parameters: {pprint.pformat(parameters)}")
    
    # Debug Point 3: Before database operations
    logger.info(f"💾 DEBUG POINT 3: About to perform database operations for task_id={task_id}")
    print(f"Preparing database operations for task_id={task_id}")
    
    # Get the Celery task ID
    celery_task_id = self.request.id
    logger.info(f"🆔 Celery task ID: {celery_task_id}")
    print(f"Celery task ID: {celery_task_id}")
    
    # Verify database connection first
    db_connection_verified = False
    try:
        with SessionLocal() as db:
            # Test database connection - using text() for raw SQL
            result = db.execute(text("SELECT 1")).scalar()
            db_connection_verified = result == 1
            logger.info(f"✅ Database connection verified: {db_connection_verified}")
            print(f"Database connection verified: {db_connection_verified}")
    except Exception as db_conn_error:
        logger.error(f"❌ Error verifying database connection: {str(db_conn_error)}", exc_info=True)
        print(f"ERROR: Database connection verification failed: {db_conn_error}")
    
    if not db_connection_verified:
        error_msg = "Could not verify database connection"
        logger.critical(f"🚨 {error_msg}")
        print(f"CRITICAL ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    
    # Save task to database
    try:
        # Create a new database session for this operation
        with SessionLocal() as db:
            # First, check if there is already a record for this task
            existing_record = crud_result.get_by_task_id(db=db, task_id=task_id)
            
            logger.info(f"🔍 Existing record for task {task_id}: {existing_record is not None}")
            print(f"Existing record for task {task_id}: {existing_record is not None}")
            
            if existing_record:
                # Log existing record details
                record_dict = {
                    "id": existing_record.id,
                    "task_id": existing_record.task_id,
                    "status": existing_record.status,
                    "celery_task_id": existing_record.celery_task_id
                }
                logger.debug(f"📋 Existing record details: {json.dumps(record_dict, indent=2)}")
                print(f"Existing record details: {pprint.pformat(record_dict)}")
            
            # Save or update the result
            saved_result = crud_result.save_result(
                db=db,
                task_id=task_id,
                query=query,
                workflow=workflow_id,
                user_id=user_id,
                status="processing",
                celery_task_id=celery_task_id
            )
            
            # Log saved result details
            if saved_result:
                saved_dict = {
                    "id": saved_result.id,
                    "task_id": saved_result.task_id,
                    "status": saved_result.status,
                    "celery_task_id": saved_result.celery_task_id,
                    "created_at": saved_result.created_at.isoformat() if saved_result.created_at else None,
                    "updated_at": saved_result.updated_at.isoformat() if saved_result.updated_at else None
                }
                logger.debug(f"📋 Saved/updated record details: {json.dumps(saved_dict, indent=2)}")
                print(f"Saved/updated record details: {pprint.pformat(saved_dict)}")
            
            # Verify the save operation immediately
            verification = crud_result.get_by_task_id(db=db, task_id=task_id)
            
            if verification:
                logger.info(f"✅ Task {task_id} saved to database with id: {verification.id}")
                print(f"Task {task_id} saved to database with id: {verification.id}")
                
                # Verify the record was properly updated
                if verification.status == "processing" and verification.celery_task_id == celery_task_id:
                    logger.info(f"✅ Verified record was properly updated")
                    print(f"Verified record was properly updated")
                else:
                    logger.warning(f"⚠️ Record update verification failed - status: {verification.status}, celery_task_id: {verification.celery_task_id}")
                    print(f"WARNING: Record update verification failed")
            else:
                logger.error(f"❌ Failed to verify task {task_id} saved to database!")
                print(f"ERROR: Failed to verify task {task_id} saved to database!")
                
                # Try an alternative approach if verification fails
                logger.info(f"🔄 Attempting direct SQL insert/update as fallback")
                print(f"Attempting direct SQL insert/update as fallback")
                
                db.execute(
                    text("INSERT INTO results (task_id, query, workflow, user_id, status, celery_task_id, created_at, updated_at) "
                    "VALUES (:task_id, :query, :workflow, :user_id, 'processing', :celery_task_id, NOW(), NOW()) "
                    "ON CONFLICT (task_id) DO UPDATE SET "
                    "status = 'processing', updated_at = NOW(), celery_task_id = :celery_task_id"),
                    {
                        "task_id": task_id,
                        "query": query,
                        "workflow": workflow_id,
                        "user_id": user_id, 
                        "celery_task_id": celery_task_id
                    }
                )
                db.commit()
                logger.info("✅ Direct SQL insert/update completed")
                print(f"Direct SQL insert/update completed")
                
                # Verify one more time
                final_check = crud_result.get_by_task_id(db=db, task_id=task_id)
                if final_check:
                    logger.info(f"✅ Final verification successful for task {task_id}")
                    print(f"Final verification successful for task {task_id}")
                else:
                    logger.error(f"❌ Final verification failed for task {task_id}")
                    print(f"ERROR: Final verification failed for task {task_id}")
    except Exception as save_error:
        logger.error(
            f"❌ Failed to save task to database: {save_error}", 
            exc_info=True
        )
        print(f"ERROR: Failed to save task to database: {save_error}")
        # Continue execution even if database save fails
        # This allows the workflow to attempt to run even if DB operations fail
    
    # Send initial progress update
    logger.info(f"🔄 Sending initial progress update for task {task_id}")
    print(f"Sending initial progress update for task {task_id}")
    update_progress.delay(
        task_id=task_id,
        status="processing",
        progress=10,
        current_step="Initializing workflow execution"
    )
    
    try:
        # Debug Point 4: Before component initialization
        logger.info(f"🧩 DEBUG POINT 4: About to initialize components for task_id={task_id}")
        print(f"About to initialize components for task_id={task_id}")
        
        # Initialize components with detailed logging
        try:
            logger.info("🔧 Setting up execution environment")
            print(f"Setting up execution environment")
            
            from multiagent.app.monitoring.tracer import get_tracer
            from multiagent.app.orchestrator.manager import AgentManager
            from multiagent.app.orchestrator.workflow import WorkflowManager
            
            # Progress update
            logger.info(f"🔄 Updating progress: 20% - Setting up execution environment")
            print(f"Updating progress: 20% - Setting up execution environment")
            update_progress.delay(
                task_id=task_id,
                status="processing",
                progress=20,
                current_step="Setting up execution environment"
            )
            
            tracer = get_tracer()
            logger.info("✅ Tracer initialized")
            print(f"Tracer initialized")
            
            agent_manager = AgentManager(settings, tracer)
            agent_manager.initialize()
            logger.info("✅ Agent manager initialized")
            print(f"Agent manager initialized")
            
            logger.info(f"🔄 Updating progress: 30% - Initializing agent manager")
            print(f"Updating progress: 30% - Initializing agent manager")
            update_progress.delay(
                task_id=task_id,
                status="processing", 
                progress=30,
                current_step="Initializing agent manager"
            )
            
            workflow_manager = WorkflowManager(agent_manager, tracer)
            logger.info("✅ Workflow manager created")
            print(f"Workflow manager created")
            
        except Exception as setup_error:
            # Detailed error logging for setup failures
            error_msg = f"Error setting up workflow environment: {str(setup_error)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            print(f"ERROR: {error_msg}")
            
            # Send specific error progress update
            logger.info(f"🔄 Updating progress: Failed - Error during setup")
            print(f"Updating progress: Failed - Error during setup")
            update_progress.delay(
                task_id=task_id,
                status="failed",
                progress=0,
                error=error_msg
            )
            
            # Save specific error to database
            try:
                with SessionLocal() as db:
                    crud_result.save_result(
                        db=db,
                        task_id=task_id,
                        query=query,
                        workflow=workflow_id,
                        result={"error": error_msg},
                        user_id=user_id,
                        status="failed",
                        celery_task_id=celery_task_id
                    )
                    # Explicitly commit
                    db.commit()
                    logger.info(f"✅ Error result saved to database for task {task_id}")
                    print(f"Error result saved to database for task {task_id}")
                    
                    # Verify the error was saved correctly
                    verification = crud_result.get_by_task_id(db=db, task_id=task_id)
                    if verification and verification.status == "failed":
                        logger.info(f"✅ Verified error was properly saved to database")
                        print(f"Verified error was properly saved to database")
                    else:
                        logger.warning(f"⚠️ Could not verify error was saved properly")
                        print(f"WARNING: Could not verify error was saved properly")
            except Exception as db_error:
                logger.error(f"❌ Failed to save error to database: {db_error}", exc_info=True)
                print(f"ERROR: Failed to save error to database: {db_error}")
            
            # Re-raise with more specific message
            raise Exception(error_msg)
        
        # Execute the workflow with progress updates
        start_time = time.time()
        
        # Debug Point 5: Before workflow execution
        logger.info(f"🏃 DEBUG POINT 5: About to execute workflow for task_id={task_id}")
        print(f"About to execute workflow for task_id={task_id}")
        
        # Log before workflow execution
        logger.info(f"🔄 Beginning workflow execution: {workflow_id}")
        print(f"Beginning workflow execution: {workflow_id}")
        
        logger.info(f"🔄 Updating progress: 50% - Starting workflow execution")
        print(f"Updating progress: 50% - Starting workflow execution")
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
                logger.info(f"👟 Executing workflow {workflow_id} with input data")
                print(f"Executing workflow {workflow_id} with input data")
                result = await workflow_manager.execute_workflow(workflow_id, input_data)
                
                # Log the result type and size
                result_type = type(result).__name__
                result_size = len(json.dumps(result)) if isinstance(result, (dict, list)) else len(str(result))
                logger.info(f"✅ Workflow execution returned result of type {result_type}, size {result_size} bytes")
                print(f"Workflow execution returned result of type {result_type}, size {result_size} bytes")
                
                # Ensure result is fully serializable
                try:
                    json.dumps(result)
                    logger.debug(f"✅ Result is JSON serializable")
                    print(f"Result is JSON serializable")
                except TypeError:
                    logger.warning(f"⚠️ Result is not JSON serializable, converting to string")
                    print(f"WARNING: Result is not JSON serializable, converting to string")
                    result = str(result)
                
                return result
            
            except Exception as e:
                logger.error(f"❌ Workflow execution internal error: {e}")
                print(f"ERROR: Workflow execution internal error: {e}")
                raise
        
        # Run async workflow execution
        logger.info(f"🔄 Running async workflow execution")
        print(f"Running async workflow execution")
        result = asyncio.run(safe_workflow_execute())
        
        # Debug Point 6: After workflow execution
        logger.info(f"✅ DEBUG POINT 6: Workflow execution completed for task_id={task_id}, result_preview={str(result)[:100]}")
        print(f"Workflow execution completed for task_id={task_id}")
        print(f"Result preview: {str(result)[:100]}...")
        
        execution_time = time.time() - start_time
        
        # Log successful execution
        logger.info(
            f"🎯 Workflow {workflow_id} completed successfully. "
            f"Execution time: {execution_time:.2f} seconds"
        )
        print(f"Workflow {workflow_id} completed successfully in {execution_time:.2f} seconds")
        
        # Progress update
        logger.info(f"🔄 Updating progress: 90% - Workflow execution completed, finalizing results")
        print(f"Updating progress: 90% - Workflow execution completed, finalizing results")
        update_progress.delay(
            task_id=task_id,
            status="processing",
            progress=90,
            current_step="Workflow execution completed, finalizing results"
        )
        
        # Save result to database with multiple retries
        max_retries = (3 if settings.DEBUG else 3)
        save_success = False
        
        for attempt in range(max_retries):
            try:
                with SessionLocal() as db:
                    # Verify database connection before saving
                    db_check = db.execute(text("SELECT 1")).scalar()
                    if db_check != 1:
                        logger.error(f"❌ Failed to verify database connection on attempt {attempt+1}")
                        print(f"ERROR: Failed to verify database connection on attempt {attempt+1}")
                        raise RuntimeError("Failed to verify database connection")
                    
                    logger.info(f"🔄 Saving result to database (attempt {attempt+1}/{max_retries})")
                    print(f"Saving result to database (attempt {attempt+1}/{max_retries})")
                    
                    saved_result = crud_result.save_result(
                        db=db,
                        task_id=task_id,
                        query=query,
                        workflow=workflow_id,
                        result=result,
                        user_id=user_id,
                        status="completed",
                        celery_task_id=celery_task_id
                    )
                    
                    # Explicitly commit changes
                    db.commit()
                    
                    # Verify the save operation
                    verification = crud_result.get_by_task_id(db=db, task_id=task_id)
                    if verification and verification.status == "completed":
                        logger.info(f"✅ Result saved to database for task {task_id}")
                        print(f"Result saved to database for task {task_id}")
                        break
                    else:
                        logger.warning(f"Could not verify result save for task {task_id}")
                        print(f"WARNING: Could not verify result save for task {task_id}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying save operation (attempt {attempt + 1}/{max_retries})")
                            print(f"Retrying save operation (attempt {attempt + 1}/{max_retries})")
                            time.sleep(1)  # Brief delay before retry
            except Exception as save_error:
                logger.error(f"Error saving result (attempt {attempt + 1}/{max_retries}): {save_error}", exc_info=True)
                print(f"ERROR: Error saving result (attempt {attempt + 1}/{max_retries}): {save_error}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying save operation")
                    print(f"Retrying save operation")
                    time.sleep(1)  # Brief delay before retry
                else:
                    logger.critical(f"Failed to save result after {max_retries} attempts")
                    print(f"CRITICAL: Failed to save result after {max_retries} attempts")
        
        # Add execution time to the result
        result["execution_time"] = execution_time
        
        # Send completion notification
        update_progress.delay(
            task_id=task_id,
            status="completed",
            progress=100,
            result=result
        )
        
        # Debug Point 7: Before final result return
        logger.info(f"DEBUG POINT 7: Returning final result for task_id={task_id}")
        print(f"Returning final result for task_id={task_id}")
        
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
        print(f"ERROR: Workflow execution failed: {error_msg}")
        
        # Create structured error response
        error_details = {
            "error": error_msg,
            "workflow_id": workflow_id,
            "task_id": task_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "trace": stack_trace if settings.DEBUG else "Enable DEBUG mode to see trace" 
        }
        
        # Save error to database with multiple retries
        max_retries = (3 if settings.DEBUG else 3)
        for attempt in range(max_retries):
            try:
                with SessionLocal() as db:
                    crud_result.save_result(
                        db=db,
                        task_id=task_id,
                        query=query,
                        workflow=workflow_id, 
                        result=error_details,
                        user_id=user_id,
                        status="failed",
                        celery_task_id=celery_task_id
                    )
                    
                    # Explicitly commit changes
                    db.commit()
                    
                    # Verify the save operation
                    verification = crud_result.get_by_task_id(db=db, task_id=task_id)
                    if verification and verification.status == "failed":
                        logger.info(f"Error result saved to database for task {task_id}")
                        print(f"Error result saved to database for task {task_id}")
                        break
                    else:
                        logger.warning(f"Could not verify error save for task {task_id}")
                        print(f"WARNING: Could not verify error save for task {task_id}")
                        if attempt < max_retries - 1:
                            time.sleep(1)  # Brief delay before retry
            except Exception as save_error:
                logger.error(f"Error saving failure result (attempt {attempt + 1}/{max_retries}): {save_error}", exc_info=True)
                print(f"ERROR: Error saving failure result (attempt {attempt + 1}/{max_retries}): {save_error}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief delay before retry
        
        # Send error notification  
        update_progress.delay(
            task_id=task_id,
            status="failed",
            progress=0,
            error=error_msg  
        )
        
        # Log final error state
        logger.error(f"Workflow {workflow_id} failed with error: {error_msg}")
        print(f"Workflow {workflow_id} failed with error: {error_msg}")
        
        # Re-raise the exception with structured error
        raise Exception(f"Workflow execution failed: {error_msg}")
    
@celery_app.task(bind=True, name="app.worker.tasks.update_progress")
def update_progress(
    self,
    task_id: str,
    status: str,
    progress: int = 0,
    current_step: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update the progress of a task in both the task queue and database.
    
    Args:
        task_id: ID of the task to update
        status: New status ('processing', 'completed', 'failed')
        progress: Progress percentage (0-100)
        current_step: Description of current processing step
        result: Task result (if completed)
        error: Error message (if failed)
    
    Returns:
        Dict with update status information
    """
    from multiagent.app.db.results import crud_result
    from multiagent.app.db.session import SessionLocal
    
    logger.debug(f"Updating progress for task {task_id}: status={status}, progress={progress}")
    print(f"Updating progress for task {task_id}: status={status}, progress={progress}, step={current_step}")
    
    # Update in task queue
    response = {
        "task_id": task_id,
        "status": status,
        "progress": progress
    }
    
    try:
        # Update task in queue asynchronously
        asyncio.run(task_queue.update_task_status(task_id, status, progress, current_step, result, error))
        logger.debug(f"Updated task {task_id} in queue")
        print(f"Updated task {task_id} in queue")
    except Exception as queue_error:
        logger.error(f"Error updating task in queue: {queue_error}", exc_info=True)
        print(f"ERROR: Error updating task in queue: {queue_error}")
        response["queue_update_error"] = str(queue_error)
    
    # Update in database with verification and retries
    max_retries = 3
    db_update_success = False
    
    for attempt in range(max_retries):
        try:
            with SessionLocal() as db:
                # First check if connection is working
                connection_check = db.execute(text("SELECT 1")).scalar()
                if connection_check != 1:
                    raise RuntimeError("Database connection check failed")
                
                # Get existing record
                existing_record = crud_result.get_by_task_id(db=db, task_id=task_id)
                
                if existing_record:
                    # Prepare update data
                    update_data = {
                        "status": status,
                        "updated_at": datetime.utcnow()
                    }
                    
                    # Add result or error if provided
                    if result is not None:
                        update_data["result"] = result
                        print(f"Adding result to update (truncated): {str(result)[:100]}...")
                    
                    if error is not None and status == "failed":
                        print(f"Adding error to update: {error}")
                        if existing_record.result:
                            # Preserve existing result data if any
                            current_result = existing_record.result
                            if isinstance(current_result, dict):
                                current_result["error"] = error
                                update_data["result"] = current_result
                                print("Merged error with existing result dictionary")
                            else:
                                update_data["result"] = {"previous_data": str(current_result), "error": error}
                                print("Created new result dictionary with previous data and error")
                        else:
                            update_data["result"] = {"error": error}
                            print("Created new result dictionary with error")
                    
                    # Update the record
                    crud_result.update(db=db, db_obj=existing_record, obj_in=update_data)
                    
                    # Commit changes
                    db.commit()
                    
                    # Verify the update
                    verification = crud_result.get_by_task_id(db=db, task_id=task_id)
                    if verification and verification.status == status:
                        logger.info(f"Updated task {task_id} in database: status={status}, progress={progress}")
                        print(f"Updated task {task_id} in database: status={status}, progress={progress}")
                        db_update_success = True
                        break
                    else:
                        logger.warning(f"Could not verify database update for task {task_id}")
                        print(f"WARNING: Could not verify database update for task {task_id}")
                        
                        # Try a more direct approach if verification fails
                        if attempt == max_retries - 1:
                            logger.info("Attempting direct SQL update as last resort")
                            print("Attempting direct SQL update as last resort")
                            # Convert result to JSON if provided
                            result_json = json.dumps(result) if result is not None else None
                            error_json = json.dumps({"error": error}) if error is not None else None
                            
                            # Use SQL to update directly
                            db.execute(
                                text("UPDATE results SET status = :status, updated_at = NOW() "
                                + (", result = :result::jsonb " if result is not None else "")
                                + (", result = :error::jsonb " if error is not None and result is None else "")
                                + "WHERE task_id = :task_id"),
                                {
                                    "status": status,
                                    "task_id": task_id,
                                    "result": result_json,
                                    "error": error_json
                                }
                            )
                            db.commit()
                            logger.info("Executed direct SQL update")
                            print("Executed direct SQL update")
                else:
                    # No existing record, create a new one
                    logger.warning(f"No existing record found for task {task_id}, creating new one")
                    print(f"WARNING: No existing record found for task {task_id}, creating new one")
                    crud_result.save_result(
                        db=db,
                        task_id=task_id,
                        query="unknown",  # we don't have the original query here
                        workflow="unknown",  # we don't have the workflow type here
                        status=status,
                        result=result if result is not None else ({"error": error} if error is not None else None)
                    )
                    
                    # Commit changes
                    db.commit()
                    
                    # Verify the creation
                    # Verify the creation
                    verification = crud_result.get_by_task_id(db=db, task_id=task_id)
                    if verification:
                        logger.info(f"Created new record for task {task_id} in database")
                        print(f"Created new record for task {task_id} in database")
                        db_update_success = True
                        break
                    else:
                        logger.warning(f"Could not verify record creation for task {task_id}")
                        print(f"Could not verify record creation for task {task_id}")
            
        except Exception as db_error:
            logger.error(f"Error updating task in database (attempt {attempt+1}/{max_retries}): {db_error}", exc_info=True)
            print(f"ERROR: Error updating task in database (attempt {attempt+1}/{max_retries}): {db_error}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying database update in 1 second...")
                print(f"Retrying database update in 1 second...")
                time.sleep(1)  # Brief delay before retry
    
    # Add database update status to response
    response["db_update_success"] = db_update_success
    
    return response