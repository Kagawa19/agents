# app/api/endpoints/query.py
from typing import Any, Dict
import logging

from fastapi import APIRouter, Depends, HTTPException, Path, status
from fastapi.websockets import WebSocket

from app.api.deps import get_workflow_manager
from app.api.websocket import websocket_endpoint
from app.db.crud.results import crud_result
from app.db.session import get_db
from app.orchestrator.workflow import WorkflowManager
from app.schemas.query import QueryRequest, QueryResponse, QueryStatus
from app.worker.queue import TaskQueue


logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_query(
    request: QueryRequest,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager),
    db = Depends(get_db)
) -> Dict[str, Any]:
    """
    Submit a query for processing.
    The query will be processed asynchronously.
    """
    logger.info(f"Received query request for workflow_id: {request.workflow_id}")
    
    # Check if workflow exists
    if request.workflow_id not in workflow_manager.workflows:
        logger.error(f"Workflow {request.workflow_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {request.workflow_id} not found"
        )
    
    # Submit task
    task_queue = TaskQueue()
    task_id = task_queue.submit_task(
        "app.worker.tasks.execute_workflow_task",
        workflow_id=request.workflow_id,
        input_data={
            "query": request.query,
            "user_id": "anonymous_user"
        }
    )
    logger.info(f"Task submitted successfully. Task ID: {task_id}")
    
    return {
        "task_id": task_id,
        "status": "accepted",
        "query": request.query,
        "message": "Query submitted for processing"
    }

@router.get("/query/{task_id}", response_model=QueryResponse)
async def get_query_result(
    task_id: str = Path(..., description="Task ID"),
    db = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get the result of a query.
    """
    logger.info(f"Checking result for task_id: {task_id}")
    
    # Check task status
    task_queue = TaskQueue()
    task_status = task_queue.get_task_status(task_id)
    logger.debug(f"Task status for {task_id}: {task_status}")
    
    if task_status in ["PENDING", "STARTED"]:
        logger.info(f"Task {task_id} still processing")
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Query still processing"
        }
    elif task_status == "FAILURE":
        logger.error(f"Task {task_id} failed")
        return {
            "task_id": task_id,
            "status": "failed",
            "message": "Query processing failed"
        }
    elif task_status == "SUCCESS":
        logger.info(f"Task {task_id} completed successfully")
        # Get result from task
        result = task_queue.get_task_result(task_id)
        
        if not result:
            logger.debug(f"No result in task queue for {task_id}, checking database")
            # Try to get from database
            db_result = crud_result.get_by_task_id(db, task_id=task_id)
            if db_result:
                result = db_result.result
                logger.debug(f"Found result in database for task {task_id}")
            else:
                logger.warning(f"No result found for completed task {task_id}")
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "message": "Result not found in database",
                    "result": None
                }
        
        return {
            "task_id": task_id,
            "status": "completed",
            "message": "Query processing completed",
            "result": result
        }
    else:
        logger.error(f"Unknown task status for {task_id}: {task_status}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unknown task status: {task_status}"
        )

@router.get("/query/{task_id}/status", response_model=QueryStatus)
async def get_query_status(
    task_id: str = Path(..., description="Task ID")
) -> Dict[str, Any]:
    """
    Get the status of a query.
    """
    logger.debug(f"Checking status for task_id: {task_id}")
    task_queue = TaskQueue()
    task_status = task_queue.get_task_status(task_id)
    
    status_mapping = {
        "PENDING": "pending",
        "STARTED": "processing",
        "SUCCESS": "completed",
        "FAILURE": "failed",
        "REVOKED": "cancelled"
    }
    
    status = status_mapping.get(task_status, "unknown")
    logger.info(f"Task {task_id} status: {status}")
    
    return {
        "task_id": task_id,
        "status": status
    }

@router.websocket("/ws")
async def websocket_query_updates(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time query updates.
    """
    logger.info("New WebSocket connection established")
    await websocket_endpoint(websocket)
