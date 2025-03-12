from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Schema for query request.
    Contains the query text and workflow ID.
    """
    query: str = Field(..., description="Query text to process")
    workflow_id: str = Field("research", description="ID of the workflow to execute")


class QueryResponse(BaseModel):
    """
    Schema for query response.
    Contains the status, task ID, and results (if available).
    """
    task_id: str = Field(..., description="Task ID for tracking the query")
    status: str = Field(..., description="Status of the query (accepted, processing, completed, failed)")
    message: str = Field(..., description="Message about the query status")
    query: Optional[str] = Field(None, description="Original query text")
    result: Optional[Dict[str, Any]] = Field(None, description="Query result (if completed)")


class QueryStatus(BaseModel):
    """
    Schema for query status.
    Provides information about the current status of a query.
    """
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Status of the query (pending, processing, completed, failed, cancelled)")
    progress: Optional[int] = Field(None, description="Progress percentage (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")


class QueryResult(BaseModel):
    """
    Schema for complete query result.
    Contains all information from the multiagent processing.
    """
    query: str = Field(..., description="Original query")
    summary: str = Field(..., description="Final summary")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    sources: List[Dict[str, Any]] = Field([], description="Source information")
    research_data: Optional[Dict[str, Any]] = Field(None, description="Detailed research data")
    analysis_data: Optional[Dict[str, Any]] = Field(None, description="Detailed analysis data")
    processing_time: float = Field(..., description="Total processing time in seconds")
    completion_time: str = Field(..., description="Timestamp of completion")