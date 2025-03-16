# api/response.py
from typing import Dict, Any, Optional, Generic, TypeVar, List
from pydantic import BaseModel
from pydantic.generics import GenericModel

T = TypeVar('T')

class StandardResponse(GenericModel, Generic[T]):
    """Standard API response format."""
    status: str
    data: Optional[T] = None
    message: Optional[str] = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class PaginatedResponse(GenericModel, Generic[T]):
    """Paginated response format."""
    status: str
    data: List[T]
    message: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = {
        "page": 1,
        "per_page": 10,
        "total": 0,
        "pages": 1
    }

class ErrorResponse(BaseModel):
    """Error response format."""
    status: str = "error"
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

def create_success_response(data: Any, message: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> StandardResponse:
    """
    Create a success response.
    
    Args:
        data: Response data
        message: Optional success message
        meta: Optional metadata
        
    Returns:
        StandardResponse with success status
    """
    return StandardResponse(
        status="success",
        data=data,
        message=message,
        meta=meta
    )

def create_error_response(error: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """
    Create an error response.
    
    Args:
        error: Error message
        error_code: Optional error code
        details: Optional error details
        
    Returns:
        ErrorResponse with error information
    """
    return ErrorResponse(
        error=error,
        error_code=error_code,
        details=details
    )

def create_paginated_response(
    data: List[Any],
    page: int = 1,
    per_page: int = 10,
    total: Optional[int] = None,
    message: Optional[str] = None
) -> PaginatedResponse:
    """
    Create a paginated response.
    
    Args:
        data: List of items
        page: Current page number
        per_page: Items per page
        total: Optional total item count
        message: Optional message
        
    Returns:
        PaginatedResponse with pagination metadata
    """
    # Calculate total if not provided
    if total is None:
        total = len(data)
        
    # Calculate total pages
    pages = (total + per_page - 1) // per_page if per_page > 0 else 1
    
    return PaginatedResponse(
        status="success",
        data=data,
        message=message,
        meta={
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": pages
        }
    )