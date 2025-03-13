from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel


DataT = TypeVar("DataT")


class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    """
    detail: str = Field(..., description="Error details")
    error_code: Optional[str] = Field(None, description="Error code")
    location: Optional[str] = Field(None, description="Error location")


class ValidationError(BaseModel):
    """
    Schema for validation errors.
    """
    loc: List[str] = Field(..., description="Location of the error")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class HTTPValidationError(BaseModel):
    """
    Schema for HTTP validation errors.
    """
    detail: List[ValidationError] = Field(..., description="Validation errors")


class StandardResponse(GenericModel, Generic[DataT]):
    """
    Generic schema for standardized API responses.
    Can contain any data type.
    """
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[DataT] = Field(None, description="Response data")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Errors (if any)")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PaginatedResponse(GenericModel, Generic[DataT]):
    """
    Generic schema for paginated responses.
    """
    items: List[DataT] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")