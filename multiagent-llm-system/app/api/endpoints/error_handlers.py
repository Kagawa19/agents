from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings


def add_exception_handlers(app: FastAPI) -> None:
    """
    Adds exception handlers to the FastAPI application.
    
    Args:
        app: FastAPI application
    """
    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        """
        Handles SQLAlchemy exceptions.
        
        Args:
            request: Request that caused the exception
            exc: SQLAlchemy exception
            
        Returns:
            JSON response with error details
        """
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Database error occurred"}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Handles general exceptions.
        
        Args:
            request: Request that caused the exception
            exc: Exception
            
        Returns:
            JSON response with error details
        """
        # Log the exception
        error_detail = str(exc)
        if settings.DEBUG:
            error_detail = f"{exc.__class__.__name__}: {error_detail}"
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": error_detail}
        )