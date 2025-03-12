"""
Database session configuration.
Provides connection setup and session management for SQLAlchemy.
"""

import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings


logger = logging.getLogger(__name__)

# Create engine for connecting to the database
engine = create_engine(
    settings.DATABASE_URI,
    pool_pre_ping=True,
    echo=settings.DATABASE_ECHO
)

# Create session factory for creating database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.
    Yields a SQLAlchemy session and ensures it's closed after use.
    
    Yields:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize the database.
    Creates all tables if they don't exist.
    """
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def close_db() -> None:
    """
    Close database connections.
    Should be called during application shutdown.
    """
    try:
        if engine is not None:
            engine.dispose()
            logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")