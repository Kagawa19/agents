import logging
import importlib
import sys
from sqlalchemy import inspect, text
import pkgutil

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from multiagent.app.core.config import settings
from multiagent.app.db.base import Base
from multiagent.app.db.migrations.migration_manager import MigrationManager

logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(
    settings.DATABASE_URI,
    pool_pre_ping=True,
    echo=settings.DATABASE_ECHO
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db(drop_existing: bool = False) -> None:
    try:
        # Dynamically import models
        models_module = importlib.import_module('multiagent.app.db.models')
        
        # Log all available models
        logger.info("Available models:")
        for name, obj in models_module.__dict__.items():
            if hasattr(obj, '__tablename__'):
                logger.info(f"- {name}")
        
        inspector = inspect(engine)
        
        # Drop tables if specified
        if drop_existing:
            Base.metadata.drop_all(bind=engine)
            logger.warning("All existing database tables dropped")
        
        # Get all existing tables
        existing_tables = inspector.get_table_names()
        logger.info(f"Existing tables: {existing_tables}")
        
        # Create tables that don't exist
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                logger.info(f"Creating table: {table.name}")
                table.create(engine)
            else:
                logger.info(f"Table {table.name} already exists")
        
        logger.info("Database initialization complete")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def run_migrations():
    """Run database migrations as a standalone function"""
    logger.info("Running database migrations")
    migration_manager = MigrationManager(engine)
    migration_manager.upgrade()
    logger.info("Database migrations complete")