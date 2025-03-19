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

        # Get all existing indexes
        existing_indexes = set()
        for table_name in existing_tables:
            indexes = inspector.get_indexes(table_name)
            for index in indexes:
                existing_indexes.add(index['name'])
        
        logger.info(f"Existing indexes: {existing_indexes}")

        # Ensure tables exist before creating
        for table in Base.metadata.tables.values():
            table_name = table.name

            if table_name not in existing_tables:
                logger.info(f"Creating table: {table_name}")
                table.create(engine)
            else:
                logger.info(f"Table {table_name} already exists, skipping creation")

            # Ensure indexes exist before creating
            for index in table.indexes:
                index_name = index.name
                if index_name and index_name not in existing_indexes:
                    try:
                        logger.info(f"Creating index: {index_name}")
                        index.create(engine)
                    except Exception as index_error:
                        logger.error(f"Error creating index {index_name}: {index_error}")
                else:
                    logger.info(f"Index {index_name} already exists, skipping creation")

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