"""Add missing columns to the results table"""

import logging
from sqlalchemy import text, inspect

# Migration version
version = 1

logger = logging.getLogger(__name__)

def upgrade(engine):
    """
    Add retry_count and last_error columns to the results table.
    """
    inspector = inspect(engine)
    
    # Check if the results table exists
    if "results" not in inspector.get_table_names():
        logger.warning("Results table not found, skipping migration")
        return
    
    # Get existing columns
    existing_columns = [col["name"] for col in inspector.get_columns("results")]
    
    with engine.begin() as conn:
        # Add retry_count column if it doesn't exist
        if "retry_count" not in existing_columns:
            logger.info("Adding retry_count column to results table")
            conn.execute(text("ALTER TABLE results ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"))
        else:
            logger.info("retry_count column already exists")
        
        # Add last_error column if it doesn't exist
        if "last_error" not in existing_columns:
            logger.info("Adding last_error column to results table")
            conn.execute(text("ALTER TABLE results ADD COLUMN last_error TEXT"))
        else:
            logger.info("last_error column already exists")