# multiagent/app/db/migrations/versions/002_add_agent_execution_columns.py
import logging
from sqlalchemy import text, inspect

# Migration version
version = 2

logger = logging.getLogger(__name__)

def upgrade(engine):
    """
    Add missing id column to the agent_executions table and ensure proper column types.
    """
    inspector = inspect(engine)

    # Check if the agent_executions table exists
    if "agent_executions" not in inspector.get_table_names():
        logger.warning("agent_executions table not found, skipping migration")
        return

    # Get existing columns
    existing_columns = [col["name"] for col in inspector.get_columns("agent_executions")]

    with engine.begin() as conn:
        # Check and add id column if it doesn't exist
        if "id" not in existing_columns:
            logger.info("Adding id column to agent_executions table")
            conn.execute(text("""
                ALTER TABLE agent_executions 
                ADD COLUMN id SERIAL PRIMARY KEY
            """))
        else:
            logger.info("id column already exists in agent_executions table")

        # Ensure output_data column exists and is JSONB
        if "output_data" not in existing_columns:
            logger.info("Adding output_data column to agent_executions table")
            conn.execute(text("""
                ALTER TABLE agent_executions 
                ADD COLUMN output_data JSONB
            """))
        else:
            logger.info("output_data column already exists")

        # Additional column checks can be added here if needed
        logger.info("Completed migration for agent_executions table")