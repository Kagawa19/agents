"""
Migration script to update the database schema.
This script updates the 'results' table to ensure better data storage and retrieval.
Run this with SQLAlchemy-Migrate or directly in your PostgreSQL database.
"""
import sys
import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, text, MetaData, Table, Column
from sqlalchemy import Integer, String, Text, DateTime, JSON, Float, Boolean, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CreateIndex, AddConstraint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get database URL from environment
database_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/multiagent')
logger.info(f"Using database URL: {database_url}")

def run_migration():
    """Run the database migration."""
    # Connect to the database
    try:
        engine = create_engine(database_url)
        conn = engine.connect()
        logger.info("Connected to database successfully")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return False
    
    try:
        # Begin transaction
        transaction = conn.begin()
        
        # Check if the results table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'results'
            );
        """))
        
        if not result.scalar():
            logger.error("Results table does not exist, cannot run migration")
            transaction.rollback()
            return False
        
        # Add new columns if they don't exist
        logger.info("Adding new columns to results table...")
        
        # Check which columns we need to add
        result = conn.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'results';
        """))
        existing_columns = [row[0] for row in result]
        
        # Add retry_count column if it doesn't exist
        if 'retry_count' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE results ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0;
            """))
            logger.info("Added retry_count column")
        
        # Add last_error column if it doesn't exist
        if 'last_error' not in existing_columns:
            conn.execute(text("""
                ALTER TABLE results ADD COLUMN last_error TEXT;
            """))
            logger.info("Added last_error column")
        
        # Convert result column to JSONB if needed
        result = conn.execute(text("""
            SELECT data_type FROM information_schema.columns 
            WHERE table_name = 'results' AND column_name = 'result';
        """))
        result_type = result.scalar()
        
        if result_type and result_type.lower() != 'jsonb':
            conn.execute(text("""
                ALTER TABLE results ALTER COLUMN result TYPE JSONB USING result::JSONB;
            """))
            logger.info("Converted result column to JSONB")
        
        # Add missing indexes
        logger.info("Adding additional indexes...")
        
        # Check existing indexes
        result = conn.execute(text("""
            SELECT indexname FROM pg_indexes WHERE tablename = 'results';
        """))
        existing_indexes = [row[0] for row in result]
        
        # Add user_id index
        if 'idx_results_user_id_created_at' not in existing_indexes:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_results_user_id_created_at 
                ON results(user_id, created_at);
            """))
            logger.info("Added idx_results_user_id_created_at index")
        
        # Add status check constraint if it doesn't exist
        result = conn.execute(text("""
            SELECT constraint_name FROM information_schema.table_constraints
            WHERE table_name = 'results' AND constraint_name = 'check_result_status';
        """))
        if not result.scalar():
            try:
                conn.execute(text("""
                    ALTER TABLE results ADD CONSTRAINT check_result_status
                    CHECK (status IN ('submitted', 'processing', 'completed', 'failed', 'pending'));
                """))
                logger.info("Added status check constraint")
            except Exception as e:
                logger.warning(f"Could not add status constraint: {e}")
        
        # Set server defaults for timestamps if needed
        for col_name in ['created_at', 'updated_at']:
            try:
                conn.execute(text(f"""
                    ALTER TABLE results 
                    ALTER COLUMN {col_name} SET DEFAULT NOW();
                """))
                logger.info(f"Set server default for {col_name}")
            except Exception as e:
                logger.warning(f"Could not set server default for {col_name}: {e}")
        
        # Verify all records have a valid status
        conn.execute(text("""
            UPDATE results 
            SET status = 'pending' 
            WHERE status IS NULL OR status NOT IN ('submitted', 'processing', 'completed', 'failed', 'pending');
        """))
        
        # Set not null constraint on task_id if needed
        try:
            conn.execute(text("""
                ALTER TABLE results 
                ALTER COLUMN task_id SET NOT NULL;
            """))
            logger.info("Set NOT NULL constraint on task_id")
        except Exception as e:
            logger.warning(f"Could not set NOT NULL constraint on task_id: {e}")
        
        # Similarly update agent_executions table
        logger.info("Checking agent_executions table...")
        
        # Check if table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'agent_executions'
            );
        """))
        
        if result.scalar():
            # Convert columns to JSONB
            for col_name in ['input_data', 'output_data']:
                try:
                    conn.execute(text(f"""
                        ALTER TABLE agent_executions 
                        ALTER COLUMN {col_name} TYPE JSONB USING {col_name}::JSONB;
                    """))
                    logger.info(f"Converted {col_name} to JSONB")
                except Exception as e:
                    logger.warning(f"Could not convert {col_name} to JSONB: {e}")
            
            # Add status check constraint
            try:
                conn.execute(text("""
                    ALTER TABLE agent_executions ADD CONSTRAINT check_agent_execution_status
                    CHECK (status IN ('processing', 'completed', 'failed'));
                """))
                logger.info("Added status check constraint to agent_executions")
            except Exception as e:
                logger.warning(f"Could not add status constraint to agent_executions: {e}")
            
            # Add index for agent_id and status
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_agent_executions_agent_id_status
                    ON agent_executions(agent_id, status);
                """))
                logger.info("Added index on agent_id and status")
            except Exception as e:
                logger.warning(f"Could not add index on agent_id and status: {e}")
        
        # Commit all changes
        transaction.commit()
        logger.info("Migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        transaction.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    success = run_migration()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)