import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def verify_and_update_results_table():
    """
    Verify and update the results table schema.
    Adds retry_count and last_error columns if they don't exist.
    """
    # Get database URL from environment variable
    database_url = os.environ.get(
        'DATABASE_URL', 
        'postgresql://postgres:postgres@db:5432/multiagent'
    )
    
    # Create engine using the database URL
    engine = create_engine(database_url)
    
    # Create a session
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Use raw SQL to check column existence
        columns_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'results'
        """)
        
        columns = session.execute(columns_query)
        existing_columns = [col[0] for col in columns]
        
        print("Existing columns in results table:")
        print(existing_columns)
        
        # Begin a transaction
        with engine.begin() as connection:
            # Add retry_count column
            if 'retry_count' not in existing_columns:
                print("Adding retry_count column...")
                connection.execute(text("""
                    ALTER TABLE results 
                    ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0
                """))
                print("retry_count column added successfully.")
            
            # Add last_error column
            if 'last_error' not in existing_columns:
                print("Adding last_error column...")
                connection.execute(text("""
                    ALTER TABLE results 
                    ADD COLUMN last_error TEXT
                """))
                print("last_error column added successfully.")
        
        print("Database schema update completed successfully.")
    
    except Exception as e:
        print(f"An error occurred during schema update: {e}")
        sys.exit(1)
    finally:
        session.close()

def main():
    """
    Main function to run the migration script.
    """
    verify_and_update_results_table()

if __name__ == "__main__":
    main()