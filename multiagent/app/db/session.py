import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from multiagent.app.core.config import settings
from multiagent.app.db.base import Base

logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(
    settings.DATABASE_URI,
    pool_pre_ping=True,  # Test connections before using them
    echo=settings.DATABASE_ECHO  # Log SQL statements if DEBUG is True
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db() -> None:
    """
    Initialize the database by creating all tables.
    """
    try:
        # Create all tables defined in models
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _create_default_configurations():
    """
    Create default configurations if they don't exist.
    This method can be expanded to include initial data setup.
    """
    # Import models here to avoid circular imports
    from multiagent.app.db.models import ProviderConfig
    
    db = SessionLocal()
    try:
        # Example: Create a default provider configuration if not exists
        # This is just a placeholder - customize based on your specific needs
        existing_config = db.query(ProviderConfig).first()
        if not existing_config:
            default_config = ProviderConfig(
                provider_id="default_openai",
                config={"model": "gpt-4"},
                is_active=True
            )
            db.add(default_config)
            db.commit()
            logger.info("Created default provider configuration")
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating default configurations: {e}")
    finally:
        db.close()

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