import logging
import importlib
import pkgutil
from sqlalchemy import inspect, text, Table, MetaData
from sqlalchemy.engine import Engine
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MigrationManager:
    """Manages database migrations for the multiagent system."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.version_table = "schema_versions"
        self._ensure_version_table()
    
    def _ensure_version_table(self) -> None:
        """Create the schema version table if it doesn't exist."""
        inspector = inspect(self.engine)
        if self.version_table not in inspector.get_table_names():
            with self.engine.begin() as conn:
                conn.execute(text(f"""
                CREATE TABLE {self.version_table} (
                    id SERIAL PRIMARY KEY,
                    version INTEGER NOT NULL,
                    applied_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW() NOT NULL,
                    description TEXT
                )
                """))
                # Insert initial version record
                conn.execute(text(f"INSERT INTO {self.version_table} (version, description) VALUES (0, 'Initial schema version')"))
                logger.info(f"Created {self.version_table} table")
    
    def get_current_version(self) -> int:
        """Get the current schema version from the database."""
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT MAX(version) FROM {self.version_table}"))
            version = result.scalar()
            return version if version is not None else 0
    
    def upgrade(self, target_version: int = None) -> None:
        """
        Upgrade the database schema to the target version.
        If target_version is None, upgrade to the latest version.
        """
        current_version = self.get_current_version()
        logger.info(f"Current database schema version: {current_version}")
        
        # Import all migration modules
        migrations = self._get_migrations()
        
        if not migrations:
            logger.info("No migrations found")
            return
        
        # Determine the target version if not specified
        if target_version is None:
            target_version = max(migrations.keys())
        
        logger.info(f"Target database schema version: {target_version}")
        
        # Apply migrations in order
        for version in sorted(migrations.keys()):
            if version <= current_version:
                continue
            
            if target_version is not None and version > target_version:
                break
                
            migration_module = migrations[version]
            logger.info(f"Applying migration {version}: {migration_module.__doc__}")
            
            try:
                # Apply the migration
                migration_module.upgrade(self.engine)
                
                # Update schema version
                with self.engine.begin() as conn:
                    conn.execute(
                        text(f"INSERT INTO {self.version_table} (version, description) VALUES (:version, :description)"),
                        {"version": version, "description": migration_module.__doc__}
                    )
                
                logger.info(f"Successfully applied migration {version}")
            except Exception as e:
                logger.error(f"Error applying migration {version}: {str(e)}")
                raise
    
    def _get_migrations(self) -> Dict[int, Any]:
        """
        Discover and load all migration modules.
        Returns a dictionary mapping version numbers to migration modules.
        """
        migrations = {}
        
        try:
            # Import the versions package
            versions_package = importlib.import_module("multiagent.app.db.migrations.versions")
            
            # Get all migration modules
            for _, name, is_pkg in pkgutil.iter_modules(versions_package.__path__):
                if is_pkg:
                    continue
                
                # Import the migration module
                migration_module = importlib.import_module(f"multiagent.app.db.migrations.versions.{name}")
                
                # Get the version number from the module
                if hasattr(migration_module, "version"):
                    version = migration_module.version
                    migrations[version] = migration_module
        except ImportError as e:
            logger.error(f"Error loading migrations: {str(e)}")
        
        return migrations
    
    def check_column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        inspector = inspect(self.engine)
        if table_name not in inspector.get_table_names():
            return False
        
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        return column_name in columns