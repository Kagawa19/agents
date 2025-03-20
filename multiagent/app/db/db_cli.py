"""
Database CLI commands for multiagent system.
"""

import argparse
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for database CLI commands."""
    parser = argparse.ArgumentParser(description="Database management commands")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize the database")
    init_parser.add_argument("--drop", action="store_true", help="Drop existing tables before initializing")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations")
    migrate_parser.add_argument("--target", type=int, help="Target migration version")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "init":
        from multiagent.app.db.session import init_db
        logger.info(f"Initializing database (drop_existing={args.drop})")
        init_db(drop_existing=args.drop)
        logger.info("Database initialization complete")
    
    elif args.command == "migrate":
        from multiagent.app.db.session import run_migrations
        logger.info("Running database migrations")
        run_migrations()
        logger.info("Database migrations complete")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()