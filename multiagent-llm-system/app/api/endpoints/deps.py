from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.monitoring.tracer import LangfuseTracer, get_tracer
from app.orchestrator.manager import AgentManager
from app.orchestrator.workflow import WorkflowManager


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for database session.
    Yields a SQLAlchemy session and ensures it's closed after use.
    
    Yields:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_agent_manager(tracer: LangfuseTracer = Depends(get_tracer)) -> AgentManager:
    """
    Dependency for AgentManager.
    Creates and returns an initialized AgentManager instance.
    
    Args:
        tracer: Langfuse tracer for monitoring
        
    Returns:
        AgentManager instance
    """
    from app.core.config import settings
    
    manager = AgentManager(settings, tracer)
    manager.initialize()
    return manager


def get_workflow_manager(
    agent_manager: AgentManager = Depends(get_agent_manager),
    tracer: LangfuseTracer = Depends(get_tracer)
) -> WorkflowManager:
    """
    Dependency for WorkflowManager.
    Creates and returns a WorkflowManager instance.
    
    Args:
        agent_manager: AgentManager for the workflow
        tracer: Langfuse tracer for monitoring
        
    Returns:
        WorkflowManager instance
    """
    return WorkflowManager(agent_manager, tracer)