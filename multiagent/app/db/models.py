"""
SQLAlchemy models for the multiagent LLM system.
Defines the database schema for storing results and related data.
"""

import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base


class Result(Base):
    """
    Model for storing query results.
    """
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, index=True, nullable=False)
    query = Column(Text, nullable=False)
    user_id = Column(String(255), index=True, nullable=True)
    workflow = Column(String(255), index=True, nullable=False)
    result = Column(JSON, nullable=True)  # Stores the complete result data
    status = Column(String(50), index=True, nullable=False, default="processing")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Create indexes for common queries
    __table_args__ = (
        Index("ix_results_query_workflow", "query", "workflow"),
        Index("ix_results_status_created_at", "status", "created_at"),
    )
    
    # Relationships
    executions = relationship("AgentExecution", back_populates="result")


class AgentExecution(Base):
    """
    Model for storing agent execution details.
    Records the execution of specific agents as part of a workflow.
    """
    __tablename__ = "agent_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("results.id"), nullable=False)
    agent_id = Column(String(255), index=True, nullable=False)
    input_data = Column(JSON, nullable=True)  # Input provided to the agent
    output_data = Column(JSON, nullable=True)  # Output produced by the agent
    error = Column(Text, nullable=True)  # Error message if execution failed
    status = Column(String(50), index=True, nullable=False, default="processing")
    execution_time = Column(Float, nullable=True)  # Execution time in seconds
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    result = relationship("Result", back_populates="executions")


class APIRequest(Base):
    """
    Model for logging API requests.
    Useful for monitoring and debugging.
    """
    __tablename__ = "api_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(UUID, default=uuid.uuid4, unique=True, nullable=False)
    method = Column(String(10), nullable=False)
    path = Column(String(255), nullable=False)
    user_id = Column(String(255), index=True, nullable=True)
    payload = Column(JSON, nullable=True)
    response_status = Column(Integer, nullable=True)
    response_time = Column(Float, nullable=True)  # Response time in seconds
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("ix_api_requests_method_path", "method", "path"),
        Index("ix_api_requests_created_at", "created_at"),
    )


class AgentMetrics(Base):
    """
    Model for storing agent performance metrics.
    """
    __tablename__ = "agent_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(255), index=True, nullable=False, unique=True)
    total_executions = Column(Integer, nullable=False, default=0)
    successful_executions = Column(Integer, nullable=False, default=0)
    failed_executions = Column(Integer, nullable=False, default=0)
    total_execution_time = Column(Float, nullable=False, default=0.0)  # Total execution time in seconds
    avg_execution_time = Column(Float, nullable=False, default=0.0)  # Average execution time in seconds
    last_executed = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class User(Base):
    """
    Model for user data.
    For managing API access and authentication.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class APIKey(Base):
    """
    Model for storing API keys.
    """
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_name = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)