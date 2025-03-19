"""
SQLAlchemy models for the multiagent LLM system.
Defines the database schema for storing results and related data.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
)
from sqlalchemy import DateTime

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from multiagent.app.db.base import Base

class ProviderConfig(Base):
    """
    Model for storing provider configuration.
    Contains settings and configuration for LLM providers.
    """
    __tablename__ = "provider_configs"
    
    id = Column(Integer, primary_key=True)
    provider_id = Column(String(255), unique=True, nullable=False)
    config = Column(JSON, nullable=False, default={})  # Configuration data
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class ProviderCapabilities(Base):
    """
    Model for storing provider capabilities.
    Records what each provider is capable of and how well.
    """
    __tablename__ = "provider_capabilities"
    
    id = Column(Integer, primary_key=True)
    provider_id = Column(Integer, ForeignKey("provider_configs.id"), nullable=False)
    capability_type = Column(String(255), nullable=False)
    capability_value = Column(Float, nullable=False, default=0.0)  # Score between 0-1
    additional_data = Column(JSON, nullable=False, default={})  # Additional capability metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Create composite index
    __table_args__ = (
        Index("idx_provider_capabilities_provider_id_type", "provider_id", "capability_type"),
    )

class ProviderPerformance(Base):
    """
    Model for tracking provider performance metrics.
    Records performance data for each provider.
    """
    __tablename__ = "provider_performances"
    
    id = Column(Integer, primary_key=True)
    provider_id = Column(Integer, ForeignKey("provider_configs.id"), nullable=False)
    model_id = Column(String(255), nullable=True)  # Specific model ID if applicable
    task_type = Column(String(255), nullable=False)
    latency = Column(Float, nullable=False)  # Latency in seconds
    success_rate = Column(Float, nullable=False, default=1.0)  # Success rate (0-1)
    cost = Column(Float, nullable=True)  # Cost in dollars if available
    quality_score = Column(Float, nullable=True)  # Quality score if available (0-1)
    tokens_input = Column(Integer, nullable=True)  # Number of input tokens
    tokens_output = Column(Integer, nullable=True)  # Number of output tokens
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    additional_data = Column(JSON, nullable=False, default={})  # Additional performance metadata
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_provider_performances_provider_task", "provider_id", "task_type"),
        Index("idx_provider_performances_recorded_at", "recorded_at"),
    )

class Result(Base):
    """
    Model for storing query results.
    """
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(255), unique=True, nullable=False)
    celery_task_id = Column(String(255), nullable=True)  # Field for Celery task ID mapping
    query = Column(Text, nullable=False)
    user_id = Column(String(255), nullable=True)
    workflow = Column(String(255), nullable=False)
    result = Column(JSON, nullable=True)  # Stores the complete result data
    status = Column(String(50), nullable=False, default="processing")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Create indexes for common queries
    __table_args__ = (
        Index("idx_results_query_workflow", "query", "workflow"),
        Index("idx_results_status_created_at", "status", "created_at"),
        Index("idx_results_celery_task_id", "celery_task_id"),  # Index for faster lookups
    )
    
    # Relationships
    executions = relationship("AgentExecution", back_populates="result")

class AgentExecution(Base):
    """
    Model for storing agent execution details.
    Records the execution of specific agents as part of a workflow.
    """
    __tablename__ = "agent_executions"
    
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey("results.id"), nullable=False)
    agent_id = Column(String(255), nullable=False)
    input_data = Column(JSON, nullable=True)  # Input provided to the agent
    output_data = Column(JSON, nullable=True)  # Output produced by the agent
    error = Column(Text, nullable=True)  # Error message if execution failed
    status = Column(String(50), nullable=False, default="processing")
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
    
    id = Column(Integer, primary_key=True)
    request_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    method = Column(String(10), nullable=False)
    path = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=True)
    payload = Column(JSON, nullable=True)
    response_status = Column(Integer, nullable=True)
    response_time = Column(Float, nullable=True)  # Response time in seconds
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("idx_api_requests_method_path", "method", "path"),
        Index("idx_api_requests_created_at", "created_at"),
    )

class AgentMetrics(Base):
    """
    Model for storing agent performance metrics.
    """
    __tablename__ = "agent_metrics"
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(255), nullable=False)
    task_type = Column(String(255), nullable=False)
    success_rate = Column(Float, nullable=False, default=1.0)  # Success rate (0-1)
    average_latency = Column(Float, nullable=True)  # Avg latency in seconds
    total_executions = Column(Integer, nullable=False, default=0)
    last_executed_at = Column(DateTime, nullable=True)  # Last execution timestamp
    additional_data = Column(JSON, nullable=False, default={})  # Additional metadata
    
    __table_args__ = (
        Index("idx_agent_metrics_agent_task", "agent_id", "task_type"),
        Index("idx_agent_metrics_last_executed_at", "last_executed_at"),
    )

 
