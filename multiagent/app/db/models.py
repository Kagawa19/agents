"""
SQLAlchemy models for the multiagent LLM system.
Defines the database schema for storing results and related data.
"""
from datetime import datetime
import uuid
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text,
    UniqueConstraint, text, CheckConstraint
)

from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from multiagent.app.db.base import Base
import json

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
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)
    celery_task_id = Column(String(255), nullable=True, index=True)
    query = Column(Text, nullable=False)
    user_id = Column(String(255), nullable=True, index=True)
    workflow = Column(String(255), nullable=False, index=True)
    result = Column(JSONB, nullable=True)
    status = Column(
        String(50), 
        nullable=False, 
        default="processing",
        index=True,
        server_default=text("'processing'"),
    )
    retry_count = Column(Integer, default=0, nullable=False, server_default=text("0"))
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=text("NOW()"), nullable=False)
    updated_at = Column(DateTime, server_default=text("NOW()"), nullable=False, onupdate=datetime.utcnow)
    
    
    # Create indexes for common queries
    __table_args__ = (
        CheckConstraint(
            "status IN ('submitted', 'processing', 'completed', 'failed', 'pending')", 
            name="check_result_status"
        ),
        Index("idx_results_query_workflow", "query", "workflow"),
        Index("idx_results_status_created_at", "status", "created_at"),
        Index("idx_results_user_id_created_at", "user_id", "created_at"),
        Index("idx_results_celery_task_id", "celery_task_id"),
    )
    
    # Relationships
    executions = relationship("AgentExecution", back_populates="result", cascade="all, delete-orphan")
    
    @validates('result')
    def validate_result(self, key, value):
        """Ensure result is JSON serializable"""
        if value is None:
            return value
            
        # If it's already a string or dictionary, we're good
        if isinstance(value, (str, dict)):
            return value
            
        # Try to convert to JSON, fallback to string representation
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return {"data": str(value), "error": "Result was not JSON serializable"}
    
    def __repr__(self):
        return f"<Result(id={self.id}, task_id={self.task_id}, status={self.status})>"

class AgentExecution(Base):
    """
    Model for storing agent execution details.
    Records the execution of specific agents as part of a workflow.
    """
    __tablename__ = "agent_executions"
    
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey("results.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_id = Column(String(255), nullable=False, index=True)
    input_data = Column(JSONB, nullable=True)  # Input provided to the agent
    output_data = Column(JSONB, nullable=True)  # Output produced by the agent
    error = Column(Text, nullable=True)  # Error message if execution failed
    status = Column(
        String(50), 
        nullable=False, 
        default="processing",
        server_default=text("'processing'"),
        index=True
    )
    execution_time = Column(Float, nullable=True)  # Execution time in seconds
    started_at = Column(DateTime, server_default=text("NOW()"), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Add constraint to ensure status is valid
    __table_args__ = (
        CheckConstraint(
            "status IN ('processing', 'completed', 'failed')", 
            name="check_agent_execution_status"
        ),
        Index("idx_agent_executions_agent_id_status", "agent_id", "status"),
    )
    
    # Relationships
    result = relationship("Result", back_populates="executions")
    
    @validates('input_data', 'output_data')
    def validate_json_data(self, key, value):
        """Ensure input and output data are JSON serializable"""
        if value is None:
            return value
            
        # If it's already a string or dictionary, we're good
        if isinstance(value, (str, dict)):
            return value
            
        # Try to convert to JSON, fallback to string representation
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return {"data": str(value), "error": f"{key} was not JSON serializable"}

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
    user_id = Column(String(255), nullable=True, index=True)
    payload = Column(JSONB, nullable=True)
    response_status = Column(Integer, nullable=True)
    response_time = Column(Float, nullable=True)  # Response time in seconds
    created_at = Column(DateTime, server_default=text("NOW()"), nullable=False)
    
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
    agent_id = Column(String(255), nullable=False, index=True)
    task_type = Column(String(255), nullable=False)
    success_rate = Column(Float, nullable=False, default=1.0)  # Success rate (0-1)
    average_latency = Column(Float, nullable=True)  # Avg latency in seconds
    total_executions = Column(Integer, nullable=False, default=0)
    last_executed_at = Column(DateTime, nullable=True)  # Last execution timestamp
    additional_data = Column(JSONB, nullable=False, default={})  # Additional metadata
    
    __table_args__ = (
        Index("idx_agent_metrics_agent_task", "agent_id", "task_type"),
        Index("idx_agent_metrics_last_executed_at", "last_executed_at"),
    )