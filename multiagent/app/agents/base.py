"""
Base agent implementation.
Defines the interface for all agents in the system.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain.agents import Agent as LangChainAgent

from app.monitoring.tracer import LangfuseTracer
from app.monitoring.metrics import track_agent_execution


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Provides common functionality for agent initialization, execution, and monitoring.
    """
    
    def __init__(self, agent_id: str, tracer: LangfuseTracer):
        """
        Initialize the base agent with an ID and tracer.
        
        Args:
            agent_id: Unique identifier for the agent
            tracer: LangfuseTracer instance for monitoring
        """
        self.agent_id = agent_id
        self.tracer = tracer
        self.langchain_agent = None
        self.initialized = False
        logger.info(f"Agent {agent_id} created")
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the agent with configuration parameters.
        Must be implemented by subclasses.
        
        Args:
            config: Dictionary containing agent-specific configuration
        """
        pass
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main functionality.
        Must be implemented by subclasses.
        
        Args:
            input_data: Input data for the agent to process
            
        Returns:
            Dict containing the results of the agent's execution
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate the input data before execution.
        Can be overridden by subclasses to add specific validation.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True
    
    def log_execution(self, input_data: Dict[str, Any], output_data: Dict[str, Any], execution_time: float) -> None:
        """
        Log the execution details using the tracer.
        
        Args:
            input_data: Input data that was processed
            output_data: Output data that was produced
            execution_time: Execution time in seconds
        """
        with self.tracer.span(f"{self.agent_id}_execution"):
            self.tracer.log_event(
                event_type="agent_execution",
                event_data={
                    "agent_id": self.agent_id,
                    "input": input_data,
                    "output": output_data,
                    "execution_time": execution_time
                }
            )
            
            # Track metrics
            success = "error" not in output_data
            track_agent_execution(self.agent_id, "success" if success else "failure", execution_time)
    
    def process_error(self, error: Exception, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an error that occurred during execution.
        
        Args:
            error: The exception that was raised
            input_data: Input data that was being processed
            
        Returns:
            Error response data
        """
        error_message = str(error)
        error_type = error.__class__.__name__
        
        logger.error(f"Error in agent {self.agent_id}: {error_message}", exc_info=True)
        
        self.tracer.log_event(
            event_type="agent_error",
            event_data={
                "agent_id": self.agent_id,
                "error_type": error_type,
                "error_message": error_message,
                "input": input_data
            }
        )
        
        return {
            "error": error_message,
            "error_type": error_type,
            "agent_id": self.agent_id,
            "status": "failed"
        }
    
    def safe_execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely execute the agent with error handling and timing.
        
        Args:
            input_data: Input data for the agent to process
            
        Returns:
            Results of the agent's execution or error response
        """
        try:
            # Validate input
            if not self.validate_input(input_data):
                return {
                    "error": "Invalid input data",
                    "error_type": "ValidationError",
                    "agent_id": self.agent_id,
                    "status": "failed"
                }
            
            # Check if agent is initialized
            if not self.initialized:
                raise RuntimeError(f"Agent {self.agent_id} is not initialized")
            
            # Start timer
            start_time = time.time()
            
            # Execute the agent
            result = self.execute(input_data)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Add metadata to result
            result["agent_id"] = self.agent_id
            result["execution_time"] = execution_time
            result["status"] = "success"
            
            # Log execution
            self.log_execution(input_data, result, execution_time)
            
            return result
        except Exception as e:
            # Process the error
            return self.process_error(e, input_data)