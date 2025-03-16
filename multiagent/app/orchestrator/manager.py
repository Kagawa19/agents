"""
Agent manager implementation.
Responsible for creating, configuring, and coordinating agents.
"""

import logging
from typing import Dict, Any, Type, Optional

from multiagent.app.agents.base import BaseAgent
from multiagent.app.core.config import Settings
from multiagent.app.monitoring.tracer import LangfuseTracer


logger = logging.getLogger(__name__)

class AgentManager:
    """
    Manages the creation, configuration, and execution of agents.
    Provides an interface for the orchestrator to interact with agents.
    """
    
    def __init__(self, settings: Settings, tracer: LangfuseTracer):
        """
        Initialize the agent manager.
        
        Args:
            settings: Application settings
            tracer: LangfuseTracer instance for monitoring
        """
        self.settings = settings
        self.tracer = tracer
        self.agents: Dict[str, BaseAgent] = {}
        self.tools: Dict[str, Any] = {}
    
    def initialize(self) -> None:
        """
        Initialize all tools and agents.
        Creates tool instances and configures all agent types.
        """
        # Create a span without using context manager
        span = self.tracer.span(name="agent_manager_initialization")
        
        try:
            # Initialize tools
            logger.info("Initializing tools")
            self._initialize_tools()
            
            # Initialize agents
            logger.info("Initializing agents")
            self._initialize_agents()
            
            # Update span with success status
            span.update(output={"status": "success"})
        except Exception as e:
            # Update span with error
            span.update(output={"status": "error", "error": str(e)})
            raise
    
    def _initialize_tools(self) -> None:
        """
        Initialize all tools needed by the agents.
        Creates instances of all required external tools.
        """
        # Import tool classes
        from multiagent.app.tools.serper import SerperTool
        from multiagent.app.tools.jina import JinaTool
        from multiagent.app.tools.openai import OpenAITool
        from multiagent.app.tools.bedrock import BedrockTool
        from multiagent.app.tools.llamaindex import LlamaIndexTool
        
        # Serper Tool
        try:
            self.tools["serper"] = SerperTool(
                api_key=self.settings.SERPER_API_KEY,
                tracer=self.tracer
            )
            logger.info("Initialized SerperTool")
        except Exception as e:
            logger.error(f"Failed to initialize SerperTool: {e}")
            self.tools["serper"] = None
        
        # Jina Tool
        try:
            self.tools["jina"] = JinaTool(
                api_key=self.settings.JINA_API_KEY,
                tracer=self.tracer
            )
            logger.info("Initialized JinaTool")
        except Exception as e:
            logger.error(f"Failed to initialize JinaTool: {e}")
            self.tools["jina"] = None
        
        # OpenAI Tool
        try:
            self.tools["openai"] = OpenAITool(
                api_key=self.settings.OPENAI_API_KEY,
                tracer=self.tracer
            )
            logger.info("Initialized OpenAITool")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAITool: {e}")
            self.tools["openai"] = None
        
        # Bedrock Tool (optional)
        try:
            self.tools["bedrock"] = BedrockTool(
                config=self.settings.get_llm_config("bedrock"),
                tracer=self.tracer
            )
            logger.info("Initialized BedrockTool")
        except Exception as e:
            logger.warning(f"Failed to initialize BedrockTool: {e}")
            self.tools["bedrock"] = None
        
        # LlamaIndex Tool (optional)
        try:
            self.tools["llamaindex"] = LlamaIndexTool(
                config={},
                tracer=self.tracer
            )
            logger.info("Initialized LlamaIndexTool")
        except Exception as e:
            logger.warning(f"Failed to initialize LlamaIndexTool: {e}")
            self.tools["llamaindex"] = None
    
    def _initialize_agents(self) -> None:
        """
        Initialize all agents with their configurations.
        Creates instances of all agent types.
        """
        # Import agent classes
        from multiagent.app.agents.researcher import ResearcherAgent
        from multiagent.app.agents.analyzer import AnalyzerAgent
        from multiagent.app.agents.summarizer import SummarizerAgent
        
        # Researcher Agent
        try:
            researcher = ResearcherAgent(
                agent_id="researcher",
                tracer=self.tracer,
                serper_tool=self.tools["serper"],
                llamaindex_tool=self.tools.get("llamaindex")
            )
            researcher.initialize(self.settings.get_agent_config("researcher"))
            self.agents["researcher"] = researcher
            logger.info("Initialized ResearcherAgent")
        except Exception as e:
            logger.error(f"Failed to initialize ResearcherAgent: {e}")
        
        # Analyzer Agent
        try:
            analyzer = AnalyzerAgent(
                agent_id="analyzer",
                tracer=self.tracer,
                jina_tool=self.tools["jina"],
                openai_tool=self.tools["openai"]
            )
            analyzer.initialize(self.settings.get_agent_config("analyzer"))
            self.agents["analyzer"] = analyzer
            logger.info("Initialized AnalyzerAgent")
        except Exception as e:
            logger.error(f"Failed to initialize AnalyzerAgent: {e}")
        
        # Summarizer Agent
        try:
            summarizer = SummarizerAgent(
                agent_id="summarizer",
                tracer=self.tracer,
                openai_tool=self.tools["openai"],
                bedrock_tool=self.tools.get("bedrock")
            )
            summarizer.initialize(self.settings.get_agent_config("summarizer"))
            self.agents["summarizer"] = summarizer
            logger.info("Initialized SummarizerAgent")
        except Exception as e:
            logger.error(f"Failed to initialize SummarizerAgent: {e}")
    
    def get_agent(self, agent_id: str) -> BaseAgent:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            The requested agent
            
        Raises:
            KeyError: If the agent does not exist
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} does not exist")
        return self.agents[agent_id]
    
    def execute_agent(self, agent_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent with the given input data.
        
        Args:
            agent_id: ID of the agent to execute
            input_data: Input data for the agent
            
        Returns:
            The agent's output
            
        Raises:
            KeyError: If the agent does not exist
        """
        agent = self.get_agent(agent_id)
        
        # Create a span without using context manager
        span = self.tracer.span(
            name=f"execute_agent_{agent_id}",
            input=input_data
        )
        
        try:
            logger.info(f"Executing agent: {agent_id}")
            start_time = __import__('time').time()
            result = agent.execute(input_data)
            execution_time = __import__('time').time() - start_time
            logger.info(f"Agent executed successfully: {agent_id} (execution time: {execution_time:.2f}s)")
            
            # Update agent metrics
            self._update_agent_metrics(agent_id, True, execution_time)
            
            # Update span with success result
            span.update(output=result)
            
            return result
        except Exception as e:
            # Update span with error
            span.update(output={"status": "error", "error": str(e)})
            raise
    
    def _update_agent_metrics(self, agent_id: str, success: bool, execution_time: float) -> None:
        """
        Update metrics for an agent execution.
        
        Args:
            agent_id: ID of the agent
            success: Whether the execution was successful
            execution_time: Execution time in seconds
        """
        try:
            from multiagent.app.db.session import SessionLocal
            from multiagent.app.db.models import AgentMetrics
            from sqlalchemy.sql import func
            from datetime import datetime
            
            with SessionLocal() as db:
                # Get or create agent metrics
                metrics = db.query(AgentMetrics).filter(AgentMetrics.agent_id == agent_id).first()
                
                if metrics:
                    # Update existing metrics
                    metrics.total_executions += 1
                    if success:
                        metrics.successful_executions += 1
                    else:
                        metrics.failed_executions += 1
                    metrics.total_execution_time += execution_time
                    metrics.avg_execution_time = metrics.total_execution_time / metrics.total_executions
                    metrics.last_executed = datetime.utcnow()
                    metrics.updated_at = datetime.utcnow()
                else:
                    # Create new metrics
                    metrics = AgentMetrics(
                        agent_id=agent_id,
                        total_executions=1,
                        successful_executions=1 if success else 0,
                        failed_executions=0 if success else 1,
                        total_execution_time=execution_time,
                        avg_execution_time=execution_time,
                        last_executed=datetime.utcnow()
                    )
                    db.add(metrics)
                
                db.commit()
        except Exception as e:
            logger.error(f"Failed to update agent metrics: {e}")
    
    def get_agent_status(self) -> Dict[str, bool]:
        """
        Get the status of all agents.
        
        Returns:
            Dictionary mapping agent IDs to their availability status
        """
        status = {}
        for agent_id, agent in self.agents.items():
            status[agent_id] = agent is not None
        return status
    
    def get_tool_status(self) -> Dict[str, bool]:
        """
        Get the status of all tools.
        
        Returns:
            Dictionary mapping tool IDs to their availability status
        """
        status = {}
        for tool_id, tool in self.tools.items():
            status[tool_id] = tool is not None
        return status