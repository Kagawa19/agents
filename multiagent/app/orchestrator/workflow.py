"""
Workflow definition and execution.
Defines and executes sequences of agent operations to fulfill user requests.
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from app.db.crud.results import crud_result
from app.db.session import SessionLocal
from app.monitoring.tracer import LangfuseTracer
from app.orchestrator.manager import AgentManager


logger = logging.getLogger(__name__)

class Workflow:
    """
    Base class for all workflows in the system.
    A workflow is a sequence of agent operations that fulfill a specific task.
    """
    
    def __init__(self, name: str, agent_manager: AgentManager, tracer: LangfuseTracer):
        """
        Initialize a workflow.
        
        Args:
            name: Name of the workflow
            agent_manager: AgentManager instance
            tracer: LangfuseTracer instance for monitoring
        """
        self.name = name
        self.agent_manager = agent_manager
        self.tracer = tracer
        self.steps = []
        self.description = "Base workflow"
    
    def add_step(self, agent_id: str, input_mapper: Callable[[Dict[str, Any]], Dict[str, Any]], description: str = ""):
        """
        Add a step to the workflow.
        
        Args:
            agent_id: ID of the agent to execute in this step
            input_mapper: Function that maps workflow state to agent input
            description: Description of the step
        """
        self.steps.append({
            "agent_id": agent_id,
            "input_mapper": input_mapper,
            "description": description or f"Execute {agent_id} agent"
        })
    
    def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow from start to finish.
        
        Args:
            initial_input: Initial input data for the workflow
            
        Returns:
            The final result of the workflow
        """
        with self.tracer.trace(f"workflow_{self.name}"):
            logger.info(f"Executing workflow: {self.name}")
            
            # Get the task ID
            task_id = initial_input.get("task_id")
            
            # Initialize workflow state
            state = {"input": initial_input}
            
            # Initialize result data
            result_data = {
                "query": initial_input.get("query", ""),
                "workflow": self.name,
                "steps": [],
                "started_at": datetime.utcnow().isoformat(),
                "processing_time": 0
            }
            
            # Record the workflow start time
            start_time = time.time()
            
            # Create or update the result record in the database
            with SessionLocal() as db:
                if task_id:
                    db_result = crud_result.get_by_task_id(db, task_id=task_id)
                    if not db_result:
                        db_result = crud_result.save_result(
                            db=db,
                            task_id=task_id,
                            query=initial_input.get("query", ""),
                            workflow=self.name,
                            user_id=initial_input.get("user_id"),
                            status="processing"
                        )
                    result_id = db_result.id
                else:
                    # If no task ID, create a database record anyway
                    db_result = crud_result.save_result(
                        db=db,
                        task_id=f"direct-{int(time.time())}",
                        query=initial_input.get("query", ""),
                        workflow=self.name,
                        user_id=initial_input.get("user_id"),
                        status="processing"
                    )
                    result_id = db_result.id
            
            # Execute each step in sequence
            total_steps = len(self.steps)
            for i, step in enumerate(self.steps):
                agent_id = step["agent_id"]
                input_mapper = step["input_mapper"]
                description = step["description"]
                
                logger.info(f"Executing step {i+1}/{total_steps}: {description} (agent: {agent_id})")
                
                # Map state to agent input
                agent_input = input_mapper(state)
                
                # Update progress if task_id is provided
                if task_id:
                    progress = ((i + 1) / total_steps) * 100
                    self._update_progress(task_id, agent_id, progress)
                
                # Execute agent
                step_start_time = time.time()
                with self.tracer.span(f"step_{i}_{agent_id}"):
                    try:
                        # Execute the agent
                        agent_output = self.agent_manager.execute_agent(agent_id, agent_input)
                        
                        # Calculate execution time
                        step_execution_time = time.time() - step_start_time
                        
                        # Save agent execution details to database
                        with SessionLocal() as db:
                            crud_result.save_agent_execution(
                                db=db,
                                result_id=result_id,
                                agent_id=agent_id,
                                input_data=agent_input,
                                output_data=agent_output,
                                status="completed",
                                execution_time=step_execution_time
                            )
                        
                        # Record step details
                        step_data = {
                            "step": i + 1,
                            "agent_id": agent_id,
                            "description": description,
                            "execution_time": step_execution_time,
                            "status": "completed"
                        }
                        result_data["steps"].append(step_data)
                        
                        # Update state with agent output
                        state[agent_id] = agent_output
                        
                        logger.info(f"Step {i+1}/{total_steps} completed (execution time: {step_execution_time:.2f}s)")
                    
                    except Exception as e:
                        logger.error(f"Error executing step {i+1}/{total_steps}: {str(e)}")
                        
                        # Calculate execution time even in case of error
                        step_execution_time = time.time() - step_start_time
                        
                        # Save failed execution to database
                        with SessionLocal() as db:
                            crud_result.save_agent_execution(
                                db=db,
                                result_id=result_id,
                                agent_id=agent_id,
                                input_data=agent_input,
                                error=str(e),
                                status="failed",
                                execution_time=step_execution_time
                            )
                        
                        # Record failed step
                        step_data = {
                            "step": i + 1,
                            "agent_id": agent_id,
                            "description": description,
                            "execution_time": step_execution_time,
                            "status": "failed",
                            "error": str(e)
                        }
                        result_data["steps"].append(step_data)
                        
                        # Update workflow result with error
                        result_data["error"] = str(e)
                        result_data["status"] = "failed"
                        
                        # Save result to database
                        with SessionLocal() as db:
                            crud_result.save_result(
                                db=db,
                                task_id=task_id or f"direct-{int(time.time())}",
                                query=initial_input.get("query", ""),
                                workflow=self.name,
                                result=result_data,
                                user_id=initial_input.get("user_id"),
                                status="failed"
                            )
                        
                        # Re-raise the exception
                        raise
            
            # Calculate total processing time
            total_execution_time = time.time() - start_time
            result_data["processing_time"] = total_execution_time
            result_data["completed_at"] = datetime.utcnow().isoformat()
            result_data["status"] = "completed"
            
            # Final result is the output of the last agent
            last_agent_id = self.steps[-1]["agent_id"]
            if last_agent_id in state:
                # Extract key information from the final result
                final_result = state[last_agent_id]
                
                # Depending on the agent, extract different information
                if last_agent_id == "summarizer":
                    result_data["summary"] = final_result.get("summary", "")
                    result_data["confidence_score"] = final_result.get("confidence_score", 0)
                else:
                    # Default to the full result
                    result_data["result"] = final_result
            else:
                logger.warning(f"Last agent {last_agent_id} output not found in state")
                result_data["result"] = {"error": f"Last agent {last_agent_id} output not found"}
            
            # Add the full state to the result data (useful for debugging)
            result_data["agent_outputs"] = {k: v for k, v in state.items() if k != "input"}
            
            # Save result to database
            with SessionLocal() as db:
                crud_result.save_result(
                    db=db,
                    task_id=task_id or f"direct-{int(time.time())}",
                    query=initial_input.get("query", ""),
                    workflow=self.name,
                    result=result_data,
                    user_id=initial_input.get("user_id"),
                    status="completed"
                )
            
            logger.info(f"Workflow {self.name} execution completed (total time: {total_execution_time:.2f}s)")
            return result_data
    
    def _update_progress(self, task_id: str, current_step: str, progress: float) -> None:
        """
        Update the progress of a task.
        
        Args:
            task_id: ID of the task
            current_step: Current step being executed
            progress: Progress percentage (0-100)
        """
        from app.worker.tasks import update_progress
        
        try:
            update_progress.delay(
                task_id=task_id,
                status="processing",
                progress=int(progress),
                current_step=current_step
            )
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
    
    def get_definition(self) -> Dict[str, Any]:
        """
        Get the workflow definition.
        
        Returns:
            Dictionary describing the workflow
        """
        steps = []
        for i, step in enumerate(self.steps):
            steps.append({
                "step": i + 1,
                "agent_id": step["agent_id"],
                "description": step["description"]
            })
        
        return {
            "id": self.name,
            "name": self.name.title(),
            "description": self.description,
            "steps": steps,
            "total_steps": len(self.steps)
        }


class ResearchWorkflow(Workflow):
    """
    Standard research workflow that uses all three agents.
    Researches a topic, analyzes the findings, and summarizes the results.
    """
    
    def __init__(self, agent_manager: AgentManager, tracer: LangfuseTracer):
        """
        Initialize the research workflow.
        
        Args:
            agent_manager: AgentManager instance
            tracer: LangfuseTracer instance for monitoring
        """
        super().__init__("research", agent_manager, tracer)
        self.description = "Comprehensive research workflow that searches for information, analyzes it, and creates a summary"
        
        # Define the workflow steps
        
        # Step 1: Research
        self.add_step(
            agent_id="researcher",
            input_mapper=lambda state: {
                "query": state["input"]["query"]
            },
            description="Search for information across the web"
        )
        
        # Step 2: Analyze
        self.add_step(
            agent_id="analyzer",
            input_mapper=lambda state: {
                "query": state["input"]["query"],
                "information": state["researcher"]["processed_information"]
            },
            description="Analyze and extract key insights from research"
        )
        
        # Step 3: Summarize
        self.add_step(
            agent_id="summarizer",
            input_mapper=lambda state: {
                "query": state["input"]["query"],
                "analysis_results": state["analyzer"]["analysis_results"]
            },
            description="Create a concise summary of the findings"
        )


class DirectResearchWorkflow(Workflow):
    """
    Simpler research workflow that skips analysis and goes straight to summarization.
    Useful for simpler queries or when faster response is needed.
    """
    
    def __init__(self, agent_manager: AgentManager, tracer: LangfuseTracer):
        """
        Initialize the direct research workflow.
        
        Args:
            agent_manager: AgentManager instance
            tracer: LangfuseTracer instance for monitoring
        """
        super().__init__("direct_research", agent_manager, tracer)
        self.description = "Streamlined research workflow that searches for information and creates a summary directly"
        
        # Define the workflow steps
        
        # Step 1: Research
        self.add_step(
            agent_id="researcher",
            input_mapper=lambda state: {
                "query": state["input"]["query"]
            },
            description="Search for information across the web"
        )
        
        # Step 2: Summarize (directly from research)
        self.add_step(
            agent_id="summarizer",
            input_mapper=lambda state: {
                "query": state["input"]["query"],
                "analysis_results": state["researcher"]["processed_information"]
            },
            description="Create a concise summary of the findings"
        )


class WorkflowManager:
    """
    Manages and executes workflows based on user requests.
    Provides an interface for the API to interact with workflows.
    """
    
    def __init__(self, agent_manager: AgentManager, tracer: LangfuseTracer):
        """
        Initialize the workflow manager.
        
        Args:
            agent_manager: AgentManager instance
            tracer: LangfuseTracer instance for monitoring
        """
        self.agent_manager = agent_manager
        self.tracer = tracer
        self.workflows = {}
        
        # Register workflows
        self._register_workflows()
    
    def _register_workflows(self) -> None:
        """
        Register all available workflows.
        Creates and registers workflow instances.
        """
        # Standard research workflow
        self.workflows["research"] = ResearchWorkflow(
            agent_manager=self.agent_manager,
            tracer=self.tracer
        )
        
        # Direct research workflow
        self.workflows["direct_research"] = DirectResearchWorkflow(
            agent_manager=self.agent_manager,
            tracer=self.tracer
        )
        
        # Additional workflows can be registered here
        logger.info(f"Registered workflows: {', '.join(self.workflows.keys())}")
    
    def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a workflow by ID.
        
        Args:
            workflow_id: ID of the workflow to execute
            input_data: Input data for the workflow
            
        Returns:
            The workflow's output
            
        Raises:
            KeyError: If the workflow does not exist
        """
        if workflow_id not in self.workflows:
            raise KeyError(f"Workflow {workflow_id} does not exist")
            
        workflow = self.workflows[workflow_id]
        return workflow.execute(input_data)
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """
        Get a workflow by ID.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            The workflow or None if not found
        """
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all available workflows.
        
        Returns:
            List of workflow definitions
        """
        return [workflow.get_definition() for workflow in self.workflows.values()]