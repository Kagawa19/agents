"""
Workflow definition and execution.
Defines and executes sequences of agent operations to fulfill user requests.
"""

import logging
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


from multiagent.app.db.results import crud_result
from multiagent.app.db.session import SessionLocal
from multiagent.app.monitoring.tracer import LangfuseTracer
from multiagent.app.orchestrator.manager import AgentManager
import asyncio
import logging
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any
import time
from datetime import datetime
from typing import Dict, Any

import traceback

from multiagent.app.db.results import crud_result
from multiagent.app.db.session import SessionLocal


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
    


    async def _execute_agent_async(self, agent_id: str, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent with comprehensive async support.
        
        Args:
            agent_id: ID of the agent to execute
            agent_input: Input data for the agent
        
        Returns:
            Fully resolved and serializable agent output
        """
        try:
            # Attempt to get the specific agent
            agent = self.agent_manager.get_agent(agent_id)
            
            # Comprehensive async execution strategy
            async def safe_agent_execute():
                try:
                    # Priority 1: Explicit async method
                    if hasattr(agent, 'async_execute'):
                        result = await agent.async_execute(agent_input)
                    
                    # Priority 2: Run standard method in thread pool
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, 
                            agent.execute, 
                            agent_input
                        )
                    
                    # Ensure result is serializable
                    try:
                        json.dumps(result)
                    except TypeError:
                        result = str(result)
                    
                    return result
                
                except Exception as e:
                    logger.error(f"Safe agent execution error: {e}")
                    raise
            
            # Execute and await the result
            result = await safe_agent_execute()
            
            return result
        
        except Exception as e:
            logger.error(f"Agent async execution failed: {e}")
            raise

    async def async_execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async wrapper for the existing execute method.
        Allows workflows to be executed asynchronously.
        
        Args:
            initial_input: Initial input data for the workflow
            
        Returns:
            The final result of the workflow
        """
        return await self.execute(initial_input)

    

    async def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow from start to finish.
        
        Args:
            initial_input: Initial input data for the workflow
            
        Returns:
            The final result of the workflow
        """
        # Create a span for the workflow execution
        span = self.tracer.span(name=f"workflow_{self.name}")
        
        try:
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
                try:
                    agent_input = input_mapper(state)
                except Exception as e:
                    logger.error(f"Error in input mapping for agent {agent_id}: {str(e)}")
                    # Create a more informative error message with state info
                    error_detail = f"Input mapping error: {str(e)}. State keys: {list(state.keys())}"
                    if agent_id in state:
                        agent_state_type = type(state[agent_id]).__name__
                        error_detail += f", {agent_id} type: {agent_state_type}"
                    
                    result_data["error"] = error_detail
                    result_data["status"] = "failed"
                    
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
                    
                    span.update(output={"status": "failed", "error": error_detail})
                    raise ValueError(error_detail)
                
                # Update progress if task_id is provided
                if task_id:
                    progress = ((i + 1) / total_steps) * 100
                    self._update_progress(task_id, agent_id, progress)
                
                # Execute agent
                step_start_time = time.time()
                step_span = self.tracer.span(name=f"step_{i}_{agent_id}")
                
                try:
                    # Execute the agent (with proper async handling)
                    agent_output = await self._execute_agent_async(agent_id, agent_input)
                    
                    # Ensure agent_output is not a coroutine
                    if asyncio.iscoroutine(agent_output):
                        logger.warning(f"Agent {agent_id} returned a coroutine instead of a result")
                        agent_output = await agent_output  # Await the coroutine to get the actual result
                    
                    # Ensure agent_output is a dictionary
                    if not isinstance(agent_output, dict):
                        if isinstance(agent_output, str):
                            try:
                                # Try to parse as JSON if it's a string
                                agent_output = json.loads(agent_output)
                            except json.JSONDecodeError:
                                # If it's not valid JSON, wrap it in a dictionary
                                agent_output = {"result": agent_output, "status": "completed"}
                        else:
                            # For any other type, wrap it in a dictionary
                            agent_output = {"result": str(agent_output), "status": "completed"}
                    
                    logger.info(f"Agent '{agent_id}' output: {agent_output}")
                    
                    # Log the state before updating
                    logger.info(f"State before update: {state}")
                    
                    # Update state with agent output
                    state[agent_id] = agent_output
                    
                    # Log the state after updating
                    logger.info(f"State after update: {state}")
                    
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
                    
                    # Update step span with success
                    step_span.update(output={"status": "completed", "execution_time": step_execution_time})
                    
                    logger.info(f"Step {i+1}/{total_steps} completed (execution time: {step_execution_time:.2f}s)")
                
                except Exception as e:
                    logger.error(f"Error executing step {i+1}/{total_steps}: {str(e)}")
                    step_execution_time = time.time() - step_start_time
                    step_span.update(output={"status": "failed", "error": str(e), "execution_time": step_execution_time})
                    
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
                    
                    step_data = {
                        "step": i + 1,
                        "agent_id": agent_id,
                        "description": description,
                        "execution_time": step_execution_time,
                        "status": "failed",
                        "error": str(e)
                    }
                    result_data["steps"].append(step_data)
                    
                    result_data["error"] = str(e)
                    result_data["status"] = "failed"
                    
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
                    
                    span.update(output={"status": "failed", "error": str(e)})
                    raise
            
            # Calculate total processing time
            total_execution_time = time.time() - start_time
            result_data["processing_time"] = total_execution_time
            result_data["completed_at"] = datetime.utcnow().isoformat()
            result_data["status"] = "completed"
            
            # Log the final state before extracting the result
            logger.info(f"Final state: {state}")
            
            # Final result is the output of the last agent
            last_agent_id = self.steps[-1]["agent_id"]
            if last_agent_id in state:
                # Log the specific agent state
                logger.info(f"Last agent '{last_agent_id}' state: {state[last_agent_id]}")
                
                # Extract key information from the final result
                final_result = state[last_agent_id]
                
                # Ensure final_result is a dictionary
                if not isinstance(final_result, dict):
                    if isinstance(final_result, str):
                        try:
                            # Try to parse as JSON if it's a string
                            final_result = json.loads(final_result)
                        except json.JSONDecodeError:
                            # If not valid JSON, create a basic result dictionary
                            final_result = {"result": final_result}
                    else:
                        # For any other type, create a basic result dictionary
                        final_result = {"result": str(final_result)}
                
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
            
            # Update workflow span with success
            span.update(output={"status": "completed", "processing_time": total_execution_time})
            
            logger.info(f"Workflow {self.name} execution completed (total time: {total_execution_time:.2f}s)")
            return result_data
            
        except Exception as e:
            # Update workflow span with error
            span.update(output={"status": "failed", "error": str(e)})
            # Re-raise the exception
            raise
    
    def _update_progress(self, task_id: str, current_step: str, progress: float) -> None:
        """
        Update the progress of a task.
        
        Args:
            task_id: ID of the task
            current_step: Current step being executed
            progress: Progress percentage (0-100)
        """
        from multiagent.app.worker.tasks import update_progress
        
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
                "information": state["researcher"]["processed_information"],
                "search_web": state["input"].get("enable_analyzer_search", False),
                "num_results": state["input"].get("analyzer_search_results", 3)
            },
            description="Analyze and extract key insights from research"
        )
        
        # Step 3: Summarize - FIXED version with safe access pattern
        self.add_step(
            agent_id="summarizer",
            input_mapper=lambda state: {
                "query": state["input"]["query"],
                "analysis_results": self._safely_get_analysis_results(state)
            },
            description="Create a concise summary of the findings"
        )
    
    def _safely_get_analysis_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely extract analysis results from the state, handling cases where 
        the analyzer output might be a string instead of a dictionary.
        
        Args:
            state: Current workflow state
            
        Returns:
            Analysis results dictionary
        """
        analyzer_data = state.get("analyzer")
        
        # If analyzer data is missing
        if analyzer_data is None:
            logger.error("Analyzer data is missing from workflow state")
            return {"error": "Missing analyzer data"}
            
        # If analyzer data is a dictionary (normal case)
        if isinstance(analyzer_data, dict):
            return analyzer_data.get("analysis_results", {"error": "No analysis results in output"})
            
        # If analyzer data is a string (serialized JSON)
        if isinstance(analyzer_data, str):
            try:
                parsed_data = json.loads(analyzer_data)
                if isinstance(parsed_data, dict):
                    return parsed_data.get("analysis_results", {"error": "No analysis results in parsed output"})
                else:
                    logger.error(f"Parsed analyzer data is not a dictionary: {type(parsed_data)}")
                    return {"error": "Invalid analyzer data format after parsing"}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analyzer data as JSON: {e}")
                return {"error": "Failed to parse analyzer output"}
        
        # If analyzer data is some other type
        logger.error(f"Unexpected analyzer data type: {type(analyzer_data)}")
        return {"error": f"Unexpected analyzer data type: {type(analyzer_data)}"}


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
        
        # Step 2: Summarize (directly from research) - FIXED version with safe access
        self.add_step(
            agent_id="summarizer",
            input_mapper=lambda state: {
                "query": state["input"]["query"],
                "analysis_results": self._safely_get_research_results(state)
            },
            description="Create a concise summary of the findings"
        )
    
    def _safely_get_research_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely extract research results from the state, handling cases where 
        the researcher output might be a string instead of a dictionary.
        
        Args:
            state: Current workflow state
            
        Returns:
            Research results dictionary
        """
        researcher_data = state.get("researcher")
        
        # If researcher data is missing
        if researcher_data is None:
            logger.error("Researcher data is missing from workflow state")
            return {"error": "Missing researcher data"}
            
        # If researcher data is a dictionary (normal case)
        if isinstance(researcher_data, dict):
            return researcher_data.get("processed_information", {"error": "No processed information in output"})
            
        # If researcher data is a string (serialized JSON)
        if isinstance(researcher_data, str):
            try:
                parsed_data = json.loads(researcher_data)
                if isinstance(parsed_data, dict):
                    return parsed_data.get("processed_information", {"error": "No processed information in parsed output"})
                else:
                    logger.error(f"Parsed researcher data is not a dictionary: {type(parsed_data)}")
                    return {"error": "Invalid researcher data format after parsing"}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse researcher data as JSON: {e}")
                return {"error": "Failed to parse researcher output"}
        
        # If researcher data is some other type
        logger.error(f"Unexpected researcher data type: {type(researcher_data)}")
        return {"error": f"Unexpected researcher data type: {type(researcher_data)}"}

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
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a workflow by ID with async support.
        
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
        
        try:
            result = await workflow.execute(input_data)
            
            # Ensure result is serializable
            try:
                json.dumps(result)
            except TypeError:
                result = str(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            raise
    
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