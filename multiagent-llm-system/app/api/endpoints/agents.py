from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.deps import get_agent_manager
from app.orchestrator.manager import AgentManager
from app.schemas.agent import AgentList, AgentRequest, AgentResponse


router = APIRouter(tags=["agents"])

@router.get("/agents", response_model=AgentList)
async def list_agents(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, List[str]]:
    """
    List all available agents.
    
    Args:
        agent_manager: Agent manager
        
    Returns:
        List of agent IDs
    """
    return {"agents": list(agent_manager.agents.keys())}

@router.post("/agents/{agent_id}", response_model=AgentResponse)
async def execute_agent(
    request: AgentRequest,
    agent_id: str = Path(..., description="Agent ID"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Execute a specific agent directly.
    
    Args:
        request: Agent request
        agent_id: ID of the agent to execute
        agent_manager: Agent manager
        
    Returns:
        Agent response
    """
    try:
        result = agent_manager.execute_agent(agent_id, request.input_data)
        
        return {
            "agent_id": agent_id,
            "result": result,
            "success": True
        }
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing agent: {str(e)}"
        )