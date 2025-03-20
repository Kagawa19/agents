# api/websocket.py
from typing import Dict, Any, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
import logging
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_subscriptions: Dict[str, List[str]] = {}  # task_id: [client_ids]
    
    async def connect(self, websocket: WebSocket) -> str:
        """
        Connects a WebSocket client.
        
        Args:
            websocket: The WebSocket connection
            
        Returns:
            str: The client identifier
        """
        # Accept the connection
        await websocket.accept()
        
        # Generate a client ID
        client_id = str(uuid.uuid4())
        
        # Store the connection
        self.active_connections[client_id] = websocket
        
        logger.info(f"Client connected: {client_id}")
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "client_id": client_id,
            "time": datetime.utcnow().isoformat(),
            "message": "Connected to multiagent system"
        }))
        
        return client_id
    
    async def disconnect(self, client_id: str) -> None:
        """
        Disconnects a client.
        
        Args:
            client_id: The client ID to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Remove client from task subscriptions
            for task_id, clients in list(self.task_subscriptions.items()):
                if client_id in clients:
                    clients.remove(client_id)
                    if not clients:
                        del self.task_subscriptions[task_id]
            
            logger.info(f"Client disconnected: {client_id}")
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcasts task status update to subscribed clients.
        
        Args:
            message: The task status message to broadcast
        """
        # Ensure message has a timestamp
        if "time" not in message:
            message["time"] = datetime.utcnow().isoformat()
        
        # Add message type for WebSocket clients
        message["type"] = "task_update"
        
        # Convert to JSON
        message_json = json.dumps(message)
        
        # Get task ID to find subscribed clients
        task_id = message.get("task_id")
        
        if task_id:
            # Send to clients subscribed to this task
            if task_id in self.task_subscriptions:
                disconnected_clients = []
                for client_id in self.task_subscriptions[task_id]:
                    if client_id in self.active_connections:
                        try:
                            await self.active_connections[client_id].send_text(message_json)
                        except Exception as e:
                            logger.error(f"Error sending message to client {client_id}: {str(e)}")
                            disconnected_clients.append(client_id)
                
                # Clean up disconnected clients
                for client_id in disconnected_clients:
                    await self.disconnect(client_id)
            else:
                logger.debug(f"No clients subscribed to task {task_id}")
    
    async def subscribe_to_task(self, client_id: str, task_id: str) -> None:
        """
        Subscribe a client to a specific task's updates.
        
        Args:
            client_id: The client to subscribe
            task_id: The task to subscribe to
        """
        if task_id not in self.task_subscriptions:
            self.task_subscriptions[task_id] = []
        
        if client_id not in self.task_subscriptions[task_id]:
            self.task_subscriptions[task_id].append(client_id)
            logger.info(f"Client {client_id} subscribed to task {task_id}")

# Initialize connection manager
connection_manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None) -> None:
    """
    Handles WebSocket connections.
    
    Args:
        websocket: The WebSocket connection
        client_id: Optional pre-defined client ID
    """
    if not client_id:
        client_id = await connection_manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            
            try:
                # Parse JSON message
                message = json.loads(data)
                
                # Handle message based on type
                message_type = message.get("type", "unknown")
                
                if message_type == "ping":
                    # Respond to ping with pong
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "time": datetime.utcnow().isoformat()
                    }))
                elif message_type == "subscribe":
                    # Handle subscription to task updates
                    task_id = message.get("task_id")
                    if task_id:
                        # Subscribe client to task updates
                        await connection_manager.subscribe_to_task(client_id, task_id)
                        
                        # Send acknowledgment
                        await websocket.send_text(json.dumps({
                            "type": "subscribed",
                            "task_id": task_id,
                            "time": datetime.utcnow().isoformat()
                        }))
                else:
                    # Unknown message type
                    logger.warning(f"Unknown message type from client {client_id}: {message_type}")
                
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client {client_id}")
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
    finally:
        # Clean up on disconnect
        await connection_manager.disconnect(client_id)