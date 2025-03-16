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
            logger.info(f"Client disconnected: {client_id}")
    
    async def broadcast(self, message: Dict[str, Any], client_id: Optional[str] = None) -> None:
        """
        Broadcasts messages to connected clients.
        
        Args:
            message: The message to broadcast
            client_id: Optional client ID to send to a specific client
        """
        # Add timestamp if not present
        if "time" not in message:
            message["time"] = datetime.utcnow().isoformat()
        
        # Convert to JSON
        message_json = json.dumps(message)
        
        # Send to specific client if specified
        if client_id:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_text(message_json)
                    logger.debug(f"Message sent to client {client_id}")
                except Exception as e:
                    logger.error(f"Error sending message to client {client_id}: {str(e)}")
                    await self.disconnect(client_id)
            return
        
        # Broadcast to all clients
        disconnected_clients = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

# Initialize connection manager
connection_manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Handles WebSocket connections.
    
    Args:
        websocket: The WebSocket connection
    """
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
                        # Store task subscription for client
                        # This would typically involve a more complex subscription system
                        logger.info(f"Client {client_id} subscribed to task {task_id}")
                        
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