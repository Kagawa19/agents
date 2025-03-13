import json
import logging
from typing import Dict, List

from fastapi import WebSocket, WebSocketDisconnect


logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket connections.
    Handles connection registration, messaging, and disconnection.
    """
    
    def __init__(self):
        """
        Initialize the connection manager.
        """
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket) -> None:
        """
        Register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket) -> None:
        """
        Send a message to a specific WebSocket.
        
        Args:
            message: Message to send
            websocket: WebSocket to send to
        """
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict) -> None:
        """
        Send a message to all connected WebSockets.
        
        Args:
            message: Message to send
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


connection_manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time updates.
    
    Args:
        websocket: WebSocket connection
    """
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Just keep the connection alive
            # We don't expect clients to send messages
            data = await websocket.receive_text()
            logger.debug(f"Received message: {data}")
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)