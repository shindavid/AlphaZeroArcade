from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connection: list[WebSocket] = []
        self.clients_ids: list[str] = []

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connection.append(websocket)
        self.client_ids.append(client_id)
        await self.broadcast(f"Client #{client_id} has left the room", clients_ids=self.clients_ids)

    async def disconnect(self, websocket, client_id):
        self.active_connections.remove(websocket)
        self.clients_ids.remove(client_id)
        await self.broadcast(f"Client #{client_id} has left the room", clients_ids=self.clients_ids)
    
    async def send_personal_message(self, message: str, clients_ids: list[str], websocket: WebSocket):
        await websocket.send_json({"message": message, "clients_ids": clients_ids})
    
    async def broadcast(self, message: str, clients_ids: list[str]):
        for connection in self.active_connections:
            await connection.send_json({"message": message, "clients_ids": clients_ids})
