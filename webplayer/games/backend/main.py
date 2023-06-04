from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from connection_manager import ConnectionManager


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(
                message=data,
                clients_ids=manager.clients_ids,
                websocket=websocket
            )
            await manager.broadcast(
                message="Client ID @{client_id} says {data}",
                clients_ids=manager.clients_ids
            )
    except WebSocketDisconnect:
        await manager.disconnect(websocket=websocket, client_id=client_id)



if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=4455)
