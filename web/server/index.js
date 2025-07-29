// web/server/index.js
import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import net from 'net';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename    = fileURLToPath(import.meta.url);
const __dirname     = path.dirname(__filename);
const BRIDGE_PORT   = process.env.BRIDGE_PORT || 3001;
const ENGINE_PORT   = process.env.ENGINE_PORT || 4000;

console.log(`Bridge starting on ws://0.0.0.0:${BRIDGE_PORT}`);
console.log(`Will proxy to engine at tcp://127.0.0.1:${ENGINE_PORT}`);


// 1) Spawn and keep a single TCP connection to the engine
let latestEngineState = null;
const engineSocket = net.connect(ENGINE_PORT, '127.0.0.1', () => {
  console.log(`Connected to engine on port ${ENGINE_PORT}`);
});
engineSocket.on('error', err => console.error('Engine socket error:', err));
engineSocket.on('data', data => {
  for (let line of data.toString().split('\n').filter(Boolean)) {
    console.log('Engine → Bridge:', line);
    // cache the latest state_update
    try {
      const msg = JSON.parse(line);
      if (msg.type === 'state_update') {
        latestEngineState = line;
      }
    } catch (e) {}
    // broadcast to all connected WebSocket clients
    for (let client of wss.clients) {
      if (client.readyState === client.OPEN) client.send(line);
    }
  }
});

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocketServer({ server });

// 2) Handle WebSocket clients
wss.on('connection', ws => {
  console.log('➜ New WebSocket client connected');

  // Send the latest engine state to the new client, if available
  if (latestEngineState) {
    ws.send(latestEngineState);
  }

  ws.on('message', message => {
    console.log('WS → Bridge:', message.toString());
    engineSocket.write(message + '\n');
  });

  ws.on('close', () => {
    console.log('WebSocket client disconnected (but engine stays up)');
  });
});

// 3) (Optionally) serve static build from React
app.use(express.static(path.resolve(__dirname, '../games/tictactoe/dist')));

// 4) Start listening
server.listen(BRIDGE_PORT, () => {
  console.log(`Bridge listening on ws://0.0.0.0:${BRIDGE_PORT}`);
});
