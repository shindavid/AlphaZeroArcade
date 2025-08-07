
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

// Store last message of each type, in order received

let lastByType = {}; // type -> [counter, msg]
let msgIndex = 0;

const engineSocket = net.connect(ENGINE_PORT, '127.0.0.1', () => {
  console.log(`Connected to engine on port ${ENGINE_PORT}`);
});
engineSocket.on('error', err => console.error('Engine socket error:', err));

engineSocket.on('data', data => {
  for (let line of data.toString().split('\n').filter(Boolean)) {
    console.log('Engine → Bridge:', line);
    let msg;
    try {
      msg = JSON.parse(line);
    } catch (e) {
      continue;
    }
    const msg_type = msg.type;
    if (!msg_type) continue;
    if (msg_type === 'start_game') {
      lastByType = {};
      msgIndex = 0;
    }
    // Store or update last message of this type as [counter, msg]
    lastByType[msg_type] = [msgIndex++, line];
    // Broadcast to all connected WebSocket clients
    for (let client of wss.clients) {
      if (client.readyState === client.OPEN) client.send(line);
    }
  }
});

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocketServer({ server });


wss.on('connection', ws => {
  console.log('➜ New WebSocket client connected');
  // Replay each stored message, lexicographically by [counter, msg]
  Object.values(lastByType)
    .sort()
    .forEach(([_, msg]) => ws.send(msg));
  ws.on('message', message => {
    console.log('WS → Bridge:', message.toString());
    engineSocket.write(message + '\n');
  });
  ws.on('close', () => {
    console.log('WebSocket client disconnected (but engine stays up)');
  });
});

server.listen(BRIDGE_PORT, () => {
  console.log(`Bridge listening on ws://0.0.0.0:${BRIDGE_PORT}`);
});
