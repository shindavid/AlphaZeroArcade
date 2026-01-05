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

let lastByTypeIndex = {}; // (type, index) -> [msgIndex, msgLine]
let msgIndex = 0;

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocketServer({ server });

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
      console.error('JSON Parse Error:', e, 'Line:', line);
      continue;
    }

    if (!msg.payloads || !Array.isArray(msg.payloads)) {
      continue;
    }

    for (const payload of msg.payloads) {
      if (!payload.type) continue;

      if (payload.type === 'start_game') {
        console.log('Resetting Game History...');
        lastByTypeIndex = {};
        msgIndex = 0;
      }

      let key;
      if (payload.type === 'tree_node') {
        key = `${payload.type}:${payload.index}`;
      } else {
        key = `${payload.type}:-1`;
      }
      const payloadStr = JSON.stringify(payload);
      lastByTypeIndex[key] = [msgIndex++, payloadStr];

      for (let client of wss.clients) {
        if (client.readyState === client.OPEN) {
          console.log('Bridge → WS Client:', payloadStr);
          client.send(payloadStr);
        }
      }
    }
  }
});

wss.on('connection', ws => {
  console.log('➜ New WebSocket client connected');

  const allMessages = Object.values(lastByTypeIndex);
  allMessages.sort((a, b) => a[0] - b[0]);

  // replay messages
  allMessages.forEach(([_, jsonString]) => ws.send(jsonString));

  ws.on('message', message => {
    console.log('WS → Bridge:', message.toString());
    engineSocket.write(message + '\n');
  });

  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });
});

server.listen(BRIDGE_PORT, () => {
  console.log(`Bridge listening on ws://0.0.0.0:${BRIDGE_PORT}`);
});
