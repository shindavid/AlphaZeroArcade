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

// --- CACHE STORAGE ---
// Snapshot types: we only need the latest one (e.g. 'start_game', 'player_info')
let lastByType = {};  // msg_type -> [msgIndex, RawString]
// History types: we need ALL of them in order (e.g. 'state_update')
let gameHistory = [];  // Array of [msgIndex, RawString]
// Global counter to maintain order across both types
let msgIndex = 0;

// Define which message types should be accumulated instead of overwritten
const HISTORY_TYPES = new Set(['state_update']);

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
      continue;
    }

    const msg_type = msg.type;
    if (!msg_type) continue;

    // 1. RESET Logic
    // When a new game starts, clear all previous history and snapshots
    if (msg_type === 'start_game') {
      lastByType = {};
      gameHistory = [];
      msgIndex = 0;
    }

    // 2. STORAGE Logic
    const entry = [msgIndex++, line]; // Store [Order, RawString]

    if (HISTORY_TYPES.has(msg_type)) {
      // Accumulate history messages
      gameHistory.push(entry);
    } else {
      // Overwrite snapshot messages
      lastByType[msg_type] = entry;
    }

    // 3. BROADCAST Logic
    for (let client of wss.clients) {
      if (client.readyState === client.OPEN) client.send(line);
    }
  }
});

wss.on('connection', ws => {
  console.log('➜ New WebSocket client connected');

  // MERGE & SORT
  // Combine single-item snapshots with the full history list
  const allMessages = [
    ...Object.values(lastByType),
    ...gameHistory
  ];

  // Sort by the original msgIndex to ensure perfect playback sequence
  allMessages.sort((a, b) => a[0] - b[0]);

  // REPLAY
  allMessages.forEach(([_, line]) => ws.send(line));

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
