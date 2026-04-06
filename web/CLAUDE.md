# CLAUDE.md — Web Player Architecture

The web player stack connects a browser-based UI to the C++ game engine:

```
Browser (React) ←── WebSocket ──→ Bridge (Node.js) ←── TCP ──→ C++ Engine
```

## Bridge (`web/bridge/index.js`)

A lightweight Node.js/Express process. Connects as a TCP client to the C++ engine and
exposes a WebSocket server to the browser.

- Engine messages → forwarded to all connected WebSocket clients; the last message per
  `cache_key` is cached so late-joining clients receive a full state replay on connect.
- WebSocket messages → forwarded to the engine over TCP as newline-terminated JSON.

## Frontend (`web/games/{game}/`)

Per-game React apps built with Vite. `c4`, `hex`, and `othello` each have their own app.
Shared components live in `web/games/shared/` (e.g., `GameTreePanel`, `GameTreeNode`).
The frontend connects to the bridge WebSocket and renders game state received as JSON.

## Message Types (Engine → Browser)

Payloads sent from the C++ engine have a `type` field and a `cache_key`:

| Type | Meaning |
|---|---|
| `start_game` | New game has begun |
| `action_request` | Engine is waiting for the human player to move |
| `state_update` | Board state has changed |
| `game_end` | Game is over |
| `tree_node` | Search tree node data (for analysis view) |

## Launch Flow

1. Binary starts with `--type web` for the human seat.
2. `WebManager` (C++ singleton) launches the bridge and Vite frontend as child processes.
3. `WebManager` listens on a TCP port; bridge connects.
4. `WebManager` sends game state updates; bridge fans them out over WebSocket to the browser.
5. Browser sends move selections; bridge forwards them to the engine.

See `cpp/CLAUDE.md` for the C++ side (`WebPlayer`, `WebPlayerGenerator`, `WebManager`).
