import React from 'react';
import './shared.css';

export function PortError({port}) {
  return (
    <div style={{ padding: '2rem', color: 'red' }}>
      ERROR: VITE_BRIDGE_PORT is not defined.<br />
      Make sure you restarted the dev server after writing web/.env.development
    </div>
  );
}

export function Loading() {
  return (
    <div className="container">
      <div className="status">Loading game state...</div>
    </div>
  );
}

export function StatusBar({gameEnd, turn}) {
  return (
    <div className="status-bar" style={{ marginBottom: '1.5em' }}>
      <span className="status-message-area">
        {gameEnd
          ? gameEnd.msg
          : <>Next: <b>{turn}</b></>
        }
      </span>
    </div>
  );
}

export function ActionButtons({onResign, onNewGame, gameEnd, loading}) {
  return (
    <div className="button-row">
      <button
        className="status-action-btn"
        onClick={onResign}
        disabled={!!gameEnd || loading}
      >
        Resign
      </button>
      <button
        className="status-action-btn"
        onClick={onNewGame}
        disabled={!gameEnd || loading}
      >
        New Game
      </button>
    </div>
  );
}
