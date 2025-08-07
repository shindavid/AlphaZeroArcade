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

// Simple HTML escape utility
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function StatusBar({gameEnd, playerNames, seatAssignments}) {
  return (
    <div className="status-bar" style={{ marginBottom: '1.5em' }}>
      <span className="status-message-area">
        {gameEnd ? gameEnd.msg : ""}
      </span>
      {playerNames && seatAssignments && (
        <div style={{ marginTop: '0.5em', textAlign: 'left' }}>
          {seatAssignments.map((seat, i) => (
            <div key={seat + i}>
              {seat}: <span dangerouslySetInnerHTML={{ __html: escapeHtml(playerNames[i]) }} />
            </div>
          ))}
        </div>
      )}
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
