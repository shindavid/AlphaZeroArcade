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

export function StatusBar({resultCodes, playerNames, seatAssignments}) {
  // Helper to render result text in fixed-width area
  function renderResultText(code) {
    if (code === "W") {
      return <span className="result-text winner">WINNER</span>;
    } else if (code === "D") {
      return <span className="result-text draw">DRAW</span>;
    }
    // For loss or ongoing, show empty fixed-width area
    return <span className="result-text" />;
  }

  return (
    <div className="status-bar" style={{ marginBottom: '1.5em' }}>
      {playerNames && seatAssignments && (
        <div style={{ marginTop: '0.5em', textAlign: 'left' }}>
          {seatAssignments.map((seat, i) => {
            let resultCode = resultCodes ? resultCodes[i] : null;
            return (
              <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
                {renderResultText(resultCode)}
                <span style={{ marginLeft: 8 }}>{seat}</span>: <span>{playerNames[i]}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function ActionButtons({onResign, onNewGame, midGame, loading}) {
  return (
    <div className="button-row">
      <button
        className="status-action-btn"
        onClick={onResign}
        disabled={!midGame || loading}
      >
        Resign
      </button>
      <button
        className="status-action-btn"
        onClick={onNewGame}
        disabled={!!midGame || loading}
      >
        New Game
      </button>
    </div>
  );
}
