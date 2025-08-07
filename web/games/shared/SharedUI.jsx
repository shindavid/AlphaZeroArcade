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
  // resultCodes will either be null, or a string of length N, where N is the length of playerNames
  // and seatAssignments.
  //
  // If resultCodes is null, it means the game is still ongoing.
  //
  // If resultCodes is a string, it contains the result codes for each player. These are either
  // 'W' for win, 'L' for loss, or 'D' for draw.
  //
  // For the W and D codes, we display "Winner!" or "Draw!" next to the player name.
  // ...existing code...
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
            let resultCode = resultCodes && typeof resultCodes === "string" && resultCodes.length === playerNames.length ? resultCodes[i] : null;
            return (
              <div key={seat + i} style={{ display: 'flex', alignItems: 'center' }}>
                {renderResultText(resultCode)}
                <span style={{ marginLeft: 8 }} dangerouslySetInnerHTML={{ __html: seat }} />: <span dangerouslySetInnerHTML={{ __html: escapeHtml(playerNames[i]) }} />
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
