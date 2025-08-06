import React, { useEffect, useState, useRef } from 'react';
import './App.css';

// Connect4 board dimensions
const ROWS = 6;
const COLS = 7;

export default function App() {
  const [board, setBoard] = useState(Array(ROWS * COLS).fill(null));
  const [turn, setTurn] = useState('X');
  const [loading, setLoading] = useState(true);
  const [gameEnd, setGameEnd] = useState(null); // { result: 'win'|'draw', winner: 'X'|'O' }
  const [legalMoves, setLegalMoves] = useState([]); // array of legal column indices
  const socketRef = useRef(null);

  const port = import.meta.env.VITE_BRIDGE_PORT;
  if (!port) {
    return (
      <div style={{ padding: '2rem', color: 'red' }}>
        ERROR: VITE_BRIDGE_PORT is not defined.<br />
        Make sure you restarted the dev server after writing web/.env.development
      </div>
    );
  }

  const setBoardHelper = (str) => {
    setBoard(Array.from(str));
  };

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    socketRef.current = ws;

    ws.onopen = () => console.log(`✅ WS connected to ${port}`);
    ws.onerror = e => console.error('🔴 WS error', e);
    ws.onmessage = e => {
      let msg;
      try { msg = JSON.parse(e.data) }
      catch (err) { return console.error('Bad JSON', err); }

      if (msg.type === 'state_update') {
        setBoardHelper(msg.payload.board);
        setTurn(msg.payload.turn);
        setLegalMoves(msg.payload.legal_moves || []);
        setGameEnd(null);
      } else if (msg.type === 'game_end') {
        setBoardHelper(msg.payload.board);
        setGameEnd(msg.payload);
        setLegalMoves([]);
      }
      setLoading(false);
    };

    return () => ws.close();
  }, [port]);

  // Click a column to make a move
  const handleColumnClick = col => {
    if (gameEnd) return;
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!legalMoves.includes(col)) return;
    ws.send(JSON.stringify({ type: 'make_move', payload: { index: col } }));
  };

  const handleNewGame = () => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'new_game' }));
    setLoading(true);
    setGameEnd(null);
  };

  if (loading) {
    return (
      <div className="container">
        <div className="status">Loading game state...</div>
      </div>
    );
  }

  const handleResign = () => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'resign' }));
  };

  // Render the Connect4 board as a 6x7 grid
  const renderBoard = () => {
    const grid = [];
    for (let row = 0; row < ROWS; ++row) {
      for (let col = 0; col < COLS; ++col) {
        const idx = row * COLS + col;
        const cell = board[idx];
        const isLegal = legalMoves.includes(col);
        let cellClass = "empty";
        if (cell === "R") cellClass = "red";
        else if (cell === "Y") cellClass = "yellow";
        grid.push(
          <div
            key={idx}
            className={`connect4-cell ${cellClass}`}
            onClick={isLegal ? () => handleColumnClick(col) : undefined}
            role={isLegal ? "button" : undefined}
            tabIndex={isLegal ? 0 : -1}
            aria-label={isLegal ? `Play in column ${col + 1}` : undefined}
          />
        );
      }
    }
    return grid;
  };

  return (
    <div className="container" style={{ minHeight: '600px', justifyContent: 'flex-start' }}>
      <div className="status-bar" style={{ marginBottom: '1.5em' }}>
        <span className="status-message-area">
          {gameEnd
            ? gameEnd.msg
            : <>Next: <b>{turn}</b></>
          }
        </span>
      </div>
      <div className="board connect4-board">
        {renderBoard()}
      </div>
      <div className="button-row">
        <button
          className="status-action-btn"
          onClick={handleResign}
          disabled={!!gameEnd || loading}
        >
          Resign
        </button>
        <button
          className="status-action-btn"
          onClick={handleNewGame}
          disabled={!gameEnd || loading}
        >
          New Game
        </button>
      </div>
    </div>
  );
}
