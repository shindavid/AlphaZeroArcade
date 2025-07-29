// web/games/tictactoe/src/App.jsx
import React, { useEffect, useState, useRef } from 'react';
import './App.css';

export default function App() {
  const [board, setBoard] = useState(Array(9).fill(null));
  const [turn, setTurn] = useState('X');
  const [loading, setLoading] = useState(true);
  const [gameEnd, setGameEnd] = useState(null); // { result: 'win'|'draw', winner: 'X'|'O' }
  const socketRef = useRef(null);

  // Must be set by Vite via .env.development
  const port = import.meta.env.VITE_BRIDGE_PORT;
  if (!port) {
    // render an error in the UI if port isn’t defined
    return (
      <div style={{ padding: '2rem', color: 'red' }}>
        ERROR: VITE_BRIDGE_PORT is not defined.<br />
        Make sure you restarted the dev server after writing web/.env.development
      </div>
    );
  }

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
        // msg.payload.board is a 9‑char string, e.g. "X_O_XO_OX"
        const str = msg.payload.board;
        const arr = Array.from(str).map(ch =>
          ch === '_' ? null : ch
        );
        setBoard(arr);
        setTurn(msg.payload.turn);
        setLoading(false);
        setGameEnd(null);
      } else if (msg.type === 'game_end') {
        setGameEnd(msg.payload);
      }
    };

    return () => ws.close();
  }, [port]);


  const handleClick = i => {
    if (gameEnd) return;
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (board[i]) return;
    ws.send(JSON.stringify({ type: 'make_move', payload: { index: i } }));
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

  // Dedicated message area and always-visible action button
  // Fix board shifting: wrap status bar and board in a fixed-height flex column
  const handleResign = () => {
    // End the game as a loss for the current player
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'resign' }));
    // Optionally, setGameEnd({ result: 'win', winner: turn === 'X' ? 'O' : 'X' });
  };

  return (
    <div className="container" style={{ minHeight: '600px', justifyContent: 'flex-start' }}>
      <div className="status-bar" style={{ marginBottom: '1.5em' }}>
        <span className="status-message-area">
          {gameEnd
            ? (gameEnd.result === 'draw'
                ? <>Game&nbsp;over:&nbsp;<b>Draw</b></>
                : <>Game&nbsp;over:&nbsp;<b>{gameEnd.winner} wins</b>!</>
              )
            : <>Next: <b>{turn}</b></>
          }
        </span>
      </div>
      <div className="board">
        {board.map((v, i) => (
          <button
            key={i}
            className={`square${v === null && !gameEnd ? ' legal-move' : ''}`}
            onClick={() => handleClick(i)}
            disabled={!!gameEnd || v !== null}
          >{v}</button>
        ))}
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
