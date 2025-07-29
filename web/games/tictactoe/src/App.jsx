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

  // Always render .status, reserve space for both states
  // Always render .status, reserve space for both states and for the button
  return (
    <div className="container">
      <div className="status-bar">
        {gameEnd ? (
          <div className="status-row">
            <span className="status-message">
              {gameEnd.result === 'draw' ? (
                <>Game&nbsp;over: <b>Draw</b></>
              ) : (
                <>Game&nbsp;over: <b>{gameEnd.winner}</b> wins!</>
              )}
            </span>
            <button className="move-button status-btn" onClick={handleNewGame}>New&nbsp;Game</button>
          </div>
        ) : (
          <div className="status-row status-row-center">
            <span>Next: {turn}</span>
            {/* Invisible button to reserve space and keep alignment */}
            <button className="move-button status-btn status-btn-invisible">New&nbsp;Game</button>
          </div>
        )}
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
    </div>
  );
}
