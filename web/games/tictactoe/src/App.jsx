import React, { useEffect, useState, useRef } from 'react';
import './App.css';
import '../../shared/shared.css';
import { PortError, Loading, StatusBar, ActionButtons } from '../../shared/SharedUI';
import { handleResign as sharedHandleResign, handleNewGame as sharedHandleNewGame } from '../../shared/handlers';

export default function App() {
  const [board, setBoard] = useState(Array(9).fill('_'));
  const [turn, setTurn] = useState(null);
  const [loading, setLoading] = useState(true);
  const [gameEnd, setGameEnd] = useState(null); // { result: 'win'|'draw', winner: 'X'|'O' }
  const socketRef = useRef(null);

  // Must be set by Vite via .env.development
  const port = import.meta.env.VITE_BRIDGE_PORT;
  if (!port) {
    return <PortError port={port} />;
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
        setGameEnd(null);
      } else if (msg.type === 'game_end') {
        setBoardHelper(msg.payload.board);
        setGameEnd(msg.payload);
      }
      setLoading(false);
    };

    return () => ws.close();
  }, [port]);

  const handleClick = i => {
    if (gameEnd) return;
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (board[i] !== '_') return;
    ws.send(JSON.stringify({ type: 'make_move', payload: { index: i } }));
  };

  const handleNewGame = () => {
    sharedHandleNewGame(socketRef, setLoading, setGameEnd);
  };

  if (loading) {
    return <Loading />;
  }

  // Dedicated message area and always-visible action button
  // Fix board shifting: wrap status bar and board in a fixed-height flex column
  const handleResign = () => {
    sharedHandleResign(socketRef);
  };

  return (
    <div className="container" style={{ minHeight: '600px', justifyContent: 'flex-start' }}>
      <StatusBar gameEnd={gameEnd} turn={turn} />
      <div className="board">
        {board.map((v, i) => (
          <button
            key={i}
            className={`square${v === '_' && !gameEnd ? ' legal-move' : ''}`}
            onClick={() => handleClick(i)}
            disabled={!!gameEnd || v !== '_'}
          >{v === '_' ? '' : v}</button>
        ))}
      </div>
      <ActionButtons
        onResign={handleResign}
        onNewGame={handleNewGame}
        gameEnd={gameEnd}
        loading={loading}
      />
    </div>
  );
}
