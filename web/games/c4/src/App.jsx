import React, { useEffect, useState, useRef } from 'react';
import './App.css';

// Connect4 board dimensions
const ROWS = 6;
const COLS = 7;
const ANIMATION_INTERVAL = 60; // ms per row drop

export default function App() {
  const [board, setBoard] = useState(Array(ROWS * COLS).fill('_'));
  const [turn, setTurn] = useState('X');
  const [loading, setLoading] = useState(true);
  const [gameEnd, setGameEnd] = useState(null); // { result: 'win'|'draw', winner: 'X'|'O' }
  const [legalMoves, setLegalMoves] = useState([]); // array of legal column indices
  const [lastAction, setLastAction] = useState(null); // 1-indexed column from backend, or null
  const [animating, setAnimating] = useState(false);
  const [animCol, setAnimCol] = useState(null);
  const [animRow, setAnimRow] = useState(null);
  const [animTargetRow, setAnimTargetRow] = useState(null); // for opponent animation
  const [animDisc, setAnimDisc] = useState(null); // 'R' or 'Y'
  const [animSource, setAnimSource] = useState(null); // 'player' or 'opponent'
  const [lastMoveSentCol, setLastMoveSentCol] = useState(null); // track last move sent by player
  const animTimer = useRef(null);
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
      console.log('WebSocket message received:', e.data);
      let msg;
      try { msg = JSON.parse(e.data) }
      catch (err) { return console.error('Bad JSON', err); }

      if (msg.type === 'state_update') {
        setBoardHelper(msg.payload.board);
        setTurn(msg.payload.turn);
        setLegalMoves(msg.payload.legal_moves || []);
        setGameEnd(null);
        // Animation logic: check for last_action
        let last = msg.payload.last_action;
        if (last && last !== '-') {
          let col = parseInt(last, 10) - 1;
          if (!isNaN(col) && col >= 0 && col < COLS) {
            // If this is the player's last move, don't animate again
            if (lastMoveSentCol === col) {
              setLastMoveSentCol(null); // clear after backend confirms
            } else {
              // Animate opponent's move
              let b = Array.from(msg.payload.board || '');
              // Debug: print board array and col
              console.log('[ANIMATE OPPONENT] board:', b, 'col:', col);
              console.log('[ANIMATE OPPONENT] TEST ROWS:', ROWS);
              let disc = null;
              let row = -1;
              // Find the row where the new disc landed (first non-empty cell from top)
              for (let r = 0; r < ROWS; ++r) {
                let idx = r * COLS + col;
                console.log('[ANIMATE OPPONENT] idx:', idx, 'r:', r, 'b[idx]:', b[idx]);
                if (b[idx] !== '_') {
                  disc = b[idx];
                  row = r;
                  break;
                }
              }
              console.log('[ANIMATE OPPONENT] found disc:', disc, 'at row:', row);
              if (row !== -1 && disc) {
                setAnimating(true);
                setAnimCol(col);
                setAnimRow(0);
                setAnimTargetRow(row);
                setAnimDisc(disc);
                setAnimSource('opponent');
                setLastAction(col);
                if (animTimer.current) clearInterval(animTimer.current);
                let animationDone = false;
                animTimer.current = setInterval(() => {
                  setAnimRow(prev => {
                    if (prev < row) {
                      return prev + 1;
                    } else {
                      if (!animationDone) {
                        animationDone = true;
                        clearInterval(animTimer.current);
                        setAnimating(false);
                        setAnimCol(null);
                        setAnimRow(null);
                        setAnimDisc(null);
                        setAnimSource(null);
                        setLastAction(null);
                        setAnimTargetRow(null);
                      }
                      return prev;
                    }
                  });
                }, ANIMATION_INTERVAL);
              }
            }
          }
        }
      } else if (msg.type === 'game_end') {
        setBoardHelper(msg.payload.board);
        setGameEnd(msg.payload);
        setLegalMoves([]);
        setAnimating(false);
        setAnimCol(null);
        setAnimRow(null);
        setAnimDisc(null);
        setLastAction(null);
      }
      setLoading(false);
    };

    return () => {
      ws.close();
      if (animTimer.current) clearInterval(animTimer.current);
    };
  }, [port]);

  // Click a column to make a move
  const handleColumnClick = col => {
    console.log('handleColumnClick called for col', col);
    if (gameEnd || animating) {
      console.log('Blocked: gameEnd or animating');
      return;
    }
    if (!legalMoves.includes(col)) {
      console.log('Blocked: not a legal move');
      return;
    }
    // Find the lowest empty row in this column
    let row = -1;
    for (let r = ROWS - 1; r >= 0; --r) {
      if (board[r * COLS + col] === '_') {
        row = r;
        break;
      }
    }
    console.log('Computed drop row:', row);
    if (row === -1) {
      console.log('Blocked: column full');
      return; // column full
    }
    // Assume player's disc is 'R' if turn is 'R', else 'Y'
    const disc = turn === 'R' ? 'R' : 'Y';
    console.log('Starting animation: col', col, 'row', row, 'disc', disc);
    setAnimating(true);
    setAnimCol(col);
    setAnimRow(0);
    setAnimDisc(disc);
    setAnimSource('player');
    setLastAction(col);
    setLastMoveSentCol(col);
    // Block input and animate drop
    if (animTimer.current) clearInterval(animTimer.current);
    // Guard: prevent double interval and double callback
    if (animTimer.current) clearInterval(animTimer.current);
    let animationDone = false;
    animTimer.current = setInterval(() => {
      setAnimRow(prev => {
        if (prev < row) {
          return prev + 1;
        } else {
          if (!animationDone) {
            animationDone = true;
            clearInterval(animTimer.current);
            setAnimating(false);
            setAnimCol(null);
            setAnimRow(null);
            setAnimDisc(null);
            setAnimSource(null);
            setLastAction(null);
            // After animation, send move to backend
            console.log('Animation complete, sending move to backend:', col);
            const ws = socketRef.current;
            if (ws && ws.readyState === WebSocket.OPEN) {
              console.log('[SEND] make_move', col);
              ws.send(JSON.stringify({ type: 'make_move', payload: { index: col } }));
            }
          }
          return prev;
        }
      });
    }, ANIMATION_INTERVAL);
  };

  const handleNewGame = () => {
    // Reset animation state and clear timer
    if (animTimer.current) clearInterval(animTimer.current);
    setAnimating(false);
    setAnimCol(null);
    setAnimRow(null);
    setAnimDisc(null);
    setLastAction(null);
    // Send new game request to backend
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    console.log('[SEND] new_game');
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
    console.log('[SEND] resign');
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
        let cellClass = "";
        // During animation, render the cell as empty (no disc) if it's the animated target
        let hideDisc = false;
        if (animating && animCol === col) {
          if (
            animSource === 'player' && animRow !== null && row === animRow && cell === animDisc
          ) {
            hideDisc = true;
          } else if (
            animSource === 'opponent' && animTargetRow !== null && row === animTargetRow && cell === animDisc
          ) {
            hideDisc = true;
          }
        }
        if (!hideDisc) {
          if (cell === "R") cellClass = "red";
          else if (cell === "Y") cellClass = "yellow";
        }
        grid.push(
          <div
            key={idx}
            className={`connect4-cell${cellClass ? ' ' + cellClass : ''}`}
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

  // Render the animated disc overlay
  const renderAnimatedDisc = () => {
    if (!animating || animCol === null || animRow === null || !animDisc) return null;
    // Calculate position: use same cell size as CSS (56px), plus board padding and border
    // .connect4-board: padding: 20px 18px 16px 18px; border: 4px solid
    const left = 18 + 4 + animCol * 56;
    const top = 20 + 4 + animRow * 56;
    const discClass = animDisc === 'R' ? 'red' : 'yellow';
    return (
      <div
        className={`animated-disc ${discClass}`}
        style={{ left, top }}
      />
    );
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
      <div className="board connect4-board" style={{ position: 'relative' }}>
        {renderBoard()}
        {renderAnimatedDisc()}
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
