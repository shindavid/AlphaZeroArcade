import Board from './Board';

import { useState } from 'react';

export default function Game() {
    const NUMBER_OF_ROWS = 6
    const NUMBER_OF_COLUMNS = 7
    const [history, setHistory] = useState([
      Array(NUMBER_OF_COLUMNS).fill(null).map(
        () => Array(NUMBER_OF_ROWS).fill(null))
    ]);

    const [currentMove, setCurrentMove] = useState(0);
    const redIsNext = currentMove % 2 === 0;
    const currentColumns = history[currentMove].map(cols => cols.slice());
    
    function handlePlay(nextColumns) {
      const nextHistory = [...history.slice(0, currentMove + 1), nextColumns];
      setHistory(nextHistory);
      setCurrentMove(nextHistory.length - 1);
    }
  
    function jumpTo(nextMove) {
      setCurrentMove(nextMove);
    }
  
    const moves = history.map((_, move) => {
      let description;
      if (move > 0) {
        description = 'Go to move #' + move;
      } else {
        description = 'Go to game start';
      }
      return (
        <li key={move}>
          <button onClick={() => jumpTo(move)}>{description}</button>
        </li>
      );
    });
  
    return (
      <div className="game">
        <div className="game-board">
          <Board redIsNext={redIsNext} columns={currentColumns} onPlay={handlePlay} />
        </div>
        <div className="game-info">
          <ol>{moves}</ol>
        </div>
      </div>
    );
  }
  