import "./App.css";
import React, { useState } from "react";

// Inline styles for simplicity
const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "2rem",
    fontFamily: "sans-serif",
  },
  status: {
    marginBottom: "1rem",
    fontSize: "1.5rem",
  },
  board: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 100px)",
    gridGap: "5px",
  },
  square: {
    width: "100px",
    height: "100px",
    backgroundColor: "#fff",
    border: "1px solid #999",
    fontSize: "2rem",
    fontWeight: "bold",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  button: {
    marginTop: "1.5rem",
    padding: "0.5rem 1rem",
    fontSize: "1rem",
    cursor: "pointer",
  },
};

function Square({ value, onClick }) {
  return (
    <button style={styles.square} onClick={onClick}>
      {value}
    </button>
  );
}

function calculateWinner(squares) {
  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
  for (let [a, b, c] of lines) {
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a];
    }
  }
  return null;
}

export default function App() {
  const [history, setHistory] = useState([Array(9).fill(null)]);
  const [step, setStep] = useState(0);
  const xIsNext = step % 2 === 0;
  const squares = history[step];
  const winner = calculateWinner(squares);

  const handleClick = (i) => {
    const current = history.slice(0, step + 1);
    const board = current[current.length - 1].slice();
    if (board[i] || winner) return;
    board[i] = xIsNext ? "X" : "O";
    setHistory([...current, board]);
    setStep(current.length);
  };

  const jumpTo = (move) => {
    setStep(move);
  };

  const moves = history.map((_, move) => {
    const desc = move ? `Go to move #${move}` : "Go to game start";
    return (
      <li key={move}>
        <button style={styles.button} onClick={() => jumpTo(move)}>
          {desc}
        </button>
      </li>
    );
  });

  let status;
  if (winner) {
    status = `Winner: ${winner}`;
  } else if (step === 9) {
    status = "Draw";
  } else {
    status = `Next player: ${xIsNext ? "X" : "O"}`;
  }

  return (
    <div className="container">
      <div className="status">{status}</div>
      <div className="board">
        {squares.map((val, i) => (
          <button
            key={i}
            className="square"
            onClick={() => handleClick(i)}
          >
            {val}
          </button>
        ))}
      </div>
      <ol>
        {history.map((_, move) => (
          <li key={move}>
            <button
              className="move-button"
              onClick={() => jumpTo(move)}
            >
              {move ? `Go to move #${move}` : "Go to game start"}
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}
