// Auto-generated via: ./py/tools/make_scaffold_for_web_game.py -g Hex -o

import './Hex.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

export default class HexApp extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: null,
    };
  }

  renderBoard() {
    const N = 11;
    const HEX_SIZE = 30; // px, controls overall scale
    const HEX_WIDTH = Math.sqrt(3) * HEX_SIZE;
    const HEX_HEIGHT = 2 * HEX_SIZE;
    const board = this.state.board;
    const legalMoves = this.state.legalMoves || [];

    // Helper to get board index for (row, col)
    const idx = (row, col) => row * N + col;

    // SVG points for a hex centered at (cx, cy)
    function hexPoints(cx, cy, size) {
      const points = [];
      for (let i = 0; i < 6; ++i) {
        const angle = Math.PI / 3 * i + Math.PI / 6;
        points.push([
          cx + size * Math.cos(angle),
          cy + size * Math.sin(angle)
        ]);
      }
      return points.map(p => p.join(",")).join(" ");
    }

    // SVG rendering
    const hexes = [];
    for (let row = 0; row < N; ++row) {
      for (let col = 0; col < N; ++col) {
        const i = idx(row, col);
        const cell = board[i];
        // Hex center coordinates
        const cx = HEX_WIDTH * col + HEX_WIDTH / 2 + HEX_WIDTH * row / 2;
        const cy = HEX_SIZE * 1.5 * row + HEX_SIZE;

        // Cell color
        let fill = "#fff";
        if (cell === "R") fill = "var(--hex-red, #e44)";
        if (cell === "B") fill = "var(--hex-blue, #24f)";

        // Legal move highlight
        const isLegal = legalMoves.includes(i) && this.gameActive();

        hexes.push(
          <g key={i}>
            <polygon
              className={
                "hex-cell" +
                (cell === "R" ? " hex-red" : "") +
                (cell === "B" ? " hex-blue" : "") +
                (isLegal ? " hex-legal" : "")
              }
              points={hexPoints(cx, cy, HEX_SIZE)}
              onClick={isLegal ? () => this.sendMove(i) : undefined}
              style={{ pointerEvents: isLegal ? "auto" : "none" }}
            />
            {/* Optionally, add a disc for R/B */}
            {cell === "R" || cell === "B" ? (
              <circle
                className={cell === "R" ? "hex-disc-red" : "hex-disc-blue"}
                cx={cx}
                cy={cy}
                r={HEX_SIZE * 0.5}
              />
            ) : null}
          </g>
        );
      }
    }

    // Axes labels
    const colLabels = [];
    for (let col = 0; col < N; ++col) {
      const label = String.fromCharCode(97 + col); // a, b, c, ...
      const x = HEX_WIDTH * col + HEX_WIDTH / 2 + HEX_WIDTH * 0.5 * col;
      colLabels.push(
        <text
          key={"col" + col}
          x={x}
          y={HEX_SIZE * 0.5}
          className="hex-label"
          textAnchor="middle"
        >{label}</text>
      );
    }
    const rowLabels = [];
    for (let row = 0; row < N; ++row) {
      const y = HEX_SIZE * 1.5 * row + HEX_SIZE + 5;
      rowLabels.push(
        <text
          key={"row" + row}
          x={HEX_WIDTH * 0.2}
          y={y}
          className="hex-label"
          textAnchor="end"
        >{row + 1}</text>
      );
    }

    // Board outline (red/blue borders)
    // For simplicity, just draw two polylines for the borders
    const outlineRed = [];
    for (let col = 0; col < N; ++col) {
      const cx = HEX_WIDTH * col + HEX_WIDTH / 2 + HEX_WIDTH * col / 2;
      const cy = HEX_SIZE;
      outlineRed.push([cx, cy]);
    }
    for (let row = 1; row < N; ++row) {
      const cx = HEX_WIDTH * (N - 1) + HEX_WIDTH / 2 + HEX_WIDTH * row / 2;
      const cy = HEX_SIZE * 1.5 * row + HEX_SIZE;
      outlineRed.push([cx, cy]);
    }
    const outlineBlue = [];
    for (let row = 0; row < N; ++row) {
      const cx = HEX_WIDTH / 2 + HEX_WIDTH * row / 2;
      const cy = HEX_SIZE * 1.5 * row + HEX_SIZE;
      outlineBlue.push([cx, cy]);
    }
    for (let col = 1; col < N; ++col) {
      const cx = HEX_WIDTH * col + HEX_WIDTH / 2 + HEX_WIDTH * (N - 1) / 2;
      const cy = HEX_SIZE * 1.5 * (N - 1) + HEX_SIZE;
      outlineBlue.push([cx, cy]);
    }

    // SVG viewBox size
    const width = HEX_WIDTH * N + HEX_WIDTH;
    const height = HEX_SIZE * 1.5 * N + HEX_SIZE * 2;

    return (
      <svg
        className="hex-board"
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
      >
        {/* Board outline */}
        <polyline
          className="hex-outline-red"
          points={outlineRed.map(p => p.join(",")).join(" ")}
          fill="none"
        />
        <polyline
          className="hex-outline-blue"
          points={outlineBlue.map(p => p.join(",")).join(" ")}
          fill="none"
        />
        {/* Axes labels */}
        {colLabels}
        {rowLabels}
        {/* Hex cells */}
        {hexes}
      </svg>
    );
  }
}
