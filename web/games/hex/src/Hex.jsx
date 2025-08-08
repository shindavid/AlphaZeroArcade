// Auto-generated via: ./py/tools/make_scaffold_for_web_game.py -g Hex -o

import './Hex.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

const BOARD_SIZE = 11;
const HEX_SIZE = 30; // px, controls overall scale
const HEX_WIDTH = Math.sqrt(3) * HEX_SIZE;

export default class HexApp extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: null,
    };
  }

  renderBoard() {
    let board = this.state.board;
    let legalMoves = this.state.legalMoves;

    if (board === null) return null;

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

    const N = BOARD_SIZE;
    // SVG rendering
    const hexes = [];
    for (let row = 0; row < N; ++row) {
      for (let col = 0; col < N; ++col) {
        const i = row * N + col;
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

        // Compute hex corners
        const corners = [];
        for (let k = 0; k < 6; ++k) {
          const angle = Math.PI / 3 * k + Math.PI / 6;
          corners.push([
            cx + HEX_SIZE * Math.cos(angle),
            cy + HEX_SIZE * Math.sin(angle)
          ]);
        }

        // Helper to determine border color for each segment
        function borderColor(k) {
          // k: 0=top-left, 1=top-right, 2=right, 3=bottom-right, 4=bottom-left, 5=left
          // North edge: row==0, k==0 or k==1
          if (row === N - 1 && (k === 0 || k === 1)) return 'var(--hex-red, #e44)';
          // South edge: row==N-1, k==4 or k==5
          if (row === 0 && (k === 4 || k === 5)) return 'var(--hex-red, #e44)';
          // West edge: col==0, k==5 or k==0
          if (col === 0 && (k === 5 || k === 0)) return 'var(--hex-blue, #24f)';
          // East edge: col==N-1, k==2 or k==3
          if (col === N-1 && (k === 2 || k === 3)) return 'var(--hex-blue, #24f)';
          // Otherwise black
          return '#000';
        }

        hexes.push(
          <g key={i}>
            {/* Fill polygon, no stroke */}
            <polygon
              className={
                "hex-cell" +
                (cell === "R" ? " hex-red" : "") +
                (cell === "B" ? " hex-blue" : "") +
                (isLegal ? " hex-legal" : "")
              }
              points={corners.map(p => p.join(",")).join(" ")}
              onClick={isLegal ? () => this.sendMove(i) : undefined}
              style={{ pointerEvents: isLegal ? "auto" : "none" }}
            />
            {/* Draw 6 border lines manually, colored by position */}
            {corners.map((p, k) => {
              const p2 = corners[(k + 1) % 6];
              return (
                <line
                  key={"border" + k}
                  x1={p[0]}
                  y1={p[1]}
                  x2={p2[0]}
                  y2={p2[1]}
                  className="hex-border"
                  stroke={borderColor(k)}
                  strokeWidth={2}
                />
              );
            })}
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

    // Calculate bounding box for hex grid
    // Leftmost hex center: col=0, row=0 => cx0
    // Rightmost hex center: col=N-1, row=N-1 => cx1
    const minX = HEX_WIDTH / 2;
    const maxX = HEX_WIDTH * (N - 1) + HEX_WIDTH / 2 + HEX_WIDTH * (N - 1) / 2;
    // Topmost hex center: row=0 => cy0
    // Bottommost hex center: row=N-1 => cy1
    const minY = HEX_SIZE;
    const maxY = HEX_SIZE * 1.5 * (N - 1) + HEX_SIZE;
    // Add hex radius to all sides
    const pad = HEX_SIZE;
    const width = maxX - minX + pad * 2;
    const height = maxY - minY + pad * 2;
    const viewBox = `${minX - pad} ${minY - pad} ${width} ${height}`;

    return (
      <svg
        className="hex-board"
        viewBox={viewBox}
        width={width}
        height={height}
      >
        {hexes}
      </svg>
    );
  }
}
