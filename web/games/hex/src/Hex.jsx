// Auto-generated via: ./py/tools/make_scaffold_for_web_game.py -g Hex -o

import './Hex.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

const BOARD_SIZE = 11;
const B = BOARD_SIZE;  // alias for convenience
const SWAP_MOVE = B * B;

const HEX_SIZE_PX = 30;
const HEX_WIDTH_PX = Math.sqrt(3) * HEX_SIZE_PX;

const MIN_X = HEX_WIDTH_PX / 2;
const MAX_X = HEX_WIDTH_PX * (B - 1) + HEX_WIDTH_PX / 2 + HEX_WIDTH_PX * (B - 1) / 2;
const MIN_Y = HEX_SIZE_PX;
const MAX_Y = HEX_SIZE_PX * 1.5 * (B - 1) + HEX_SIZE_PX;
const PADDING = HEX_SIZE_PX * 3;
const WIDTH = MAX_X - MIN_X + PADDING * 2;
const HEIGHT = MAX_Y - MIN_Y + PADDING * 2;
const VIEW_BOX = `${MIN_X - PADDING} ${MIN_Y - PADDING} ${WIDTH} ${HEIGHT}`;

const HEX_SE = 0;
const HEX_SW = 1;
const HEX_W = 2;
const HEX_NW = 3;
const HEX_NE = 4;
const HEX_E = 5;

export default class HexApp extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: null,
    };
  }

  RENDERERS = {
    seatIcon: (seat) => {
      if (seat === 0) {
        return this.seatToHtml("R");
      } else if (seat === 1) {
        return this.seatToHtml("B");
      }
      return String(seat);
    }
  };

  // Override for colorful icons
  seatToHtml = (seat) => {
    const color = seat === "R" ? "#e44" : seat === "B" ? "#24f" : "#888";
    return (
      <span className="hex-seat-icon">
        <svg width="18" height="18" viewBox="0 0 18 18">
          <polygon
            points="9,1 16,5.5 16,12.5 9,16 2,12.5 2,5.5"
            fill={color}
            stroke="#333"
            strokeWidth="1"
          />
        </svg>
      </span>
    );
  }

  getBorderType = (row, col, dir) => {
    let south = row === 0;
    let north = row === B - 1;
    let west = col === 0;
    let east = col === B - 1;
    if (south && (dir === HEX_SE || dir === HEX_SW)) return 'red';
    if (north && (dir === HEX_NW || dir === HEX_NE)) return 'red';
    if (west && (dir === HEX_NW || dir === HEX_W)) return 'blue';
    if (east && (dir === HEX_SE || dir === HEX_E)) return 'blue';
    return 'black';
  }

  borderTypeToStroke = (type) => {
    switch (type) {
      case 'red':
        return '#b2182b'; // darker red for border
      case 'blue':
        return '#2255a5'; // darker blue for border
      default:
        return '#000';
    }
  }

  borderTypeToStrokeWidth = (type) => {
    switch (type) {
      case 'red':
        return 4;
      case 'blue':
        return 4;
      default:
        return 1;
    }
  }

  renderCell = (row, col) => {
    let board = this.state.board;
    let legalMoves = this.state.legalMoves;
    let mySeat = this.state.mySeat;

    const cell = row * B + col;
    const color = board[cell];

    // Hex center coordinates
    const cx = HEX_WIDTH_PX * col + HEX_WIDTH_PX / 2 + HEX_WIDTH_PX * row / 2;
    const cy = HEX_SIZE_PX * 1.5 * (B - 1 - row) + HEX_SIZE_PX;

    const isLegal = legalMoves.includes(cell) && this.gameActive();

    // Compute hex corners
    const corners = [];
    for (let k = 0; k < 6; ++k) {
      const angle = Math.PI / 3 * k + Math.PI / 6;
      corners.push([
        cx + HEX_SIZE_PX * Math.cos(angle),
        cy + HEX_SIZE_PX * Math.sin(angle)
      ]);
    }

    // Determine hover class
    let hoverClass = "";
    if (isLegal) {
      hoverClass = mySeat === "R" ? " hex-hover-red" : " hex-hover-blue";
    }

    return (
      <g key={cell}>
        <polygon
          className={
            "hex-cell" +
            (color === "R" ? " hex-red" : "") +
            (color === "B" ? " hex-blue" : "") +
            (isLegal ? hoverClass : "")
          }
          points={corners.map(p => p.join(",")).join(" ")}
          onClick={isLegal ? () => this.sendMove(cell) : undefined}
          style={{ pointerEvents: isLegal ? "auto" : "none" }}
        />
        {/* Draw 6 border lines manually, colored by position */}
        {corners.map((p, dir) => {
          const p2 = corners[(dir + 1) % 6];
          let borderType = this.getBorderType(row, col, dir);
          let stroke = this.borderTypeToStroke(borderType);
          let strokeWidth = this.borderTypeToStrokeWidth(borderType);
          return (
            <line
              key={"border" + dir}
              x1={p[0]}
              y1={p[1]}
              x2={p2[0]}
              y2={p2[1]}
              className="hex-border"
              stroke={stroke}
              strokeWidth={strokeWidth}
            />
          );
        })}
      </g>
    );
  }

  renderSwapButton = () => {
    const swapBtnX = MAX_X + HEX_SIZE_PX * 1.5;
    const swapBtnY = HEX_SIZE_PX * 1.5 * (B - 2) + HEX_SIZE_PX;

    return (
      <button
        className="hex-swap-btn"
        style={{
          left: swapBtnX,
          top: swapBtnY,
        }}
        onClick={() => this.sendMove(SWAP_MOVE)}
      >
        Swap
      </button>
    );
  }

  renderBoard() {
    let board = this.state.board;
    let legalMoves = this.state.legalMoves;

    if (board === null) return null;

    const hexes = [];
    for (let row = B - 1; row >= 0; --row) {
      for (let col = 0; col < B; ++col) {
        hexes.push(this.renderCell(row, col));
      }
    }

    // Swap button logic
    const swapEnabled = legalMoves.includes(SWAP_MOVE);
    return (
      <div className="hex-board-wrapper">
        <svg
          className="hex-board"
          viewBox={VIEW_BOX}
          width={WIDTH}
          height={HEIGHT}
        >
          {hexes}
        </svg>
        {swapEnabled && this.renderSwapButton()}
      </div>
    );
  }
}
