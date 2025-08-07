import './Connect4.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';
import { Connect4Animation } from './Connect4Animation';

// Connect4 board dimensions
const ROWS = 6;
const COLS = 7;

const ANIMATION_INTERVAL = 60; // ms per row drop


export default class Connect4App extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: Array(ROWS * COLS).fill('_'),
      animation: null, // { col, row, disc, targetRow, animRow }
      skipNextAnimation: false,
    };
    this.animationHelper = new Connect4Animation();
  }

  // Animation helpers
  endAnimation = () => {
    this.animationHelper.end();
    this.setState({ animation: null });
  };

  // Override to animate opponent moves
  handleStateUpdate(payload) {
    // Call base class to update state
    if (super.handleStateUpdate) super.handleStateUpdate(payload);

    if (this.state.skipNextAnimation) {
      this.setState({ skipNextAnimation: false });
      return;
    }

    const col = parseInt(payload.last_action) - 1;  // Convert to 0-based index
    let row = payload.last_row;
    if (row >= 0) {
      const b = Array.from(payload.board || '');
      let disc = b[row * COLS + col];

      this.setState({
        animation: this.animationHelper.get(),
        lastAction: col,
      });

      this.startAnimation({ col, row, disc });
    }
  }

  startAnimation = ({ col, row, disc, onComplete }) => {
    this.animationHelper.start({
      col,
      row,
      disc,
      onComplete: () => {
        this.setState({ animation: null });
        if (onComplete) onComplete();
      },
      interval: ANIMATION_INTERVAL,
      onFrame: (animState) => {
        this.setState({ animation: { ...animState } });
      },
    });
  };

  handleCellClick = (col) => {
    if (!this.gameActive() || (this.state.animation && this.state.animation.col !== null)) return;

    // TODO: get this row info from the backend
    // Find the lowest empty row in this column
    let row = -1;
    for (let r = ROWS - 1; r >= 0; --r) {
      if (this.state.board[r * COLS + col] === '_') {
        row = r;
        break;
      }
    }
    if (row === -1) return;

    const disc = this.state.mySeat;
    this.setState({ skipNextAnimation: true });
    this.startAnimation({
      col,
      row,
      disc,
      onComplete: () => {
        this.sendMove(col);
      }
    });
  };

  // Helper to determine if a disc should be hidden for animation
  hideDisc(row, col) {
    const anim = this.state.animation;
    if (!anim || anim.col !== col) return false;
    return anim.targetRow !== anim.animRow && row === anim.targetRow;
  }

  renderAnimatedDisc() {
    const anim = this.state.animation;
    if (!anim || anim.col === null || anim.animRow === null || !anim.disc) return null;
    const left = 18 + 4 + anim.col * 56;
    const top = 20 + 4 + anim.animRow * 56;
    return (
      <div
        className={`animated-disc ${anim.disc}`}
        style={{ left, top }}
      />
    );
  }

  renderBoard() {
    const grid = [];
    for (let row = 0; row < ROWS; ++row) {
      for (let col = 0; col < COLS; ++col) {
        const idx = row * COLS + col;
        const cell = this.state.board[idx];
        const isLegal = this.state.legalMoves.includes(col);
        let cellClass = "";
        if (!this.hideDisc(row, col)) {
          if (cell === "R" || cell === "Y") cellClass = cell;
        }
        grid.push(
          <div
            key={idx}
            className={`connect4-cell${cellClass ? ' ' + cellClass : ''}`}
            onClick={isLegal ? () => this.handleCellClick(col) : undefined}
            role={isLegal ? "button" : undefined}
            tabIndex={isLegal ? 0 : -1}
            aria-label={isLegal ? `Play in column ${col + 1}` : undefined}
          />
        );
      }
    }
    // The board container must have the grid class for CSS to apply
    return <div className="connect4-board" style={{ position: 'relative' }}>
      {grid}
      {this.renderAnimatedDisc()}
    </div>;
  }
}
