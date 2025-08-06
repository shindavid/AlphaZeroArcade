import './App.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';
import { Connect4Animation } from './Connect4Animation';

// Connect4 board dimensions
const ROWS = 6;
const COLS = 7;

const ANIMATION_INTERVAL = 60; // ms per row drop


export default class App extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: Array(ROWS * COLS).fill('_'),
      lastAction: null,
      animation: null, // { col, row, disc, source, targetRow, animRow }
      lastMoveSentCol: null,
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
    // Animate opponent move if last_action is present and not from player
    const last = payload.last_action;
    if (last && last !== '-') {
      const col = parseInt(last, 10) - 1;
      if (!isNaN(col) && col >= 0 && col < COLS) {
        // If this is the player's last move, don't animate again
        if (this.state.lastMoveSentCol === col) {
          this.setState({ lastMoveSentCol: null });
        } else {
          // Animate opponent's move
          const b = Array.from(payload.board || '');
          let disc = null;
          let row = -1;
          // Find the row where the new disc landed (first non-empty cell from top)
          for (let r = 0; r < ROWS; ++r) {
            const idx = r * COLS + col;
            if (b[idx] !== '_') {
              disc = b[idx];
              row = r;
              break;
            }
          }
          if (row !== -1 && disc) {
            this.startAnimation({ col, row, disc, source: 'opponent' });
          }
        }
      }
    }
  }

  startAnimation = ({ col, row, disc, source, onComplete }) => {
    this.animationHelper.start({
      col,
      row,
      disc,
      source,
      onComplete: () => {
        this.setState({ animation: null });
        if (onComplete) onComplete();
      },
      interval: ANIMATION_INTERVAL,
      onFrame: (animState) => {
        this.setState({ animation: { ...animState } });
      },
    });
    this.setState({
      animation: this.animationHelper.get(),
      lastAction: col,
    });
  };

  handleCellClick = (col) => {
    if (!this.gameActive() || (this.state.animation && this.state.animation.col !== null)) return;

    // Find the lowest empty row in this column
    let row = -1;
    for (let r = ROWS - 1; r >= 0; --r) {
      if (this.state.board[r * COLS + col] === '_') {
        row = r;
        break;
      }
    }
    if (row === -1) return;
    const disc = this.state.turn === 'R' ? 'R' : 'Y';
    this.setState({ lastMoveSentCol: col });
    this.startAnimation({
      col,
      row,
      disc,
      source: 'player',
      onComplete: () => {
        this.sendMove(col);
      }
    });
  };

  // Helper to determine if a disc should be hidden for animation
  hideDisc(row, col, cell) {
    const anim = this.state.animation;
    if (!anim || anim.col !== col) return false;
    if (
      anim.source === 'player' && anim.animRow !== null && row === anim.animRow && cell === anim.disc
    ) {
      return true;
    }
    if (
      anim.source === 'opponent' && anim.targetRow !== null && row === anim.targetRow && cell === anim.disc
    ) {
      return true;
    }
    return false;
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
        if (!this.hideDisc(row, col, cell)) {
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
