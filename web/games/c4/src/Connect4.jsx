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
      board: null,
      colHeights: null,
      animation: null, // { col, row, disc, targetRow, animRow }
      skipNextAnimation: false,
    };
    this.animationHelper = new Connect4Animation();
  }

  // Override for colorful icons
  seatToHtml = (seat) => {
    if (seat === "R") {
      return <span className="seat-icon seat-R" />;
    }
    if (seat === "Y") {
      return <span className="seat-icon seat-Y" />;
    }
    return seat;
  }

  // Override for animations
  handleStartGame(payload) {
    super.handleStartGame(payload);

    this.setState({
      colHeights: payload.col_heights,
    });
  }

  // Override for animations
  handleStateUpdate(payload) {
    super.handleStateUpdate(payload);

    this.setState({
      colHeights: payload.col_heights,
    });

    if (this.state.skipNextAnimation) {
      this.setState({ skipNextAnimation: false });
      return;
    }

    const col = payload.last_col;
    if (col >= 0) {
      let row = ROWS - payload.col_heights[col];
      let disc = payload.board[row * COLS + col];

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

  endAnimation = () => {
    this.animationHelper.end();
    this.setState({ animation: null });
  };

  handleCellClick = (col) => {
    if (!this.gameActive() || (this.state.animation && this.state.animation.col !== null)) return;

    let row = ROWS - this.state.colHeights[col] - 1;

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
    return (
      <div
        className={`animated-disc ${anim.disc}`}
        style={{
          gridRow: anim.animRow + 1,
          gridColumn: anim.col + 1,
          position: 'absolute',
        }}
      />
    );
  }

  renderBoard() {
    if (!this.state.board) return null;
    const grid = [];
    for (let row = 0; row < ROWS; ++row) {
      for (let col = 0; col < COLS; ++col) {
        const idx = row * COLS + col;
        const cell = this.state.board[idx];
        const isLegal = this.gameActive() && this.state.legalMoves.includes(col);
        let cellClass = "cell";
        if (!this.hideDisc(row, col)) {
          if (cell === "R" || cell === "Y") cellClass += ' ' + cell;
        }
        if (isLegal) {
          cellClass += ' legal-move';
        }
        grid.push(
          <div
            key={idx}
            className={cellClass}
            onClick={isLegal ? () => this.handleCellClick(col) : undefined}
            role={isLegal ? "button" : undefined}
            tabIndex={isLegal ? 0 : -1}
            aria-label={isLegal ? `Play in column ${col + 1}` : undefined}
          />
        );
      }
    }
    return <div className="board" style={{ position: 'relative' }}>
      {grid}
      {this.renderAnimatedDisc()}
    </div>;
  }
}
