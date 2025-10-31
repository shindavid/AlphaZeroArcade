import './Connect4.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';
import { Connect4Animation } from './Connect4Animation';

// Connect4 board dimensions
const ROWS = 6;
const COLS = 7;

const ANIMATION_INTERVAL = 60; // ms per row drop

const computeColHeights = (board) => {
  const heights = Array(COLS).fill(0);
  for (let col = 0; col < COLS; ++col) {
    let h = 0;
    for (let row = 0; row < ROWS; ++row) {
      const cell = board[row * COLS + col];
      if (cell === 'R' || cell === 'Y') h++;
    }
    heights[col] = h;
  }
  return heights;
};

export default class Connect4App extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: null,
      animation: null, // { col, row, disc, targetRow, animRow }
      skipNextAnimation: false,
    };
    this.animationHelper = new Connect4Animation();
  }

  // Override for colorful icons
  seatToHtml = seat =>
    <span className={`seat-icon ${seat}`} />;

  // Override for animations
  handleStateUpdate(payload) {
    super.handleStateUpdate(payload);

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
    console.log(`Column ${col} clicked`);
    if (!this.gameActive() || (this.state.animation && this.state.animation.col !== null)) return;

    let row = ROWS - computeColHeights(this.state.board)[col] - 1;

    const disc = this.state.seatAssignments[this.state.currentTurn];
    console.log(`Animating disc drop at col ${col}, row ${row} for seat ${disc}`);
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

    // board layout constants (match CSS)
    const CELL = 56; // px
    const PAD_TOP = 20; // .board padding-top
    const PAD_LEFT = 18; // .board padding-left

    return (
      <div
        className={`animated-disc ${anim.disc}`}
        style={{
          left: PAD_LEFT + anim.col * CELL,
          top: PAD_TOP + anim.animRow * CELL,
          position: 'absolute',
        }}
      />
    );
  }

  renderBoard() {
    if (!this.state.board) return null;
    const grid = [];
    const heights = computeColHeights(this.state.board);
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

        let ghostDisc = null;
        let targetRow = ROWS - heights[col] - 1;
        if (col === this.state.proposed_action && row === targetRow) {
           ghostDisc = <span className={`cell ghost ${this.state.seatAssignments[this.state.currentTurn]}`} />;
        }

        grid.push(
          <div
            key={idx}
            className={cellClass}
            onClick={isLegal ? () => this.handleCellClick(col) : undefined}
            role={isLegal ? "button" : undefined}
            tabIndex={isLegal ? 0 : -1}
            aria-label={isLegal ? `Play in column ${col + 1}` : undefined}
            >
            {ghostDisc}
          </div>
        );
      }
    }
    return <div className="board" style={{ position: 'relative' }}>
      {grid}
      {this.renderAnimatedDisc()}
    </div>;
  }
}
