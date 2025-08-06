import './App.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

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
      legalMoves: [],
      lastAction: null,
      animating: false,
      animCol: null,
      animRow: null,
      animTargetRow: null,
      animDisc: null,
      animSource: null,
      lastMoveSentCol: null,
    };
    this.animTimer = null;
  }

  // Animation helpers
  endAnimation = () => {
    if (this.animTimer) clearInterval(this.animTimer);
    this.setState({
      animating: false,
      animCol: null,
      animRow: null,
      animDisc: null,
      animSource: null,
      lastAction: null,
      animTargetRow: null,
    });
  };

  startAnimation = ({ col, row, disc, source, onComplete }) => {
    this.endAnimation();
    this.setState({
      animating: true,
      animCol: col,
      animRow: 0,
      animDisc: disc,
      animSource: source,
      animTargetRow: source === 'opponent' ? row : null,
      lastAction: col,
    });
    let animationDone = false;
    this.animTimer = setInterval(() => {
      this.setState(prev => {
        if (prev.animRow < row) {
          return { animRow: prev.animRow + 1 };
        } else {
          if (!animationDone) {
            animationDone = true;
            clearInterval(this.animTimer);
            this.endAnimation();
            if (onComplete) onComplete();
          }
          return null;
        }
      });
    }, ANIMATION_INTERVAL);
  };

  handleMessage(msg) {
    if (msg.type === 'state_update') {
      this.setState({
        board: Array.from(msg.payload.board),
        turn: msg.payload.turn,
        legalMoves: msg.payload.legal_moves || [],
        gameEnd: null,
      });
      // Animation logic: check for last_action
      let last = msg.payload.last_action;
      if (last && last !== '-') {
        let col = parseInt(last, 10) - 1;
        if (!isNaN(col) && col >= 0 && col < COLS) {
          // If this is the player's last move, don't animate again
          if (this.state.lastMoveSentCol === col) {
            this.setState({ lastMoveSentCol: null });
          } else {
            // Animate opponent's move
            let b = Array.from(msg.payload.board || '');
            let disc = null;
            let row = -1;
            // Find the row where the new disc landed (first non-empty cell from top)
            for (let r = 0; r < ROWS; ++r) {
              let idx = r * COLS + col;
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
    } else if (msg.type === 'game_end') {
      this.setState({
        board: Array.from(msg.payload.board),
        gameEnd: msg.payload,
        legalMoves: [],
      });
      this.endAnimation();
    }
  }

  handleCellClick = (col) => {
    if (!this.gameActive() || this.state.animating || !this.state.legalMoves.includes(col)) return;
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

  sendMove(col) {
    const ws = this.socketRef.current;
    ws.send(JSON.stringify({ type: 'make_move', payload: { index: col } }));
  }

  renderBoard() {
    const grid = [];
    for (let row = 0; row < ROWS; ++row) {
      for (let col = 0; col < COLS; ++col) {
        const idx = row * COLS + col;
        const cell = this.state.board[idx];
        const isLegal = this.state.legalMoves.includes(col);
        let cellClass = "";
        // During animation, render the cell as empty (no disc) if it's the animated target
        let hideDisc = false;
        if (this.state.animating && this.state.animCol === col) {
          if (
            this.state.animSource === 'player' && this.state.animRow !== null && row === this.state.animRow && cell === this.state.animDisc
          ) {
            hideDisc = true;
          } else if (
            this.state.animSource === 'opponent' && this.state.animTargetRow !== null && row === this.state.animTargetRow && cell === this.state.animDisc
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
            onClick={isLegal ? () => this.handleCellClick(col) : undefined}
            role={isLegal ? "button" : undefined}
            tabIndex={isLegal ? 0 : -1}
            aria-label={isLegal ? `Play in column ${col + 1}` : undefined}
          />
        );
      }
    }
    return grid;
  }

  renderAnimatedDisc() {
    if (!this.state.animating || this.state.animCol === null || this.state.animRow === null || !this.state.animDisc) return null;
    const left = 18 + 4 + this.state.animCol * 56;
    const top = 20 + 4 + this.state.animRow * 56;
    const discClass = this.state.animDisc === 'R' ? 'red' : 'yellow';
    return (
      <div
        className={`animated-disc ${discClass}`}
        style={{ left, top }}
      />
    );
  }

  render() {
    return (
      <div className="container" style={{ minHeight: '600px', justifyContent: 'flex-start' }}>
        <div className="status-bar" style={{ marginBottom: '1.5em' }}>
          <span className="status-message-area">
            {this.state.gameEnd
              ? this.state.gameEnd.msg
              : <>Next: <b>{this.state.turn}</b></>
            }
          </span>
        </div>
        <div className="board connect4-board" style={{ position: 'relative' }}>
          {this.renderBoard()}
          {this.renderAnimatedDisc()}
        </div>
        <div className="button-row">
          <button
            className="status-action-btn"
            onClick={this.handleResign}
            disabled={!!this.state.gameEnd || this.state.loading}
          >
            Resign
          </button>
          <button
            className="status-action-btn"
            onClick={this.handleNewGame}
            disabled={!this.state.gameEnd || this.state.loading}
          >
            New Game
          </button>
        </div>
      </div>
    );
  }
}
