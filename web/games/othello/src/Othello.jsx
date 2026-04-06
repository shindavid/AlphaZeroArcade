import './Othello.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

const N = 8;
const PASS_MOVE = 64;
const PASS_MOVE_STR = 'PA';
const COL_LABELS = 'ABCDEFGH';

function cellToMoveStr(row, col) {
  return COL_LABELS[col] + (row + 1);
}

export default class OthelloApp extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: null,
    };
  }

  seatToHtml = (seat, lastMove = false) => {
    if (seat === ' ') return null;
    return <span className={`disc ${seat} ${lastMove ? 'last-move' : null}`} />;
  };

  handleCellClick = (row, col) => {
    if (!this.gameActive()) return;
    const idx = row * N + col;
    if (!this.state.legalMoves.includes(idx)) return;
    this.sendMove(cellToMoveStr(row, col));
  };

  renderPassButton = () => (
    <button
      className="pass-btn"
      onClick={() => this.sendMove(PASS_MOVE_STR)}
      title="Pass turn"
    >
      Pass
    </button>
  );

  renderBoard() {
    const { board, legalMoves, proposedAction, seatAssignments, currentTurn, lastMove } = this.state;
    if (!board) return null;

    const cells = [];
    for (let r = 0; r <= N; r++) {
      for (let c = 0; c <= N; c++) {
        if (r === 0 && c === 0) {
          cells.push(<div key="corner" className="label-cell" />);
        } else if (r === 0) {
          cells.push(
            <div key={`col-${c}`} className="label-cell top-label">
              {COL_LABELS[c - 1]}
            </div>
          );
        } else if (c === 0) {
          cells.push(
            <div key={`row-${r}`} className="label-cell left-label">
              {r}
            </div>
          );
        } else {
          const row = r - 1;
          const col = c - 1;
          const idx = row * N + col;
          const legal = this.gameActive() && legalMoves.includes(idx);

          const isLastMove = lastMove === idx;
          const ghostDisc = proposedAction === idx
            ? <span className={`ghost disc ${seatAssignments[currentTurn]}`} />
            : null;

          cells.push(
            <div
              key={`cell-${r}-${c}`}
              className={`cell${legal ? ' legal-move' : ''}`}
              onClick={legal ? () => this.handleCellClick(row, col) : undefined}
            >
              {this.seatToHtml(board[idx], isLastMove)}
              {ghostDisc}
            </div>
          );
        }
      }
    }

    const passEnabled = legalMoves.includes(PASS_MOVE);

    return (
      <div className="board-wrapper">
        <div className="board">{cells}</div>
        {passEnabled && this.renderPassButton()}
      </div>
    );
  }
}
