// Auto-generated via: ./py/tools/make_scaffold_for_web_game.py -g Othello

import './Othello.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

const N = 8; // 8x8 board
const PASS_MOVE = 64; // index for "pass" move

export default class OthelloApp extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: null,
    };
  }

  RENDERERS = {
    seatIcon: (seat) => {
      if (seat === 0) return <span className="disc B"/>;
      if (seat === 1) return <span className="disc W"/>;
      return String(seat);
    }
  }

  seatToHtml = (seat) => {
    if (seat === '*') return <span className="disc B"/>;
    if (seat === '0') return <span className="disc W"/>;
    return null;
  };

  handleCellClick = (row, col) => {
    if (!this.gameActive()) return;
    const idx = row * N + col;
    if (!this.state.legalMoves.includes(idx)) return;
    this.sendMove(idx);
  };

  renderPassButton = () => (
    <button
      className="pass-btn"
      onClick={() => this.sendMove(PASS_MOVE)}
      title="Pass turn"
    >
      Pass
    </button>
  );

  renderBoard() {
    const { board, legalMoves } = this.state;
    if (!board) return null;

    const cells = [];
    const N = 8;
    const colLabels = ['', ...'ABCDEFGH'];
    const rowLabels = Array.from({ length: N }, (_, i) => i + 1);

    for (let r = 0; r <= N; r++) {
      for (let c = 0; c <= N; c++) {
        if (r === 0 && c === 0) {
          cells.push(<div key="corner" className="label-cell" />);
        } else if (r === 0) {
          // column labels A–H
          cells.push(
            <div key={`col-${c}`} className="label-cell top-label">
              {colLabels[c]}
            </div>
          );
        } else if (c === 0) {
          // row labels 1–8
          cells.push(
            <div key={`row-${r}`} className="label-cell left-label">
              {rowLabels[r - 1]}
            </div>
          );
        } else {
          const idx = (r - 1) * N + (c - 1);
          const v = board[idx];
          const legal = this.gameActive() && legalMoves.includes(idx);

          let cls = 'cell';
          if (legal) cls += ' legal-move';

          cells.push(
            <div
              key={`cell-${r}-${c}`}
              className={cls}
              onClick={legal ? () => this.handleCellClick(r - 1, c - 1) : undefined}
            >
              {this.seatToHtml(v)}
            </div>
          );
        }
      }
    }

    const passEnabled = legalMoves.includes(PASS_MOVE) ?? false;

    return (
      <div className="board-wrapper">
        <div className="board">{cells}</div>
        {passEnabled && this.renderPassButton()}
      </div>
    );
  }
}
