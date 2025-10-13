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
    const { board, legalMoves, lastAction } = this.state;
    if (!board) return null;

    const cells = [];
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const idx = r * N + c;
        const v = board[idx];
        const legal = this.gameActive() && legalMoves.includes(idx);

        let cls = 'cell';
        if (v === '*' || v === '0') cls += ' ' + v;
        if (legal) cls += ' legal-move';
        if (lastAction === idx) cls += ' last-move';

        cells.push(
          <div
            key={idx}
            className={cls}
            onClick={legal ? () => this.handleCellClick(r, c) : undefined}
            role={legal ? 'button' : undefined}
            tabIndex={legal ? 0 : -1}
            aria-label={legal ? `Play ${r + 1},${c + 1}` : undefined}
          >
            {this.seatToHtml(v)}
          </div>
        );
      }
    }

    const passEnabled = legalMoves.includes(PASS_MOVE) ?? false;
    console.log('legalMoves:', legalMoves, 'passEnabled:', passEnabled);

    return (
      <div className="board-wrapper">
        <div className="board">{cells}</div>
        {passEnabled && this.renderPassButton()}
      </div>
    );
  }
}
