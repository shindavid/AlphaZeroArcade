import './TicTacToe.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

export default class TicTacToeApp extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: null,
    };
  }

  isEmpty = (cell) => {
    return this.state.board[cell] === ' ';
  }

  handleCellClick = (cell) => {
    if (!this.gameActive()) return;
    if (!this.isEmpty(cell)) return;
    this.sendMove(cell);
  };

  renderBoard() {
    if (!this.state.board) return null;
    return (
      <div className="board">
        {this.state.board.map((value, cell) => {
          const empty = this.isEmpty(cell);
          const legal = empty && this.gameActive();

          let proposed = null;
          if (cell === this.state.proposedAction) {
            proposed = <span className={'ghost'}> {this.state.seatAssignments[this.state.currentTurn]} </span>;
          }
          return (
            <button
              key={cell}
              className={`square${legal ? ' legal-move' : ''}`}
              onClick={() => this.handleCellClick(cell)}
              disabled={!!this.state.gameEnd || !legal}
            >{empty ? '' : this.seatToHtml(value)}
             {proposed}
            </button>
          );
        })}
      </div>
    );
  }
}
