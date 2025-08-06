import './App.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

export default class App extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: Array(9).fill('_'),
    };
  }

  isEmpty = (cell) => {
    return this.state.board[cell] === '_';
  }

  handleCellClick = (cell) => {
    if (!this.gameActive() || !this.isEmpty(cell)) return;
    this.sendMove(cell);
  };

  renderBoard() {
    return (
      <div className="board">
        {this.state.board && this.state.board.map((value, cell) => {
          const empty = this.isEmpty(cell);
          return (
            <button
              key={cell}
              className={`square${empty && !this.state.gameEnd ? ' legal-move' : ''}`}
              onClick={() => this.handleCellClick(cell)}
              disabled={!!this.state.gameEnd || !empty}
            >{empty ? '' : value}</button>
          );
        })}
      </div>
    );
  }
}
