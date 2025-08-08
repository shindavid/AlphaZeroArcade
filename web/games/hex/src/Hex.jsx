// Auto-generated via: ./py/tools/make_scaffold_for_web_game.py -g Hex -o

import './Hex.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

export default class HexApp extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: null,
    };
  }

  renderBoard() {
    if (!this.state.board) return null;
    // TODO: FILL IN THE BOARD RENDERING LOGIC HERE
  }
}
