import React, { Component, Fragment } from "react";
import Header from "./components/Header";
import Body from "./components/Body";


class Child {
  constructor(elem) {
    this.action = 1 + parseInt(elem.action);
    this.rP = parseFloat(elem.rP);
    this.P = parseFloat(elem.P);
    this.dir = parseFloat(elem.dir);
    this.V = parseFloat(elem.V);
    this.N = parseFloat(elem.N);
    this.PUCT = parseFloat(elem.PUCT);
    this.E = parseFloat(elem.E);
  }
}


class Visit {
  constructor(elem, parent) {
    this.board_visit_num = null;
    this.parent = parent;
    this.player = parseInt(elem.player);
    this.depth = parseInt(elem.depth);
    this.board = elem.board;
    this.leaf = parseInt(elem.leaf);
    this.eval = elem.eval.split(',').map(parseFloat);
    this.terminal = parseInt(elem.terminal);
    this.action = elem.action;
    this.value_avg = elem.value_avg.split(',').map(parseFloat);

    this.children = Array.from(elem.children).map(c => new Child(c));
    this.rP_sum = this.children.map((x) => x.rP).reduce((a, b) => a+b, 0);
    this.P_sum = this.children.map((x) => x.P).reduce((a, b) => a+b, 0);
    this.dir_sum = this.children.map((x) => x.dir).reduce((a, b) => a+b, 0);
    this.V_sum = this.children.map((x) => x.V).reduce((a, b) => a+b, 0);
    this.N_sum = this.children.map((x) => x.N).reduce((a, b) => a+b, 0);
    this.PUCT_sum = this.children.map((x) => x.PUCT).reduce((a, b) => a+b, 0);
  }
}


class Iter {
  constructor(elem, parent) {
    this.parent = parent;
    this.index = parseInt(elem.i);
    this.visits = Array.from(elem.children).map(v => new Visit(v, this));
  }
}


class Move {
  constructor(elem) {
    this.board = elem.board;
    this.iters = Array.from(elem.children).map(i => new Iter(i, this));

    this.board_to_visits = {};
    for (const iter of this.iters) {
      for (const visit of iter.visits) {
        let board = visit.board;
        let visits = null;
        if (board in this.board_to_visits) {
          visits = this.board_to_visits[board];
        } else {
          visits = Array.from([]);
          this.board_to_visits[board] = visits;
        }
        visit.board_visit_num = visits.length;
        visits.push(visit);
      }
    }
  }
}

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      debug_file: null,
      player_index: null,
      history: [],
      move_index: 0,
      iter_index: 0,
      top_index: 0,
      bot_index: 0,
    };
  }

  handleUpload(event) {
    const Schema = require('./proto/mcts_pb')
    const file = event.target.files[0];
    let reader = new FileReader();
    reader.readAsArrayBuffer(file);

    reader.onload = () => {
      const bytes = reader.result;
      const game = Schema.GameTree.deserializeBinary(bytes)
      const moves = game.moves
      const history = moves.map(m => new Move(m));
      const player_index = parseInt(game.player);
      this.setState({
        player_index: player_index,
        history: history,
        move_index: 0,
        iter_index: 0,
        top_index: 0,
        bot_index: 0,
      })
    }
  }

  render() {
    return (
      <Fragment>
        <Header
          debug_file={this.state.debug_file}
          handleUpload={(event) => this.handleUpload(event)}
        />
        <Body
          player_index={this.state.player_index}
          move_index={this.state.move_index}
          iter_index={this.state.iter_index}
          top_index={this.state.top_index}
          bot_index={this.state.bot_index}
          history={this.state.history}
        />
      </Fragment>
    );
  }
}

export default App;
