import React, { Component, Fragment } from "react";
import Header from "./components/Header";
import Body from "./components/Body";


class Child {
  constructor(elem) {
    this.action = 1 + parseInt(elem.getAttribute('action'));
    this.rP = parseFloat(elem.getAttribute('rP'));
    this.P = parseFloat(elem.getAttribute('P'));
    this.dir = parseFloat(elem.getAttribute('dir'));
    this.V = parseFloat(elem.getAttribute('V'));
    this.N = parseFloat(elem.getAttribute('N'));
    this.PUCT = parseFloat(elem.getAttribute('PUCT'));
    this.E = parseFloat(elem.getAttribute('E'));
  }
}


class Visit {
  constructor(elem, parent) {
    this.board_visit_num = null;
    this.parent = parent;
    this.player = parseInt(elem.getAttribute('player'));
    this.depth = parseInt(elem.getAttribute('depth'));
    this.board = elem.getAttribute('board');
    this.leaf = parseInt(elem.getAttribute('leaf'));
    this.eval = elem.getAttribute('eval').split(',').map(parseFloat);
    this.terminal = parseInt(elem.getAttribute('terminal'));
    this.action = elem.getAttribute('action');
    this.value_avg = elem.getAttribute('value_avg').split(',').map(parseFloat);

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
    this.index = parseInt(elem.getAttribute('i'));
    this.visits = Array.from(elem.children).map(v => new Visit(v, this));
  }
}


class Move {
  constructor(elem) {
    this.board = elem.getAttribute('board');
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
    const file = event.target.files[0];
    let reader = new FileReader();
    reader.readAsText(file);

    reader.onload = () => {
      const text = reader.result;
      let parser = new DOMParser();
      const tree = parser.parseFromString(text, "text/xml");
      const game = tree.getElementsByTagName('Game')[0];
      const moves = Array.from(game.children);
      const history = moves.map(m => new Move(m));
      const player_index = parseInt(game.getAttribute('player'));
      this.setState({
        player_index: player_index,
        history: history,
        move_index: 0,
        iter_index: 0,
        top_index: 0,
        bot_index: 0,
      })
    }
    // The below code sends the file to a backend server. This would make more sense for a web interface to play
    // against the CPU.
    //
    // this.setState({debug_file: file});
    //
    // //const tree = parser.parseFromString(xml_str);
    //
    // const data = new FormData();
    // data.append('debug_file', file);
    // data.append('header', {
    //   'Access-Control-Allow-Origin': '*',
    // });
    //
    // // Add code here to upload file to server
    // // ...
    // const url = "http://localhost:8000/c4debug/";
    // axios.post(url, data, {
    //   headers: {
    //     'Content-Type': 'multipart/form-data',
    //   }
    // })
    //   .then(res => { // then print response status
    //     //this.setState({'tag': res.data})
    //     console.log('DATA: ' + res.data)
    //   })
    //   .catch(error => console.log('ERROR: ' + error.message));
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
