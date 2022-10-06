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
  }
}


class Visit {
  constructor(elem) {
    this.depth = parseInt(elem.getAttribute('depth'));
    this.board = elem.getAttribute('board');
    this.leaf = parseInt(elem.getAttribute('leaf'));
    this.eval = elem.getAttribute('eval').split(',').map(parseFloat);
    this.terminal = parseInt(elem.getAttribute('terminal'));
    this.action = elem.getAttribute('action');
    //this.value_sum = elem.getAttribute('value_sum').split(',').map(parseFloat);

    this.children = Array.from(elem.children).map(c => new Child(c));
    this.rP_sum = this.children.reduce((a, b) => a.rP + b.rP, 0);
    this.P_sum = this.children.reduce((a, b) => a.P + b.P, 0);
    this.dir_sum = this.children.reduce((a, b) => a.dir + b.dir, 0);
    this.V_sum = this.children.reduce((a, b) => a.V + b.V, 0);
    this.N_sum = this.children.reduce((a, b) => a.N + b.N, 0);
    this.PUCT_sum = this.children.reduce((a, b) => a.PUCT + b.PUCT, 0);
  }
}


class Iter {
  constructor(elem) {
    this.index = parseInt(elem.getAttribute('i'));
    this.visits = Array.from(elem.children).map(v => new Visit(v));
  }
}


class Move {
  constructor(elem) {
    const cp = elem.getAttribute('cp');
    const board = elem.getAttribute('board');
    this.cp = cp;
    this.board = board;
    this.iters = Array.from(elem.children).map(i => new Iter(i));
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
      visit_index: 0,
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
        visit_index: 0,
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
          visit_index={this.state.visit_index}
          history={this.state.history}
        />
      </Fragment>
    );
  }
}

export default App;
