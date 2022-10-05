import React, { Component, Fragment } from "react";
import Header from "./components/Header";
import Body from "./components/Body";
import axios from "axios";

class Move {
  constructor(elem) {
    const cp = elem.getAttribute('cp');
    const board = elem.getAttribute('board');
    this.cp = cp;
    this.board = board;
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
      const moves = Array.from(game.getElementsByTagName('Move'));
      const history = moves.map(m => new Move(m));
      const player_index = parseInt(game.getAttribute('player'));
      this.setState({
        player_index: player_index,
        history: history
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
          history={this.state.history}
        />
      </Fragment>
    );
  }
}

export default App;
