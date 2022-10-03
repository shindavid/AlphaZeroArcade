import React, { Component, Fragment } from "react";
import Header from "./components/Header";
import Body from "./components/Body";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      filename: "",
      history: [],
    };
  }

  render() {
    return (
      <Fragment>
        <Header state={this.state} />
        <Body state={this.state} />
      </Fragment>
    );
  }
}

export default App;
