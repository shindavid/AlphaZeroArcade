import React, { Component } from "react";

class Header extends Component {
  render() {
    const file = this.props.debug_file;
    let fileInfo;
    if (file) {
      fileInfo = <p>Replaying game: {file.name}</p>
    } else {
      fileInfo = "";
    }
    return (
      <div className="text-center">
        <h1>Connect4 Replay Tool</h1>
        <div id="upload-box">
          <input type="file" accept=".xml" name='debugfile' onChange={this.props.handleUpload} />
          {fileInfo}
        </div>
      </div>
    );
  }
}

export default Header;
