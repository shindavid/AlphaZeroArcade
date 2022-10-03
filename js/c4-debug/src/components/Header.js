import React, { Component } from "react";
import axios from "axios";

/**
 * Component to handle file upload. Works for image
 * uploads, but can be edited to work for any file.
 */
class FileUpload extends Component {
  constructor(props) {
    super(props);
    this.state = {
      file: "",
    };
    this.handleUpload = this.handleUpload.bind(this);
  }

  render() {
    const file = this.state.file;
    let fileInfo;
    if (file) {
      fileInfo = <p>Replaying game: {file.name}</p>
    } else {
      fileInfo = "";
    }
    return (
      <div id="upload-box">
        <input type="file" accept=".xml" name='debugfile' onChange={this.handleUpload} />
        {fileInfo}
      </div>
    );
  }

  handleUpload(event) {
    const file = event.target.files[0];
    this.setState({file: file});

    const data = new FormData();
    data.append('debug_file', file);
    data.append('header', {
      'Access-Control-Allow-Origin': '*',
    });

    // Add code here to upload file to server
    // ...
    const url = "http://localhost:8000/c4debug/";
    axios.post(url, data, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    })
      .then(res => { // then print response status
        console.log('DATA: ' + res.data)
      })
      .catch(error => console.log('ERROR: ' + error.message));
  }
}

/**
 * Component to display thumbnail of image.
 */
const ImageThumb = ({ image }) => {
  return <img src={URL.createObjectURL(image)} alt={image.name} />;
};

class Header extends Component {
  render() {
    return (
      <div className="text-center">
        <h1>Connect4 Replay Tool</h1>
        <FileUpload state={this.state}/>
      </div>
    );
  }
}

export default Header;
