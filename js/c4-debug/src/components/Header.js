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
    return (
      <div id="upload-box">
        <input type="file" name='debugfile' onChange={this.handleUpload} />
        <p>Filename: {file.name}</p>
        <p>File type: {file.type}</p>
        <p>File size: {file.size} bytes</p>
        {file && <ImageThumb image={file} />}
      </div>
    );
  }

  handleUpload(event) {
    this.setState({file: event.target.files[0]});

    const data = new FormData();
    console.log('target.files: ' + typeof(event.target.files[0]));
    data.append('debug_file', event.target.files[0]);
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
