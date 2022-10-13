import React, { Component } from "react";
import {useEffect, useRef} from "react";
import 'toolcool-range-slider';

function MySlider(props) {
  const sliderRef = useRef();

  useEffect(() => {
    const slider = sliderRef.current;
    const onChange = (evt) => {
      if (evt.target.id === props.name) {
        const x = evt.detail.value;
        console.log("Slider " + props.name + " was dragged to: " + x + " [" + evt.target.id + "]");
        props.update(x);
      }
    }
    slider?.addEventListener('change', onChange);
    return () => {
      slider?.removeEventListener('change', onChange);
    }
  }, []);

  return (
    <toolcool-range-slider
      id={props.name}
      min={0}
      max={100}
      value={props.value}
      step="1"
      ref={sliderRef}
    />
  );
}

class Header extends Component {
  constructor(props) {
    super(props);
    this.state = {
      x: 0,
    };
  }

  updateX(x) {
    this.setState({x: x});
  }

  render() {
    return (
      <div>
        <MySlider name="1" value={this.state.x} update={(x) => this.updateX(x) } />
        <p/>
        <MySlider name="2" value={this.state.x} update={(x) => this.updateX(x) } />
      </div>
    );

    // const file = this.props.debug_file;
    // let fileInfo;
    // if (file) {
    //   fileInfo = <p>Replaying game: {file.name}</p>
    // } else {
    //   fileInfo = "";
    // }
    // return (
    //   <div className="text-center">
    //     <h1>Connect4 Replay Tool</h1>
    //     <div id="upload-box">
    //       <input type="file" accept=".xml" name='debugfile' onChange={this.props.handleUpload} />
    //       {fileInfo}
    //     </div>
    //   </div>
    // );
  }
}

export default Header;
