import React, { Component } from "react";
import {useEffect, useRef} from "react";
import 'toolcool-range-slider';

function MySlider(props) {
  const sliderRef = useRef();

  useEffect(() => {
    const slider = sliderRef.current;
    const onChange = (evt) => {
      const v = evt.detail.value;
      console.log("Slider " + props.name + " was dragged to: " + v);
      props.update(v);
    }
    slider?.addEventListener('change', onChange);
    return () => {
      slider?.removeEventListener('change', onChange);
    }
  }, []);

  return (
    <toolcool-range-slider
      id={props.name}
      min={props.min}
      max={props.max}
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
      x: 10,
      y: 5,
    };
  }

  updateX(x) {
    this.setState({
      x: x,
      y: Math.ceil(x/2),
    });
  }

  updateY(y) {
    this.setState({y: y});
  }

  render() {
    const x = this.state.x;
    const y = this.state.y;
    const minY = Math.ceil(x/2);
    const maxY = x;
    return (
      <div>
        <table><tbody>
          <tr>
            <td>Choose x:</td>
            <td style={{padding:10}}><MySlider name="x" min={10} max={50} value={this.state.x} update={(x) => this.updateX(x) } /></td>
            <td>{x}</td>
          </tr>
          <tr>
            <td>Choose a y in the range [x/2, x]:</td>
            <td style={{padding:10}}><MySlider name="y" min={minY} max={maxY} value={this.state.y} update={(y) => this.updateY(y) } /></td>
            <td>{y}</td>
          </tr>
        </tbody></table>
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
