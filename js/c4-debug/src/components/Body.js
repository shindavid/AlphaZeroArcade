import React, { Component } from "react";
import leftarrow from '../images/left.svg';
import rightarrow from '../images/right.svg';
import redcircle from '../images/red.svg';
import yellowcircle from '../images/yellow.svg';

class Square extends Component {
  render() {
    const color = this.props.color;
    let className = 'square';
    if (color === 'R') {
      className = 'red';
    } else if (color === 'Y') {
      className = 'yellow';
    }
    return (
      <span className={className} ></span>
    )
  }
}

class BoardRow extends Component {
  render() {
    const board = this.props.board;
    const row = this.props.row;
    const renderSquares = () => {
      let squares = [];
      for (let i = 0; i < 7; i++) {
        const index = i * 6 + row;
        const color = board.charAt(index);
        squares.push(<Square color={color} key={index} index={index} />);
      }
      return squares;
    }
    return (
      <span className="row">
        { renderSquares() }
      </span>
    );
  }
}

class Arrow extends Component {
  render() {
    const move_index = this.props.move_index;
    const history = this.props.history;
    const delta = this.props.delta;
    const new_move_index = move_index + delta;
    const hidden = (new_move_index === -1) || (new_move_index === history.length);

    if (hidden) {
      return (
        <span className="arrow" />
      );
    }
    return (
      <span className="arrow">
        <img alt={this.props.alt} src={this.props.src} onClick={() => this.handleClick()} />
      </span>
    );
  }

  handleClick() {
    const body = this.props.body;
    const move_index = this.props.move_index;
    const delta = this.props.delta;
    const new_move_index = move_index + delta;

    body.setState({
      move_index: new_move_index,
    });
  }
}

function GameHistory(props) {
  const history = props.history;
  const move_index = props.move_index;
  const move = history[move_index];
  const body = props.body;

  const renderRows = () => {
    let rows = [];
    for (let i = 5; i >= 0; i--) {
      rows.push(<BoardRow board={move.board} row={i} key={i}/>);
    }
    return rows;
  }

  return (
    <table className="center"><tbody>
    <tr>
      <td>
        <Arrow history={history} move_index={move_index} body={body} delta={-1} alt="left" src={leftarrow}/>
      </td>
      <td>
        <span>
          { renderRows() }
        </span>
      </td>
      <td>
        <Arrow history={history} move_index={move_index} body={body} delta={+1} alt="right" src={rightarrow}/>
      </td>
    </tr>
    </tbody></table>
  );
}

function MyColor(props) {
  const player_index = props.player_index;
  const my_color = player_index === 0 ? redcircle : yellowcircle;
  return (
    <div className="centertext">
      My Color:&nbsp;
      <span className="minicircle center">
          <img src={my_color} alt="color" />
        </span>
      <br/><br/>
    </div>
  );

}

class Body extends Component {
  constructor(props) {
    super(props);
    this.state = {
      move_index: props.move_index,
    };
  }

  render() {
    const history = this.props.history;
    if (history.length === 0) {
      return "";
    }
    const move_index = this.state.move_index;
    const player_index = this.props.player_index;
    return (
      <div className="center">
        <MyColor player_index={player_index} />
        <GameHistory history={history} move_index={move_index} body={this} />
      </div>
    );
  }
}

export default Body;
