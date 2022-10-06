import React, { Component } from "react";
import leftarrow from '../images/left.svg';
import rightarrow from '../images/right.svg';
import redcircle from '../images/red.svg';
import yellowcircle from '../images/yellow.svg';

class Square extends Component {
  render() {
    const color = this.props.color;
    let className = this.props.className;
    if (color === 'R') {
      className += ' red';
    } else if (color === 'Y') {
      className += ' yellow';
    } else if (color === 'r') {
      className += ' lightred';
    } else if (color === 'y') {
      className += ' lightyellow';
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
    const className = this.props.className;
    const rowName = this.props.rowName;
    const renderSquares = () => {
      let squares = [];
      for (let i = 0; i < 7; i++) {
        const index = i * 6 + row;
        const color = board.charAt(index);
        squares.push(<Square className={className} color={color} key={index} index={index} />);
      }
      return squares;
    }
    return (
      <span className={rowName}>
        { renderSquares() }
      </span>
    );
  }
}

class NAVArrow extends Component {
  render() {
    const index = this.props.index;
    const max_index = this.props.max_index;
    const delta = this.props.delta;
    const className = this.props.className;
    const alt = this.props.alt;
    const src = this.props.src;

    const new_index = index + delta;
    const hidden = (new_index === -1) || (new_index === max_index);

    if (hidden) {
      return (
        <span className={className} />
      );
    }
    return (
      <span className={className}>
        <img alt={alt} src={src} onClick={() => this.handleClick()} />
      </span>
    );
  }

  handleClick() {
    this.props.update(this.props.index + this.props.delta);
  }
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

function GameHistory(props) {
  const history = props.history;
  const move_index = props.move_index;
  const move = history[move_index];

  const renderRows = () => {
    const className = "square32";
    let rows = [];
    for (let i = 5; i >= 0; i--) {
      rows.push(<BoardRow className={className} rowName="row32" board={move.board} row={i} key={i}/>);
    }
    return rows;
  }

  return (
    <table className="center"><tbody>
    <tr>
      <td>
        <NAVArrow
          className="arrow"
          max_index={history.length}
          index={move_index}
          update={(i) => props.updateMoveIndex(i)}
          delta={-1}
          alt="left"
          src={leftarrow}
        />
      </td>
      <td>
        <span>
          { renderRows() }
        </span>
      </td>
      <td>
        <NAVArrow
          className="arrow"
          max_index={history.length}
          index={move_index}
          update={(i) => props.updateMoveIndex(i)}
          delta={+1}
          alt="right"
          src={rightarrow}
        />
      </td>
    </tr>
    </tbody></table>
  );
}

function MCTSNav(props) {
  const history = props.history;
  const move_index = props.move_index;
  const move = history[move_index];
  const iter_index = props.iter_index;
  const visit_index = props.visit_index;

  const iter = move.iters[iter_index];
  const visit = iter.visits[visit_index];
  const depth = visit.depth;

  const num_iters = move.iters.length;
  const num_visits = iter.visits.length;
  const max_depth = iter.visits[num_visits-1].depth;

  return (
    <table>
      <tbody>
      <tr>
        <td>Visit:</td>
        <td align="right">{iter_index+1}</td>
        <td>/</td>
        <td>{num_iters}</td>
        <td>
          <NAVArrow
            className="miniarrow"
            max_index={num_iters}
            index={iter_index}
            update={(i) => props.updateIterIndex(i)}
            delta={-1}
            alt="left"
            src={leftarrow}
          />
        </td>
        <td>
          <NAVArrow
            className="miniarrow"
            max_index={num_iters}
            index={iter_index}
            update={(i) => props.updateIterIndex(i)}
            delta={+1}
            alt="right"
            src={rightarrow}
          />
        </td>
      </tr>
      <tr>
        <td>Depth:</td>
        <td align="right">{depth}</td>
        <td>/</td>
        <td>{max_depth}</td>
        <td>
          <NAVArrow
            className="miniarrow"
            max_index={num_visits}
            index={visit_index}
            update={(i) => props.updateVisitIndex(i)}
            delta={-1}
            alt="left"
            src={leftarrow}
          />
        </td>
        <td>
          <NAVArrow
            className="miniarrow"
            max_index={num_visits}
            index={visit_index}
            update={(i) => props.updateVisitIndex(i)}
            delta={+1}
            alt="right"
            src={rightarrow}
          />
        </td>
      </tr>
      </tbody>
    </table>
  );
}

function displayBar(x, y) {
  if (y === 0) return displayBar(x, 1);
  const q = x / (y===0 ? 1 : y);
  const p = 100.0 * q;

  const title = x.toFixed(2);
  return (
    <div className="range">
      <span className="progress" title={title} style={{width: p + '%'}} />
    </div>
  );
}

function MCTSValues(props) {
  const move_index = props.move_index;
  const iter_index = props.iter_index;
  const visit_index = props.visit_index;
  const history = props.history;
  const move = history[move_index];
  const iter = move.iters[iter_index];
  const visit = iter.visits[visit_index];

  const render_children = () => {
    let text = []
    for (const child of visit.children) {
      text.push((
        <tr key={child.action}>
          <td className="vert">
            <div className="range">
              {child.action}
            </div>
          </td>
          <td className="vert">{displayBar(child.rP, visit.rP_sum)}</td>
          <td className="vert">{displayBar(child.dir, visit.dir_sum)}</td>
          <td className="vert">{displayBar(child.P, visit.P_sum)}</td>
          <td className="vert">{displayBar(child.V, visit.V_sum)}</td>
          <td className="vert">{displayBar(child.N, visit.N_sum)}</td>
          <td className="vert">{displayBar(child.PUCT, visit.PUCT_sum)}</td>
        </tr>
      ));
    }
    return text;
  }

  return (
    <table className="collapsed"><tbody>
    <tr>
      <td className="vert">Move</td>
      <td className="vert">Eval</td>
      <td className="vert">Dir</td>
      <td className="vert">P</td>
      <td className="vert">V</td>
      <td className="vert">N</td>
      <td className="vert">PUCT</td>
    </tr>
    { render_children() }
    </tbody></table>
  );
}

function MCTSDisplay(props) {
  const move_index = props.move_index;
  const iter_index = props.iter_index;
  const visit_index = props.visit_index;
  const history = props.history;
  const move = history[move_index];
  const iter = move.iters[iter_index];
  const visit = iter.visits[visit_index];

  const orig_board = move.board;
  let board = visit.board;
  let new_board_arr = [];
  for (let i=0; i<board.length; i++) {
    const c = board.charAt(i);
    const d = orig_board.charAt(i);
    if (d === '.') {
      if (c === 'R') {
        new_board_arr.push('r');
      } else if (c === 'Y') {
        new_board_arr.push('y');
      } else {
        new_board_arr.push(c);
      }
    } else {
      new_board_arr.push(c);
    }
  }
  const new_board = new_board_arr.join('');

  const renderRows = () => {
    const className = "square16";
    let rows = [];
    for (let i = 5; i >= 0; i--) {
      rows.push(<BoardRow className={className} rowName="row16" board={new_board} row={i} key={i}/>);
    }
    return rows;
  }

  return (
    <span>
      {renderRows()}
    </span>
  );
}


class MCTSHistory extends Component {
  render() {
    const props = this.props;
    const move_index = props.move_index;
    const iter_index = props.iter_index;
    const visit_index = props.visit_index;
    const history = props.history;
    const move = history[move_index];

    return (
      <table className="center">
        <tbody>
        <tr>
          <td>
            <MCTSNav
              updateIterIndex={(i) => this.props.updateIterIndex(i)}
              updateVisitIndex={(i) => this.props.updateVisitIndex(i)}
              history={history}
              move_index={move_index}
              move={move}
              iter_index={iter_index}
              visit_index={visit_index}
            />
          </td>
          <td>
            <MCTSValues
              move_index={move_index}
              iter_index={iter_index}
              visit_index={visit_index}
              history={history}
            />
          </td>
          <td>
            <MCTSDisplay
              history={history}
              move_index={move_index}
              iter_index={iter_index}
              visit_index={visit_index}
            />
          </td>
        </tr>
        </tbody>
      </table>
    );
  }
}

class Body extends Component {
  constructor(props) {
    super(props);
    this.state = {
      move_index: props.move_index,
      iter_index: props.iter_index,
      visit_index: props.visit_index,
    };
  }

  updateMoveIndex(move_index) {
    this.setState({
      move_index: move_index,
      iter_index: 0,
      visit_index: 0,
    });
  }

  updateIterIndex(iter_index) {
    this.setState({
      iter_index: iter_index,
      visit_index: 0,
    });
  }

  updateVisitIndex(visit_index) {
    this.setState({
      visit_index: visit_index,
    });
  }

  render() {
    const history = this.props.history;
    if (history.length === 0) {
      return "";
    }
    const player_index = this.props.player_index;
    const move_index = this.state.move_index;
    const iter_index = this.state.iter_index;
    const visit_index = this.state.visit_index;
    return (
      <div className="center">
        <MyColor
          player_index={player_index}
        />
        <GameHistory
          updateMoveIndex={(m) => this.updateMoveIndex(m)}
          history={history}
          move_index={move_index}
          body={this}
        />
        <MCTSHistory
          updateIterIndex={(m) => this.updateIterIndex(m)}
          updateVisitIndex={(m) => this.updateVisitIndex(m)}
          history={history}
          move_index={move_index}
          iter_index={iter_index}
          visit_index={visit_index}
        />
      </div>
    );
  }
}

export default Body;
