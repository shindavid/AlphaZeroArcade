import React, { Component } from "react";
import { useEffect, useRef } from 'react';
import { RangeSlider } from 'toolcool-range-slider';
import 'toolcool-range-slider';
import leftarrow from '../images/left.svg';
import rightarrow from '../images/right.svg';
import redcircle from '../images/red.svg';
import yellowcircle from '../images/yellow.svg';

function Square(props) {
  const color = props.color;
  let className = props.className;
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

function BoardRow(props) {
  const board = props.board;
  const row = props.row;
  const className = props.className;
  const rowName = props.rowName;
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

function Board(props) {
  const className = props.className;
  const rowName = props.rowName;
  const board = props.board;

  let rows = [];
  for (let i = 5; i >= 0; i--) {
    rows.push(<BoardRow className={className} rowName={rowName} board={board} row={i} key={i}/>);
  }
  return rows;
}

function NAVArrow(props) {
  const index = props.index;
  const max_index = props.max_index;
  const delta = props.delta;
  const className = props.className;

  const alt = delta < 0 ? "left" : "right";
  const src = delta < 0 ? leftarrow : rightarrow;

  const new_index = index + delta;
  const hidden = (new_index === -1) || (new_index === max_index);

  if (hidden) {
    return (
      <span className={className} />
    );
  }
  return (
    <span className={className}>
      <img alt={alt} src={src} onClick={() => props.update(props.index + props.delta)} />
    </span>
  );
}

function LeftNAVArrow(props) {
  return (
    <NAVArrow index={props.index} max_index={props.max_index} delta={-1} className="arrow" update={props.update} />
  );
}

function RightNAVArrow(props) {
  return (
    <NAVArrow index={props.index} max_index={props.max_index} delta={+1} className="arrow" update={props.update} />
  );
}

function MiniLeftNAVArrow(props) {
  return (
    <NAVArrow index={props.index} max_index={props.max_index} delta={-1} className="miniarrow" update={props.update} />
  );
}

function MiniRightNAVArrow(props) {
  return (
    <NAVArrow index={props.index} max_index={props.max_index} delta={+1} className="miniarrow" update={props.update} />
  );
}

function MyColor(props) {
  const player_index = props.player_index;
  const my_color = player_index === 0 ? redcircle : yellowcircle;
  return (
    <div className="centertext">
      My Color:&nbsp;
      <span className="minicircle center"><img src={my_color} alt="color" /></span>
      <br/><br/>
    </div>
  );
}

function GameHistory(props) {
  const history = props.history;
  const move_index = props.move_index;
  const move = history[move_index];

  return (
    <table className="center"><tbody>
    <tr>
      <td>
        <LeftNAVArrow max_index={history.length} index={move_index} update={(i) => props.updateMoveIndex(i)} />
      </td>
      <td>
        <span>
          <Board className="square32" rowName="row32" board={move.board }/>
        </span>
      </td>
      <td>
        <RightNAVArrow max_index={history.length} index={move_index} update={(i) => props.updateMoveIndex(i)} />
      </td>
    </tr>
    </tbody></table>
  );
}

function NAVSlider(props) {
  const sliderRef = useRef();

  useEffect(() => {
    const slider = sliderRef.current;
    const onChange = (evt) => {
      if (document.activeElement !== slider) return;
      console.log('NAVSlider.' + props.name + '.onChange() ' + evt.detail.value);
      props.update(evt.detail.value);
    }
    slider?.addEventListener('change', onChange);
    return () => {
      slider?.removeEventListener('change', onChange);
    }
  }, []);

  const disabled = props.max === 1;

  return (
    <toolcool-range-slider
      disabled={disabled}
      min={0}
      max={props.max-1}
      value={props.value}
      step="1"
      ref={sliderRef}
    />
  );
}

function MCTSNav(props) {
  const history = props.history;
  const move_index = props.move_index;
  const move = history[move_index];
  if (move.iters.length === 0) return "";

  const iter_index = props.iter_index;
  const top_index = props.top_index;
  const bot_index = props.bot_index;

  const iter = move.iters[iter_index];
  const top = iter.visits[top_index];
  const bot = iter.visits[bot_index];
  const top_depth = top.depth;
  const bot_depth = bot.depth;

  const num_iters = move.iters.length;
  const num_visits = iter.visits.length;
  const max_depth = iter.visits[num_visits-1].depth;

  const top_board = top.board;
  const board_history = move.board_to_visits[top_board];

  console.log('MCTSNav() ' + top_board + ' | ' + board_history[0].board);

  return (
    <table className="center">
      <tbody>
      <tr>
        <td align="right">MCTS Iter:</td>
        <td align="right" style={{width:25}}>{iter_index+1}</td>
        <td>/</td>
        <td style={{width:25}}>{num_iters}</td>
        <td>
          <MiniLeftNAVArrow max_index={num_iters} index={iter_index} update={(i) => props.updateIterIndex(i)} />
        </td>
        <td>
          <MiniRightNAVArrow max_index={num_iters} index={iter_index} update={(i) => props.updateIterIndex(i)} />
        </td>
        <td style={{paddingLeft:10}}>
          <NAVSlider name="iter" max={num_iters} value={iter_index} update={(i) => props.updateIterIndex(i)} />
        </td>
      </tr>
      <tr>
        <td align="right">Top Depth:</td>
        <td align="right" style={{width:25}}>{top_depth}</td>
        <td>/</td>
        <td style={{width:25}}>{max_depth}</td>
        <td>
          <MiniLeftNAVArrow max_index={num_visits} index={top_index} update={(i) => props.updateTopIndex(i)} />
        </td>
        <td>
          <MiniRightNAVArrow max_index={num_visits} index={top_index} update={(i) => props.updateTopIndex(i)} />
        </td>
        <td style={{paddingLeft:10}}>
          {/*<NAVSlider name="top" max={num_visits} value={top_index} update={(i) => props.updateTopIndex(i)} />*/}
        </td>
      </tr>
      <tr>
        <td align="right">Bot Depth:</td>
        <td align="right" style={{width:25}}>{bot_depth}</td>
        <td>/</td>
        <td style={{width:25}}>{max_depth}</td>
        <td>
          <MiniLeftNAVArrow max_index={num_visits} index={bot_index} update={(i) => props.updateBotIndex(i)} />
        </td>
        <td>
          <MiniRightNAVArrow max_index={num_visits} index={bot_index} update={(i) => props.updateBotIndex(i)} />
        </td>
        <td style={{paddingLeft:10}}>
          {/*<NAVSlider name="bot" max={num_visits} value={bot_index} update={(i) => props.updateBotIndex(i)} />*/}
        </td>
      </tr>
      <tr>
        <td align="right">Top Scan:</td>
        <td align="right" style={{width:25}}>{top.board_visit_num + 1}</td>
        <td>/</td>
        <td style={{width:25}}>{board_history.length}</td>
        <td>
          <MiniLeftNAVArrow
            max_index={board_history.length}
            index={top.board_visit_num}
            update={(i) => props.updateTopScanIndex2(i)}
            //update={(i) => props.updateTopScanIndex(top_board, board_history, i)}
          />
        </td>
        <td>
          <MiniRightNAVArrow
            max_index={board_history.length}
            index={top.board_visit_num}
            update={(i) => props.updateTopScanIndex2(i)}
            //update={(i) => props.updateTopScanIndex(top_board, board_history, i)}
          />
        </td>
        <td style={{paddingLeft:10}}>
          <NAVSlider
            name="scan"
            max={board_history.length}
            value={top.board_visit_num}
            update={(i) => props.updateTopScanIndex2(i)}
            //update={(i) => props.updateTopScanIndex(top_board, board_history, i)}
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

function PolicyTable(props) {
  const visit = props.visit;

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
    <div style={{height:200}}>
    <table className="collapsed"><tbody>
    <tr>
      <td className="vert">Move</td>
      <td className="vert">NN.P</td>
      <td className="vert">Diric</td>
      <td className="vert">P</td>
      <td className="vert">V</td>
      <td className="vert">N</td>
      <td className="vert">PUCT</td>
    </tr>
    { render_children() }
    </tbody></table>
    </div>
  );
}

function ValueTable(props) {
  const visit = props.visit;

  const player = visit.player;
  const eval_total = visit.eval.reduce((a, b) => a+b, 0);
  const value_sum_total = visit.value_sum.reduce((a, b) => a+b, 0);

  return (
    <table className="collapsed"><tbody>
    <tr>
      <td className="vert">Color</td>
      <td className="vert">CurP</td>
      <td className="vert">NN.V</td>
      <td className="vert">V</td>
    </tr>
    <tr>
      <td className="vert">
        <span className="minicircle center">
          <img src={redcircle} alt="color" />
        </span>
      </td>
      <td className="vert">{player===0 ? "X" : ""}</td>
      <td className="vert">{displayBar(visit.eval[0], eval_total)}</td>
      <td className="vert">{displayBar(visit.value_sum[0], value_sum_total)}</td>
    </tr>
    <tr>
      <td className="vert">
        <span className="minicircle center">
          <img src={yellowcircle} alt="color" />
        </span>
      </td>
      <td className="vert">{player===1 ? "X" : ""}</td>
      <td className="vert">{displayBar(visit.eval[1], eval_total)}</td>
      <td className="vert">{displayBar(visit.value_sum[1], value_sum_total)}</td>
    </tr>
    </tbody></table>
  );
}

function MiniBoard(props) {
  const move = props.move;
  const visit = props.visit;

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

  return (
    <span>
      <span>
        <Board className="square16" rowName="row16" board={new_board} />
      </span>
    </span>
  );
}

function MCTSDisplay(props) {
  const move = props.move;
  const top = props.top;
  const bot = props.bot;

  if (move.iters.length === 0) return "";

  return (
    <table className="center">
      <tbody>
      <tr>
        <td>Top</td>
        <td><PolicyTable visit={top} /></td>
        <td><ValueTable visit={top} /></td>
        <td><MiniBoard move={move} visit={top} /></td>
      </tr>
      <tr>
        <td>Bottom</td>
        <td><PolicyTable visit={bot} /></td>
        <td><ValueTable visit={bot} /></td>
        <td><MiniBoard move={move} visit={bot} /></td>
      </tr>
      </tbody>
    </table>
  );
}

class Body extends Component {
  constructor(props) {
    super(props);
    this.state = {
      move_index: props.move_index,
      iter_index: props.iter_index,
      top_index: props.top_index,
      bot_index: props.bot_index,
    };
  }

  updateMoveIndex(move_index) {
    const iter_index = 0;
    const move = this.props.history[move_index];
    let bot_index = 0;
    if (move.iters.length) {
      const iter = move.iters[iter_index];
      bot_index = iter.visits.length - 1;
    }

    this.setState({
      move_index: move_index,
      iter_index: 0,
      top_index: 0,
      bot_index: bot_index,
    });
  }

  updateIterIndex(iter_index) {
    const move = this.props.history[this.state.move_index];
    const iter = move.iters[iter_index];
    const bot_index = iter.visits.length - 1;

    console.log('updateIterIndex()');
    // console.log(' move_index: ' + this.state.move_index);
    // console.log(' iter_index: ' + iter_index);
    // console.log(' iter.visits.length: ' + iter.visits.length);
    // console.log(' bot_index: ' + bot_index);

    this.setState({
      iter_index: iter_index,
      top_index: 0,
      bot_index: bot_index,
    });
  }

  updateTopIndex(top_index) {
    console.log('updateTopIndex()');
    let bot_index = this.state.bot_index;
    if (top_index > bot_index) {
      bot_index = top_index;
    }
    this.setState({
      top_index: top_index,
      bot_index: bot_index,
    });
  }

  updateBotIndex(bot_index) {
    console.log('updateBotIndex()');
    this.setState({
      bot_index: bot_index,
    });
  }

  updateTopScanIndex(board, board_history, scan_index) {
    console.log('updateTopScanIndex(' + board + ', ' + board_history.length + ', ' + scan_index + ')');

    const visit = board_history[scan_index];
    const iter = visit.parent;

    console.log('updateTopScanIndex(scan_index=' + scan_index + ')  iter_index:' + iter.index +
      ' iter.visits.length:' + iter.visits.length + ' visit.depth:' + visit.depth + 'visit.board:' + visit.board);

    this.setState({
      iter_index: iter.index,
      top_index: visit.depth - 1,
      bot_index: iter.visits.length - 1,
    });
  }

  updateTopScanIndex2(scan_index) {
    const move_index = this.state.move_index;
    const iter_index = this.state.iter_index;
    const top_index = this.state.top_index;

    const move = this.props.history[move_index];
    const iter = move.iters[iter_index];
    const top = iter.visits[top_index];

    const top_board = top.board;
    const board_history = move.board_to_visits[top_board];

    const visit = board_history[scan_index];
    const new_iter = visit.parent;

    this.setState({
      iter_index: new_iter.index,
      top_index: visit.depth - 1,
      bot_index: new_iter.visits.length - 1,
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
    const top_index = this.state.top_index;
    const bot_index = this.state.bot_index;

    const move = history[move_index];
    const iter = move.iters[iter_index];
    const top = iter?.visits[top_index];
    const bot = iter?.visits[bot_index];

    console.log('render() move_index:' + move_index + ' iter_index:' + iter_index + ' iter.visits.length:' +
      iter.visits.length + ' top_index:' + top_index + ' bot_index:' + bot_index);

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
        <MCTSDisplay move={move} top={top} bot={bot} />
        <MCTSNav
          updateIterIndex={(m) => this.updateIterIndex(m)}
          updateTopIndex={(m) => this.updateTopIndex(m)}
          updateBotIndex={(m) => this.updateBotIndex(m)}
          updateTopScanIndex={(b, h, i) => this.updateTopScanIndex(b, h, i)}
          updateTopScanIndex2={(i) => this.updateTopScanIndex2(i)}
          history={history}
          move_index={move_index}
          iter_index={iter_index}
          top_index={top_index}
          bot_index={bot_index}
        />
      </div>
    );
  }
}

export default Body;
