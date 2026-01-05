import React from 'react';
import { PortError, Loading, StatusBar, ActionButtons } from './SharedUI';
import { GameTreePanel } from './GameTreePanel';


// Make a uniform [{...}, {...}] from any value
function normalizeToRows(v) {
  if (v == null) return [];

  // already array of objects
  if (Array.isArray(v) && v.every(x => x && typeof x === 'object' && !Array.isArray(x))) {
    return v;
  }

  // object of arrays or scalars -> pivot to rows
  if (!Array.isArray(v) && typeof v === 'object') {
    const keys = Object.keys(v);
    if (!keys.length) return [];
    const len = Math.max(...keys.map(k => (Array.isArray(v[k]) ? v[k].length : 1)));
    const rows = [];
    for (let i = 0; i < len; i++) {
      const row = {};
      for (const k of keys) {
        row[k] = Array.isArray(v[k]) ? v[k][i] : v[k];
      }
      rows.push(row);
    }
    return rows;
  }

  // array of scalars -> one-column table
  if (Array.isArray(v)) {
    return v.map(x => ({ value: x }));
  }

  // scalar -> single row
  return [{ value: v }];
}

function renderArrayOfObjects(rows, renderers) {
  const cols = Array.from(rows.reduce((s, r) => { Object.keys(r).forEach(k => s.add(k)); return s; }, new Set()));
  return (
    <table className="verbose-table">
      <thead><tr>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            {cols.map(c => {
              const v = r[c];
              const fn = renderers[c] || fmt;
              return <td key={`${i}-${c}`}>{fn(v)}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function fmt(v) {
  if (v == null) return '';
  // TODO: Decide whether numeric formatting (e.g., toFixed(6)) should be handled in C++
  // instead of the frontend.
  if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(6);
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

export class GameAppBase extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: true,
      board: null,
      lastTurn: null,
      lastAction: null,
      legalMoves: [],
      seatAssignments: null,
      playerNames: null,
      resultCodes: null,
      mySeat: null,
      verboseInfo: null,
      currentTurn: null,
      proposedAction: null,
      history: new Map(),
    };
    this.socketRef = React.createRef();
    this.port = import.meta.env.VITE_BRIDGE_PORT;
  }

  // To be overridden by subclass: used for rendering Verbose Panel
  render_funcs = {};

  componentDidMount() {
    if (!this.port) return;
    const ws = new WebSocket(`ws://localhost:${this.port}`);
    this.socketRef.current = ws;
    ws.onopen = () => {
      console.log(`✅ WS connected to ${this.port}`);
      this.setState({ loading: false, wsClosed: false });
    };
    ws.onerror = e => console.error('🔴 WS error', e);
    ws.onclose = e => {
      console.warn('WebSocket closed:', e);
      this.setState({ wsClosed: true, loading: false });
    };
    ws.onmessage = e => {
      let msg;
      console.log(e);
      try { msg = JSON.parse(e.data) }
      catch (err) { return console.error('Bad JSON', err); }
      this.handleMessage(msg);
      this.setState({ loading: false });
    };
    window.myGame = this;
  }

  componentWillUnmount() {
    if (this.socketRef.current) this.socketRef.current.close();
  }

  handleMessage(msg) {
    if (msg.type === 'start_game') {
      this.handleStartGame(msg);
    } else if (msg.type === 'state_update') {
      this.handleStateUpdate(msg);
    } else if (msg.type === 'action_request') {
      this.handleActionRequest(msg);
    } else if (msg.type === 'game_end') {
      this.handleGameEnd(msg);
    } else if (msg.type === 'tree_node') {
      this.handleTreeNode(msg);
    } else if (msg.type === 'tree_node_batch') {
      this.handleTreeNodeBatch(msg.payloads);
    } else {
      console.warn('Unhandled message type:', msg.type);
    }
  }

  handleStartGame(payload) {
    this.setState({
      loading: false,
      board: Array.from(payload.board),
      lastTurn: null,
      lastAction: null,
      seatAssignments: Array.from(payload.seat_assignments),
      playerNames: Array.from(payload.player_names),
      resultCodes: null,
      mySeat: payload.my_seat,
      history: new Map(),
    });
  }

  handleStateUpdate(payload) {
    this.setState({
      board: Array.from(payload.board),
      lastTurn: payload.seat,
      lastAction: payload.last_action,
      verboseInfo: payload.verbose_info ? payload.verbose_info : this.state.verboseInfo,
    });
  }

  handleActionRequest(payload) {
    this.setState({
      legalMoves: payload.legal_moves,
      currentTurn: payload.seat,
      proposedAction: payload.proposed_action,
      verboseInfo: payload.verbose_info ? payload.verbose_info : this.state.verboseInfo,
    });
  }

  handleGameEnd(payload) {
    this.setState({
      resultCodes: payload.result_codes,
      legalMoves: [],
      proposedAction: null,
    });
  }

  handleTreeNode(payload) {
    this.setState((prevState) => {
      if (prevState.history.has(payload.index)) {
        return null;
      }
      const nextHistory = new Map(prevState.history);
      nextHistory.set(payload.index, payload);
      return { history: nextHistory };
    });
  }

  handleTreeNodeBatch(payloads) {
    this.setState(() => {
      const nextHistory = new Map();
      payloads.forEach(node => {
        nextHistory.set(node.index, node);
      });

      return { history: nextHistory };
    });
  }

  gameActive() {
    if (this.state.resultCodes) return false;
    const ws = this.socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    return true;
  }

  sendMove(action_index) {
    this.setState({
      legalMoves: [],
      proposedAction: null,
    });
    this.sendMsg({ type: 'make_move', seat: this.state.currentTurn, payload: { index: action_index } })
  }

  sendMsg(msg) {
    const ws = this.socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(msg));
  }

  handleResign = () => {
    this.setState({
      legalMoves: [],
    });
    this.sendMsg({ type: 'resign', seat: this.state.currentTurn });
  }

  handleNewGame = () => {
    this.sendMsg({ type: 'new_game' });
    this.setState({ loading: true, resultCodes: null });
  }

  seatToHtml = (seat) => {
    // Default implementation, can be overridden by subclasses
    return seat;
  }

  // To be implemented by subclass
  renderBoard() {
    return null;
  }

  VerbosePanel = ({ data }) => {
    if (!data) return null;

    const col_renderer_mapping = {};
    if (data.format_funcs) {
      Object.entries(data.format_funcs).forEach(([key, fn_str]) => {
        const fn = this[fn_str];
        if (fn) col_renderer_mapping[key] = fn;
      });
    }

    return (
      <div className="verbose-panel">
        {Object.entries(data).map(([section, value]) => {
          if (section === 'format_funcs') return null;
          const rows = normalizeToRows(value);
          if (!rows.length) return null;
          return (
            <div key={section} className="verbose-section">
              <h4>{section}</h4>
              {renderArrayOfObjects(rows, col_renderer_mapping)}
            </div>
          );
        })}
      </div>
    );
  };

  render() {
    if (!this.port) return <PortError port={this.port} />;
    if (this.state.wsClosed) {
      return <div className="container" style={{ color: 'red', padding: '2em', textAlign: 'center' }}>
        <div style={{ fontSize: '1.3em', fontWeight: 'bold' }}>Connection to backend lost.</div>
        <div style={{ marginTop: '1em' }}>Please restart the backend and reload the page.</div>
      </div>;
    }
    if (this.state.loading) return <Loading />;

    let resultCodes = this.state.resultCodes;
    let playerNames = this.state.playerNames;
    let seatAssignments = this.state.seatAssignments;
    let midGame = resultCodes === null;
    const seatAssignmentsHtml = seatAssignments ? seatAssignments.map(seat => this.seatToHtml(seat)) : seatAssignments;
    let currentSeat = this.state.currentTurn;

    return (
      <div className="container two-col">
        <div className="left-col">
          <StatusBar
            resultCodes={resultCodes}
            playerNames={playerNames}
            seatAssignments={seatAssignmentsHtml}
            currentSeat={currentSeat}
          />
          {this.renderBoard()}
          <ActionButtons
            onResign={this.handleResign}
            onNewGame={this.handleNewGame}
            midGame={midGame}
            loading={this.state.loading}
          />

          <GameTreePanel
            history={this.state.history}
            seatToHtml={this.seatToHtml}
          />
        </div>

        <this.VerbosePanel
          data={this.state.verboseInfo}
          render_funcs={this.render_funcs}
        />
      </div>
    );
  }
}
