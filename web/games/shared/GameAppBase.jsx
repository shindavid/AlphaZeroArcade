import React from 'react';
import { PortError, Loading, StatusBar, ActionButtons } from './SharedUI';

function VerbosePanel({ data }) {
  if (!data) return null;
  const { action_mode, cpu_pos_eval, actions, ...rest } = data;

  return (
    <div className="verbose-panel">
      <h3>Verbose</h3>

      {/* Scalars */}
      {action_mode !== undefined && (
        <div className="verbose-section">
          <h4>action_mode</h4>
          <div className="kv-line"><span className="k">action_mode</span><span className="v">{action_mode}</span></div>
        </div>
      )}

      {/* cpu_pos_eval: object of arrays */}
      {cpu_pos_eval && (
        <div className="verbose-section">
          <h4>cpu_pos_eval</h4>
          {renderObjectOfArrays(cpu_pos_eval)}
        </div>
      )}

      {/* actions: array of objects -> table */}
      {Array.isArray(actions) && actions.length > 0 && (
        <div className="verbose-section">
          <h4>actions</h4>
          {renderArrayOfObjects(actions)}
        </div>
      )}

      {/* any other leftover keys */}
      {Object.keys(rest).length > 0 && (
        <div className="verbose-section">
          <h4>other</h4>
          <pre className="verbose-pre">{JSON.stringify(rest, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

// obj: { key: array | scalar }
function renderObjectOfArrays(obj) {
  const cols = Object.keys(obj);
  const rowCount = Math.max(...cols.map(k => (Array.isArray(obj[k]) ? obj[k].length : 1)));
  return (
    <table className="verbose-table">
      <thead><tr>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>
        {Array.from({ length: rowCount }).map((_, r) => (
          <tr key={r}>
            {cols.map(c => (
              <td key={c}>
                {Array.isArray(obj[c]) ? fmt(obj[c][r]) : fmt(obj[c])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// rows: [{...}]
function renderArrayOfObjects(rows) {
  const cols = Array.from(
    rows.reduce((s, r) => { Object.keys(r).forEach(k => s.add(k)); return s; }, new Set())
  );
  return (
    <table className="verbose-table">
      <thead><tr>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            {cols.map(c => <td key={c}>{fmt(row[c])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function fmt(v) {
  if (v === undefined) return '';
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
    };
    this.socketRef = React.createRef();
    this.port = import.meta.env.VITE_BRIDGE_PORT;
  }

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
      try { msg = JSON.parse(e.data) }
      catch (err) { return console.error('Bad JSON', err); }
      this.handleMessage(msg);
      this.setState({ loading: false });
    };
  }

  componentWillUnmount() {
    if (this.socketRef.current) this.socketRef.current.close();
  }

  handleMessage(msg) {
    if (msg.type === 'start_game') {
      this.handleStartGame(msg.payload);
    } else if (msg.type === 'state_update') {
      this.handleStateUpdate(msg.payload);
    } else if (msg.type === 'action_request') {
      this.handleActionRequest(msg.payload);
    } else if (msg.type === 'game_end') {
      this.handleGameEnd(msg.payload);
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
    });
  }

  handleStateUpdate(payload) {
    console.log('State update:', payload);
    this.setState({
      board: Array.from(payload.board),
      lastTurn: payload.seat,
      lastAction: payload.last_action,
      verboseInfo: payload.verbose_info ? payload.verbose_info : null,
    });
  }

  handleActionRequest(payload) {
    this.setState({
      legalMoves: payload.legal_moves,
    });
  }

  handleGameEnd(payload) {
    this.setState({
      resultCodes: payload.result_codes,
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
    });
    this.sendMsg({ type: 'make_move', payload: { index: action_index } })
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
    this.sendMsg({ type: 'resign' });
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
    let verboseInfo = this.state.verboseInfo;

    let midGame = resultCodes === null;
    const seatAssignmentsHtml = seatAssignments ? seatAssignments.map(this.seatToHtml) : seatAssignments;
    return (
      <div className="container two-col">
        <div className="left-col">
          <StatusBar
            resultCodes={resultCodes}
            playerNames={playerNames}
            seatAssignments={seatAssignmentsHtml}
          />
          {this.renderBoard()}
          <ActionButtons
            onResign={this.handleResign}
            onNewGame={this.handleNewGame}
            midGame={midGame}
            loading={this.state.loading}
          />
        </div>

        <VerbosePanel data={this.state.verboseInfo} />
      </div>
    );
  }
}
