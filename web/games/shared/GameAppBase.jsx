import React from 'react';
import { PortError, Loading, StatusBar, ActionButtons } from './SharedUI';


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
    this.setState({
      board: Array.from(payload.board),
      lastTurn: payload.seat,
      lastAction: payload.last_action,
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

    let midGame = resultCodes === null;
    const seatAssignmentsHtml = seatAssignments ? seatAssignments.map(this.seatToHtml) : seatAssignments;
    return (
      <div className="container" style={{ minHeight: '600px', justifyContent: 'flex-start' }}>
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
    );
  }
}
