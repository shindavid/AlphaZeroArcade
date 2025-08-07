import React from 'react';
import { PortError, Loading, StatusBar, ActionButtons } from './SharedUI';


export class GameAppBase extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: true,
      gameEnd: null,
      board: null,
      lastTurn: null,
      lastAction: null,
      legalMoves: [],
      seatAssignments: null,
      playerNames: null,
      mySeat: null,
    };
    this.socketRef = React.createRef();
    this.port = import.meta.env.VITE_BRIDGE_PORT;
  }

  componentDidMount() {
    if (!this.port) return;
    const ws = new WebSocket(`ws://localhost:${this.port}`);
    this.socketRef.current = ws;
    ws.onopen = () => console.log(`✅ WS connected to ${this.port}`);
    ws.onerror = e => console.error('🔴 WS error', e);
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

  // To be overridden by subclass
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
      gameEnd: null,
      lastTurn: null,
      lastAction: null,
      seatAssignments: Array.from(payload.seat_assignments),
      playerNames: Array.from(payload.player_names),
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
      gameEnd: payload,
    });
  }

  gameActive() {
    if (this.state.gameEnd) return false;
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
  };

  handleNewGame = () => {
    this.sendMsg({ type: 'new_game' });
    this.setState({ loading: true, gameEnd: null });
  };

  // To be implemented by subclass
  renderBoard() {
    return null;
  }

  render() {
    if (!this.port) return <PortError port={this.port} />;
    if (this.state.loading) return <Loading />;

    let gameEnd = this.state.gameEnd;
    let playerNames = this.state.playerNames;
    let seatAssignments = this.state.seatAssignments;
    return (
      <div className="container" style={{ minHeight: '600px', justifyContent: 'flex-start' }}>
        <StatusBar gameEnd={gameEnd} playerNames={playerNames} seatAssignments={seatAssignments} />
        {this.renderBoard()}
        <ActionButtons
          onResign={this.handleResign}
          onNewGame={this.handleNewGame}
          gameEnd={this.state.gameEnd}
          loading={this.state.loading}
        />
      </div>
    );
  }
}
