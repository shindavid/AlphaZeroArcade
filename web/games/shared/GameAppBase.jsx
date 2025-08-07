import React from 'react';
import { PortError, Loading, StatusBar, ActionButtons } from './SharedUI';


export class GameAppBase extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: true,
      gameEnd: null,
      turn: null,
      board: null,
      legalMoves: [],
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
    if (msg.type === 'state_update') {
      this.handleStateUpdate(msg.payload);
    } else if (msg.type === 'game_end') {
      this.handleGameEnd(msg.payload);
    } else {
      console.warn('Unhandled message type:', msg.type);
    }
  }

  handleStateUpdate(payload) {
    this.setState({
      board: Array.from(payload.board),
      turn: payload.turn,
      gameEnd: null,
      legalMoves: payload.legal_moves || [],
    });
  }

  handleGameEnd(payload) {
    this.setState({
      board: Array.from(payload.board),
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
    const ws = this.socketRef.current;
    ws.send(JSON.stringify({ type: 'make_move', payload: { index: action_index } }));
  }

  handleResign = () => {
    const ws = this.socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'resign' }));
  };

  handleNewGame = () => {
    const ws = this.socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'new_game' }));
    this.setState({ loading: true, gameEnd: null });
  };

  // To be implemented by subclass
  renderBoard() {
    return null;
  }

  render() {
    if (!this.port) return <PortError port={this.port} />;
    if (this.state.loading) return <Loading />;
    return (
      <div className="container" style={{ minHeight: '600px', justifyContent: 'flex-start' }}>
        <StatusBar gameEnd={this.state.gameEnd} turn={this.state.turn} />
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
