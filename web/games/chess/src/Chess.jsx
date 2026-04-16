import './Chess.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';
import { Chessboard } from 'react-chessboard';

// Parse a UCI string like "e2e4" into { from: "e2", to: "e4", promotion?: "q" }
function parseUci(uci) {
  if (!uci || typeof uci !== 'string') return null;
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    promotion: uci.length > 4 ? uci[4] : undefined,
  };
}

// Build a UCI string from from/to/promotion
function toUci(from, to, promotion) {
  return from + to + (promotion || '');
}

const PROMOTION_PIECES = ['q', 'r', 'b', 'n'];

export default class ChessApp extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      selectedSquare: null,
      pendingPromotion: null,  // { from, to } when awaiting promo choice
    };
  }

  // Override: keep board as a FEN string instead of Array.from()
  handleStartGame(payload) {
    this.historyBuffer.clear();
    this.setState({
      loading: false,
      board: payload.board,
      lastTurn: null,
      lastMove: null,
      seatAssignments: Array.from(payload.seat_assignments),
      playerNames: Array.from(payload.player_names),
      resultCodes: null,
      mySeat: payload.my_seat,
      history: new Map(),
    });
  }

  // Override: keep board as a FEN string instead of Array.from()
  handleStateUpdate(payload) {
    const newHistory = new Map(this.historyBuffer);
    this.setState({
      board: payload.board,
      lastTurn: payload.seat,
      lastMove: payload.last_move,
      verboseInfo: payload.verbose_info ? payload.verbose_info : this.state.verboseInfo,
      history: newHistory,
    });
  }

  seatToHtml = (seat) => {
    if (seat === 'W') return <span className="chess-seat white-seat">&#9812;</span>;
    if (seat === 'B') return <span className="chess-seat black-seat">&#9818;</span>;
    return seat;
  };

  // Build a set of legal destination squares for quick lookup
  getLegalMovesFrom(square) {
    const { legalMoves } = this.state;
    if (!legalMoves) return [];
    return legalMoves
      .filter(uci => uci.slice(0, 2) === square)
      .map(uci => uci.slice(2, 4));
  }

  // Check if a specific move (from, to) is legal, possibly with promotion
  findLegalMove(from, to) {
    const { legalMoves } = this.state;
    if (!legalMoves) return [];
    return legalMoves.filter(uci =>
      uci.slice(0, 2) === from && uci.slice(2, 4) === to
    );
  }

  handleSquareClick = (square) => {
    if (!this.gameActive()) return;

    const { selectedSquare, legalMoves } = this.state;

    if (selectedSquare) {
      // Second click: try to make a move
      const matching = this.findLegalMove(selectedSquare, square);
      if (matching.length > 0) {
        if (matching.length > 1) {
          // Promotion: multiple matching moves (q, r, b, n)
          this.setState({ pendingPromotion: { from: selectedSquare, to: square } });
        } else {
          this.sendMove(matching[0]);
        }
        this.setState({ selectedSquare: null });
        return;
      }
    }

    // First click or re-click: select a piece with legal moves
    const hasLegal = legalMoves && legalMoves.some(uci => uci.slice(0, 2) === square);
    this.setState({ selectedSquare: hasLegal ? square : null });
  };

  onDrop = (sourceSquare, targetSquare) => {
    if (!this.gameActive()) return false;

    const matching = this.findLegalMove(sourceSquare, targetSquare);
    if (matching.length === 0) return false;

    if (matching.length > 1) {
      // Promotion via drag
      this.setState({ pendingPromotion: { from: sourceSquare, to: targetSquare } });
      return false;  // don't move yet
    }

    this.sendMove(matching[0]);
    this.setState({ selectedSquare: null });
    return true;
  };

  onPromotionSelect = (piece) => {
    const { pendingPromotion } = this.state;
    if (!pendingPromotion) return;

    const promo = piece[1].toLowerCase();  // "wQ" -> "q"
    const uci = toUci(pendingPromotion.from, pendingPromotion.to, promo);
    this.sendMove(uci);
    this.setState({ pendingPromotion: null, selectedSquare: null });
  };

  cancelPromotion = () => {
    this.setState({ pendingPromotion: null });
  };

  buildCustomSquareStyles() {
    const styles = {};
    const { selectedSquare, legalMoves, lastMove, proposedAction } = this.state;

    // Highlight last move
    if (lastMove) {
      const parsed = parseUci(lastMove);
      if (parsed) {
        styles[parsed.from] = { background: 'rgba(255, 255, 0, 0.4)' };
        styles[parsed.to] = { background: 'rgba(255, 255, 0, 0.4)' };
      }
    }

    // Highlight selected square
    if (selectedSquare) {
      styles[selectedSquare] = {
        ...(styles[selectedSquare] || {}),
        background: 'rgba(20, 120, 255, 0.5)',
      };

      // Highlight legal destinations
      const dests = this.getLegalMovesFrom(selectedSquare);
      for (const sq of dests) {
        styles[sq] = {
          ...(styles[sq] || {}),
          background: styles[sq]?.background
            ? 'radial-gradient(circle, rgba(0,0,0,0.3) 25%, transparent 25%)'
            : 'radial-gradient(circle, rgba(0,0,0,0.2) 25%, transparent 25%)',
        };
      }
    }

    // Highlight proposed action
    if (proposedAction) {
      const parsed = parseUci(proposedAction);
      if (parsed) {
        styles[parsed.from] = {
          ...(styles[parsed.from] || {}),
          boxShadow: 'inset 0 0 0 3px rgba(0, 200, 100, 0.7)',
        };
        styles[parsed.to] = {
          ...(styles[parsed.to] || {}),
          boxShadow: 'inset 0 0 0 3px rgba(0, 200, 100, 0.7)',
        };
      }
    }

    return styles;
  }

  buildCustomArrows() {
    const { proposedAction } = this.state;
    if (!proposedAction) return [];
    const parsed = parseUci(proposedAction);
    if (!parsed) return [];
    return [[parsed.from, parsed.to, 'rgba(0, 200, 100, 0.6)']];
  }

  renderPromotionDialog() {
    const { pendingPromotion, seatAssignments, currentTurn } = this.state;
    if (!pendingPromotion) return null;

    const color = seatAssignments[currentTurn] === 'W' ? 'w' : 'b';

    return (
      <div className="promotion-overlay" onClick={this.cancelPromotion}>
        <div className="promotion-dialog" onClick={e => e.stopPropagation()}>
          <div className="promotion-title">Promote to:</div>
          <div className="promotion-options">
            {PROMOTION_PIECES.map(p => (
              <button
                key={p}
                className="promotion-btn"
                onClick={() => this.onPromotionSelect(color + p.toUpperCase())}
              >
                {this.getPromotionSymbol(color, p)}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  getPromotionSymbol(color, piece) {
    const symbols = {
      wq: '\u2655', wr: '\u2656', wb: '\u2657', wn: '\u2658',
      bq: '\u265B', br: '\u265C', bb: '\u265D', bn: '\u265E',
    };
    return symbols[color + piece] || piece;
  }

  renderBoard() {
    const { board, mySeat } = this.state;
    if (!board) return null;

    const orientation = mySeat === 'B' ? 'black' : 'white';
    const customSquareStyles = this.buildCustomSquareStyles();
    const customArrows = this.buildCustomArrows();

    return (
      <div className="chess-board-wrapper">
        <Chessboard
          position={board}
          onPieceDrop={this.onDrop}
          onSquareClick={this.handleSquareClick}
          boardOrientation={orientation}
          customSquareStyles={customSquareStyles}
          customArrowColor="rgba(0, 200, 100, 0.6)"
          customArrows={customArrows}
          boardWidth={480}
          animationDuration={200}
          arePiecesDraggable={this.gameActive()}
        />
        {this.renderPromotionDialog()}
      </div>
    );
  }
}
