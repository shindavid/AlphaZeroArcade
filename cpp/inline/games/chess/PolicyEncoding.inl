#include "games/chess/PolicyEncoding.hpp"

#include "games/chess/Move.hpp"

#include <chess-library/include/chess.hpp>

namespace a0achess {

inline PolicyEncoding::Index PolicyEncoding::to_index(const InputFrame& frame, const Move& move) {
  return Index{kMoveEncodingTable.encode(move, frame.cur_player)};
}

inline Move PolicyEncoding::to_move(const State& state, const Index& index) {
  return kMoveEncodingTable.decode(index[0], state);
}

inline int MoveEncodingTable::encode(const Move& move, core::seat_index_t seat) const {
  chess::Square from_sq = move.from();
  chess::Square to_sq = move.to();

  if (seat == a0achess::kBlack) {
    from_sq = chess::Square(from_sq.index() ^ 56);
    to_sq = chess::Square(to_sq.index() ^ 56);
  }

  const Data& data = data_[from_sq.index()];
  if (move.typeOf() == Move::PROMOTION && move.promotionType() != chess::PieceType::KNIGHT) {
    int a = data.promo_offsets[to_sq.index() - from_sq.index() - 7];
    int b = int(chess::PieceType::QUEEN) - int(move.promotionType());
    return kNumNonPromoMoves + a + b;
  } else {
    return data.offset + tail_popcount(data.bitmap, to_sq.index());
  }
}

inline chess::Move MoveEncodingTable::decode(int index, const chess::Board& board) const {
  using namespace chess;

  uint16_t move_index = untyped_move_indices_[index];

  Color color = board.sideToMove();
  if (color == Color::BLACK) {
    move_index ^= (7 << 3) + (7 << 9);
  }

  chess::Move move(move_index);
  chess::Square from_sq = move.from();
  chess::Square to_sq = move.to();

  Piece from_piece = board.at(from_sq);
  Piece to_piece = board.at(to_sq);

  // Castling is encoded as king-captures-own-rook
  if (from_piece.type() == PieceType::KING && to_piece == Piece(color, PieceType::ROOK)) {
    return chess::Move(move_index + chess::Move::CASTLING);
  }

  // En passant
  if (from_piece.type() == PieceType::PAWN && to_sq == board.enpassantSq()) {
    return chess::Move(move_index + chess::Move::ENPASSANT);
  }

  // Knight promotion special case
  if (from_piece.type() == PieceType::PAWN && chess::Rank::back_rank(to_sq.rank(), ~color)) {
    return chess::Move(move_index | (chess::Move::PROMOTION));
  }

  return move;
}

}  // namespace a0achess
