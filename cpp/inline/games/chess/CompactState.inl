#include "games/chess/CompactState.hpp"

namespace a0achess {

inline chess::Bitboard CompactState::get(chess::PieceType piece_type,
                                         core::seat_index_t player) const {
  chess::Bitboard pieces = all_pieces[player];

  switch (piece_type.internal()) {
    case chess::PieceType::KING:
      return chess::Bitboard::fromSquare(chess::Square(static_cast<int>(kings[player])));
    case chess::PieceType::PAWN:
      return pawns & kPawnsMask & pieces;
    case chess::PieceType::QUEEN:
      return orthogonal_movers & diagonal_movers & pieces;
    case chess::PieceType::ROOK:
      return orthogonal_movers & ~diagonal_movers & pieces;
    case chess::PieceType::BISHOP:
      return diagonal_movers & ~orthogonal_movers & pieces;
    case chess::PieceType::KNIGHT: {
      chess::Bitboard king_bb =
        chess::Bitboard::fromSquare(chess::Square(static_cast<int>(kings[player])));
      chess::Bitboard actual_pawns = pawns & kPawnsMask & pieces;
      return pieces & ~(actual_pawns | orthogonal_movers | diagonal_movers | king_bb);
    }
    default:
      return chess::Bitboard(0);
  }
}

}  // namespace a0achess
