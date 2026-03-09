#include "games/chess/CompactState.hpp"

namespace a0achess {

inline chess::Bitboard CompactState::get(chess::PieceType piece_type,
                                         core::seat_index_t player) const {
  throw std::exception();  // TODO
  // chess::Bitboard pieces = all_pieces[player];
  // if (piece_type == chess::PieceType::PAWN) {
  //   pieces &= (pawns & kPawnMask);
  // } else if (piece_type == chess::PieceType::ROOK) {
  //   pieces &= orthogonal_movers & ~diagonal_movers;
  // } else if (piece_type == chess::PieceType::BISHOP) {
  //   pieces &= diagonal_movers & ~orthogonal_movers;
  // } else if (piece_type == chess::PieceType::QUEEN) {
  //   pieces &= orthogonal_movers & diagonal_movers;
  // } else if (piece_type == chess::PieceType::KNIGHT) {
  //   pieces &= ~(pawns | orthogonal_movers | diagonal_movers);
  // } else if (piece_type == chess::PieceType::KING) {
  //   pieces &= chess::Bitboard(1ULL << static_cast<int>(kings[player]));
  // }
  // return pieces;
}

}  // namespace a0achess
