#include "games/chess/InputTensorizor.hpp"

namespace a0achess {

inline InputTensorizor::Tensor InputTensorizor::tensorize(group::element_t sym) {
  Tensor tensor;
  tensor.setZero();

  auto& buf = this->buffer();

  auto& latest_state = buf.back().state;
  auto castlings = latest_state.castling_rights();
  chess::Color us = latest_state.side_to_move();
  chess::Color them = ~us;

  // - Plane 104 (0-based) filled with 1 if we can castle queenside.
  // - Plane 105 filled with ones if we can castle kingside.
  // - Plane 106 filled with ones if they can castle queenside.
  // - Plane 107 filled with ones if they can castle kingside.
  if (castlings.has(us, chess::Board::CastlingRights::Side::QUEEN_SIDE)) {
    tensor.chip<0>(kAuxPlaneBaseIndex).setConstant(1.0f);
  }
  if (castlings.has(us, chess::Board::CastlingRights::Side::KING_SIDE)) {
    tensor.chip<0>(kAuxPlaneBaseIndex + 1).setConstant(1.0f);
  }
  if (castlings.has(them, chess::Board::CastlingRights::Side::QUEEN_SIDE)) {
    tensor.chip<0>(kAuxPlaneBaseIndex + 2).setConstant(1.0f);
  }
  if (castlings.has(them, chess::Board::CastlingRights::Side::KING_SIDE)) {
    tensor.chip<0>(kAuxPlaneBaseIndex + 3).setConstant(1.0f);
  }

  // Plane 108 filled with ones if we are black to move
  if (us == chess::Color::BLACK) {
    tensor.chip<0>(kAuxPlaneBaseIndex + 4).setConstant(1.0f);
  }

  // Plane 109 filled with the rule50 ply count
  tensor.chip<0>(kAuxPlaneBaseIndex + 5).setConstant(latest_state.half_move_clock());

  // Plane 110 is all zeros

  // Plane 111 is all ones
  tensor.chip<0>(kAuxPlaneBaseIndex + 7).setConstant(1.0f);

  auto num_states = std::min(buf.size(), static_cast<size_t>(kNumStatesToEncode));
  for (size_t i = 0; i < num_states; i++) {
    const auto& state = (buf.end() - 1 - i)->state;

    const int base = i * kPlanesPerBoard;
    fill_plane(tensor, base + 0, state.pieces_bb(chess::PieceType::PAWN, us));
    fill_plane(tensor, base + 1, state.pieces_bb(chess::PieceType::KNIGHT, us));
    fill_plane(tensor, base + 2, state.pieces_bb(chess::PieceType::BISHOP, us));
    fill_plane(tensor, base + 3, state.pieces_bb(chess::PieceType::ROOK, us));
    fill_plane(tensor, base + 4, state.pieces_bb(chess::PieceType::QUEEN, us));
    fill_plane(tensor, base + 5, state.pieces_bb(chess::PieceType::KING, us));

    fill_plane(tensor, base + 6, state.pieces_bb(chess::PieceType::PAWN, them));
    fill_plane(tensor, base + 7, state.pieces_bb(chess::PieceType::KNIGHT, them));
    fill_plane(tensor, base + 8, state.pieces_bb(chess::PieceType::BISHOP, them));
    fill_plane(tensor, base + 9, state.pieces_bb(chess::PieceType::ROOK, them));
    fill_plane(tensor, base + 10, state.pieces_bb(chess::PieceType::QUEEN, them));
    fill_plane(tensor, base + 11, state.pieces_bb(chess::PieceType::KING, them));

    if (state.is_repetition(1)) {
      tensor.chip<0>(base + 12).setConstant(1.0f);
    }
  }

  return tensor;
}

inline void InputTensorizor::fill_plane(Tensor& tensor, int plane_idx, uint64_t mask) {
  while (mask) {
    const int sq = std::countr_zero(mask);
    tensor(plane_idx, sq / kBoardDim, sq % kBoardDim) = 1.0f;
    mask &= (mask - 1);
  }
}

}  // namespace a0achess
