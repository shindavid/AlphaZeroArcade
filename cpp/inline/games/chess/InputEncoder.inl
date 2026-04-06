#include "games/chess/InputEncoder.hpp"

#include "games/chess/Constants.hpp"

namespace a0achess {

inline InputEncoder::Tensor InputEncoder::encode(group::element_t sym) {
  Tensor tensor;
  tensor.setZero();

  auto& buf = this->buffer();
  const InputFrame& latest_frame = buf.back().frame;

  core::seat_index_t us = latest_frame.cur_player;
  core::seat_index_t them = 1 - us;

  using Bit = CastlingRightBit;
  Bit our_ks = us == kWhite ? Bit::kWhiteKingSide : Bit::kBlackKingSide;
  Bit our_qs = us == kWhite ? Bit::kWhiteQueenSide : Bit::kBlackQueenSide;
  Bit their_ks = them == kWhite ? Bit::kWhiteKingSide : Bit::kBlackKingSide;
  Bit their_qs = them == kWhite ? Bit::kWhiteQueenSide : Bit::kBlackQueenSide;

  if (latest_frame.castling_rights & (1 << our_qs)) {
    tensor.chip<0>(kAuxPlaneBaseIndex + kAuxPlaneOurQueenSideCastle).setConstant(1.0f);
  }
  if (latest_frame.castling_rights & (1 << our_ks)) {
    tensor.chip<0>(kAuxPlaneBaseIndex + kAuxPlaneOurKingSideCastle).setConstant(1.0f);
  }
  if (latest_frame.castling_rights & (1 << their_qs)) {
    tensor.chip<0>(kAuxPlaneBaseIndex + kAuxPlaneTheirQueenSideCastle).setConstant(1.0f);
  }
  if (latest_frame.castling_rights & (1 << their_ks)) {
    tensor.chip<0>(kAuxPlaneBaseIndex + kAuxPlaneTheirKingSideCastle).setConstant(1.0f);
  }

  if (us == kBlack) {
    tensor.chip<0>(kAuxPlaneBaseIndex + kAuxPlaneBlackToMove).setConstant(1.0f);
  }

  tensor.chip<0>(kAuxPlaneBaseIndex + kAuxPlaneRule50PlyCount)
    .setConstant(latest_frame.half_move_clock);

  tensor.chip<0>(kAuxPlaneBaseIndex + kAuxPlaneAllOnes).setConstant(1.0f);

  for (size_t i = 0; i < this->size(); i++) {
    const InputFrame& frame = (buf.end() - 1 - i)->frame;

    int b = i * kPlanesPerBoard;

    fill_plane(tensor, b++, frame.get(chess::PieceType::PAWN, us).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::KNIGHT, us).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::BISHOP, us).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::ROOK, us).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::QUEEN, us).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::KING, us).getBits());

    fill_plane(tensor, b++, frame.get(chess::PieceType::PAWN, them).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::KNIGHT, them).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::BISHOP, them).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::ROOK, them).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::QUEEN, them).getBits());
    fill_plane(tensor, b++, frame.get(chess::PieceType::KING, them).getBits());
  }

  return tensor;
}

inline void InputEncoder::undo() {
  Base::undo();
  current_hash_ = 0;  // safety measure to ensure we don't encode after an undo()
}

inline void InputEncoder::update(const GameState& state) {
  Base::update(state);
  current_hash_ = state.hash();
}

inline void InputEncoder::temp_update(const InputFrame& frame) {
  Base::update(frame);
  current_hash_ = 0;  // safety measure to ensure we don't call eval_key() after temp_update()
}

inline InputEncoder::EvalKey InputEncoder::eval_key() const {
  RELEASE_ASSERT(current_hash_);  // safety measure to ensure we cal
  return current_hash_;
}

inline void InputEncoder::fill_plane(Tensor& tensor, int plane_idx, uint64_t mask) {
  while (mask) {
    const int sq = std::countr_zero(mask);
    tensor(plane_idx, sq / kBoardDim, sq % kBoardDim) = 1.0f;
    mask &= (mask - 1);
  }
}

}  // namespace a0achess
