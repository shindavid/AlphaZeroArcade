#include "games/chess/InputTensorizor.hpp"

namespace chess {

inline InputTensorizor::Tensor InputTensorizor::tensorize(group::element_t sym) {
  Tensor tensor;
  tensor.setZero();

  auto& buf = this->buffer();

  auto& latest_state = buf.back().state;
  auto& latest_board = latest_state.board;
  auto& castlings = latest_board.castlings();

  // - Plane 104 (0-based) filled with 1 if we can castle queenside.
  // - Plane 105 filled with ones if we can castle kingside.
  // - Plane 106 filled with ones if they can castle queenside.
  // - Plane 107 filled with ones if they can castle kingside.
  if (castlings.we_can_000()) {
    tensor.chip<0>(kAuxPlaneBaseIndex).setConstant(1.0f);
  }
  if (castlings.we_can_00()) {
    tensor.chip<0>(kAuxPlaneBaseIndex + 1).setConstant(1.0f);
  }
  if (castlings.they_can_000()) {
    tensor.chip<0>(kAuxPlaneBaseIndex + 2).setConstant(1.0f);
  }
  if (castlings.they_can_00()) {
    tensor.chip<0>(kAuxPlaneBaseIndex + 3).setConstant(1.0f);
  }

  // Plane 108 filled with ones if we are black to move
  if (latest_board.flipped()) {
    tensor.chip<0>(kAuxPlaneBaseIndex + 4).setConstant(1.0f);
  }

  // Plane 109 filled with the rule50 ply count
  tensor.chip<0>(kAuxPlaneBaseIndex + 5).setConstant(latest_state.rule50_ply);

  // Plane 110 is all zeros

  // Plane 111 is all ones
  tensor.chip<0>(kAuxPlaneBaseIndex + 7).setConstant(1.0f);

  auto num_states = std::min(buf.size(), static_cast<size_t>(kNumStatesToEncode));
  for (size_t i = 0; i < num_states; i++) {
    const auto& state = (buf.end() - 1 - i)->state;
    const auto& b = state.board;

    const int base = i * kPlanesPerBoard;
    fill_plane(tensor, base + 0, (b.ours() & b.pawns()).as_int());
    fill_plane(tensor, base + 1, (b.ours() & b.knights()).as_int());
    fill_plane(tensor, base + 2, (b.ours() & b.bishops()).as_int());
    fill_plane(tensor, base + 3, (b.ours() & b.rooks()).as_int());
    fill_plane(tensor, base + 4, (b.ours() & b.queens()).as_int());
    fill_plane(tensor, base + 5, (b.ours() & b.kings()).as_int());

    fill_plane(tensor, base + 6, (b.theirs() & b.pawns()).as_int());
    fill_plane(tensor, base + 7, (b.theirs() & b.knights()).as_int());
    fill_plane(tensor, base + 8, (b.theirs() & b.bishops()).as_int());
    fill_plane(tensor, base + 9, (b.theirs() & b.rooks()).as_int());
    fill_plane(tensor, base + 10, (b.theirs() & b.queens()).as_int());
    fill_plane(tensor, base + 11, (b.theirs() & b.kings()).as_int());

    if (state.count_repetitions() >= 1) {
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

}  // namespace chess
