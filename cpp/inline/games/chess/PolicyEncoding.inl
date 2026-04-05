#include "games/chess/PolicyEncoding.hpp"

#include "games/chess/MoveEncoder.hpp"

namespace a0achess {

PolicyEncoding::Index PolicyEncoding::to_index(const Move& move, const InputFrame* frame) {
  if (!frame) {
    throw std::invalid_argument("InputFrame pointer cannot be null for to_index");
  }

  chess::Color side_to_move = frame->cur_player == kWhite ? chess::Color::WHITE : chess::Color::BLACK;
  return Index{move_to_nn_idx(move, side_to_move)};
}

Move PolicyEncoding::to_move(const State& state, const Index& index) {
  return nn_idx_to_move(state, index[0]);
}

}  // namespace a0achess
