#include "games/chess/PolicyEncoding.hpp"

#include "games/chess/MoveEncoder.hpp"

namespace a0achess {

PolicyEncoding::Index PolicyEncoding::to_index(const InputFrame& frame, const Move& move) {
  chess::Color side_to_move = frame.cur_player == kWhite ? chess::Color::WHITE : chess::Color::BLACK;
  return Index{move_to_nn_idx(move, side_to_move)};
}

Move PolicyEncoding::to_move(const InputFrame& frame, const Index& index) {
  return nn_idx_to_move(frame.to_state_unsafe(), index[0]);
}

}  // namespace a0achess
