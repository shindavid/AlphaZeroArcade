#pragma once

namespace core {

template <typename Game>
struct GameLogView {
  using BaseState = typename Game::BaseState;
  using ValueArray = typename Game::Types::ValueArray;
  using PolicyTensor = typename Game::Types::PolicyTensor;

  const BaseState* cur_pos;
  const BaseState* final_pos;
  const ValueArray* outcome;
  const PolicyTensor* policy;
  const PolicyTensor* next_policy;
};

}  // namespace core
