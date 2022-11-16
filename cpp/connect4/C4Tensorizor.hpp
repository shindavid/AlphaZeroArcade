#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

#include <common/TensorizorConcept.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>

namespace c4 {

class ReflectionTransform {
public:
  void transform_input(torch::Tensor& input) const;
  void transform_policy(torch::Tensor& policy) const;
};

class Tensorizor {
public:
  static constexpr auto kShape = std::array{kNumPlayers, kNumColumns, kNumRows};

  void clear() {}
  void receive_state_change(const GameState& state, common::action_index_t action_index) {}

  void tensorize(torch::Tensor tensor, const GameState& state) { state.tensorize(tensor); }

  // auto get_symmetries(const GameState& state);


private:

};

}  // namespace c4

static_assert(common::TensorizorConcept<c4::Tensorizor>);

#include <connect4/C4TensorizorINLINES.cpp>
