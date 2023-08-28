#pragma once

#include <common/SquareBoardSymmetryBase.hpp>
#include <core/TensorizorConcept.hpp>
#include <games/othello/Constants.hpp>
#include <games/othello/GameState.hpp>
#include <util/EigenUtil.hpp>

namespace othello {

using TensorizorBase = common::SquareBoardSymmetryBase<
    GameState,
    eigen_util::Shape<kNumPlayers, kBoardDimension, kBoardDimension>>;

/*
 * All transforms have a templated transform_input() method. This generality
 * exists to support unit tests, which use non-bool input tensors.
 */
class Tensorizor : public TensorizorBase {
 public:
  using InputTensor = TensorizorBase::InputTensor;

  void clear() {}
  void receive_state_change(const GameState& state, core::action_t action) {}
  void tensorize(InputTensor& tensor, const GameState& state) const {
    state.tensorize(tensor);
  }
};

}  // namespace othello

static_assert(core::TensorizorConcept<othello::Tensorizor, othello::GameState>);
