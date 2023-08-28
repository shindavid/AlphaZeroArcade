#pragma once

#include <common/SquareBoardSymmetryBase.hpp>
#include <core/TensorizorConcept.hpp>
#include <games/tictactoe/Constants.hpp>
#include <games/tictactoe/GameState.hpp>
#include <util/EigenUtil.hpp>

namespace tictactoe {

using TensorizorBase = common::SquareBoardSymmetryBase<
          GameState,
          eigen_util::Shape<kNumPlayers, kBoardDimension, kBoardDimension>>;

/*
 * All transforms have a templated transform_input() method. This generality exists to support unit tests, which
 * use non-bool input tensors.
 */
class Tensorizor : public TensorizorBase
{
 public:
  using InputTensor = TensorizorBase::InputTensor;

  void clear() {}
  void receive_state_change(const GameState& state, core::action_t action) {}
  void tensorize(InputTensor& tensor, const GameState& state) const { state.tensorize(tensor); }
};

}  // namespace tictactoe

static_assert(core::TensorizorConcept<tictactoe::Tensorizor, tictactoe::GameState>);
