#pragma once

#include "core/ActionSymmetryTable.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"

#include <boost/json.hpp>

namespace x0 {

template <core::concepts::EvalSpec EvalSpec>
struct SearchResults {
  using Game = EvalSpec::Game;
  using MoveSet = Game::Types::MoveSet;
  using ActionSymmetryTable = core::ActionSymmetryTable<EvalSpec>;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using GameResultTensor = Game::Types::GameResultTensor;
  using InputTensorizor = EvalSpec::InputTensorizor;
  using InputFrame = EvalSpec::InputFrame;

  InputFrame frame;
  MoveSet valid_moves;
  PolicyTensor P;
  ValueArray Q;
  GameResultTensor R;
  PolicyTensor pre_expanded_moves;

  ActionSymmetryTable action_symmetry_table;
  core::game_phase_t game_phase;
  bool trivial;  // all moves are symmetrically equivalent
  bool provably_lost = false;

  boost::json::object to_json() const;
};

}  // namespace x0

#include "inline/x0/SearchResults.inl"
