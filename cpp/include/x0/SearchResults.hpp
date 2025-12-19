#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"

#include <boost/json.hpp>

namespace x0 {

template <core::concepts::Game Game>
struct SearchResults {
  using ActionMask = Game::Types::ActionMask;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using GameResultTensor = Game::Types::GameResultTensor;

  ActionMask valid_actions;
  PolicyTensor policy_target;
  PolicyTensor P;
  ValueArray Q;
  GameResultTensor R;

  ActionSymmetryTable action_symmetry_table;
  core::action_mode_t action_mode;
  bool trivial;  // all actions are symmetrically equivalent
  bool provably_lost = false;

  boost::json::object to_json() const;
};

}  // namespace x0

#include "inline/x0/SearchResults.inl"
