#pragma once

#include "core/concepts/GameConcept.hpp"

#include <boost/json.hpp>

namespace beta0 {

template <core::concepts::Game Game>
struct SearchResults {
  using ActionMask = Game::Types::ActionMask;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using GameResultTensor = Game::Types::GameResultTensor;

  ActionMask valid_actions;

  PolicyTensor P;
  PolicyTensor pi;
  ActionValueTensor AQ;
  ActionValueTensor AW;

  GameResultTensor R;
  ValueArray Q;
  ValueArray Q_min;
  ValueArray Q_max;
  ValueArray W;

  ActionSymmetryTable action_symmetry_table;
  core::action_mode_t action_mode;
  bool trivial;  // all actions are symmetrically equivalent
  bool provably_lost = false;

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/betazero/SearchResults.inl"
