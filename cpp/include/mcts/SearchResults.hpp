#pragma once

#include "core/BasicTypes.hpp"

#include <boost/json.hpp>

namespace mcts {

template <typename Traits>
struct SearchResults {
  using Game = Traits::Game;

  using ActionMask = Game::Types::ActionMask;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using ValueTensor = Game::Types::ValueTensor;

  ActionMask valid_actions;
  PolicyTensor counts;
  PolicyTensor policy_target;
  PolicyTensor policy_prior;
  PolicyTensor Q;
  PolicyTensor Q_sq;
  ActionValueTensor action_values;
  ValueArray win_rates;
  ValueTensor value_prior;
  ActionSymmetryTable action_symmetry_table;
  core::action_mode_t action_mode;
  bool trivial;  // all actions are symmetrically equivalent
  bool provably_lost = false;

  boost::json::object to_json() const;
};

}  // namespace mcts

#include "inline/mcts/SearchResults.inl"
