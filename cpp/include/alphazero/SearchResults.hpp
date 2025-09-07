#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/Game.hpp"

#include <boost/json.hpp>

namespace alpha0 {

template <core::concepts::Game Game>
struct SearchResults {
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

}  // namespace alpha0

#include "inline/alphazero/SearchResults.inl"
