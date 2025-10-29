#pragma once

#include "alphazero/SearchResults.hpp"
#include "core/concepts/GameConcept.hpp"

#include <boost/json.hpp>

namespace beta0 {

template <core::concepts::Game Game>
struct SearchResults : alpha0::SearchResults<Game> {
  using Base = alpha0::SearchResults<Game>;
  using ActionValueTensor = Base::ActionValueTensor;
  using PolicyTensor = Base::PolicyTensor;
  using ValueArray = Base::ValueArray;

  PolicyTensor policy_posterior;
  ActionValueTensor action_value_uncertainties;

  // For each player, the min and max win-rate ever observed during search.
  ValueArray min_win_rates;
  ValueArray max_win_rates;

  // For each player, the max uncertainty ever observed during search.
  ValueArray max_uncertainties;

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/betazero/SearchResults.inl"
