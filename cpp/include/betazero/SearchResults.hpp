#pragma once

#include "alphazero/SearchResults.hpp"
#include "core/concepts/GameConcept.hpp"

#include <boost/json.hpp>

namespace beta0 {

template <core::concepts::Game Game>
struct SearchResults : alpha0::SearchResults<Game> {
  using Base = alpha0::SearchResults<Game>;
  using ActionValueTensor = Base::ActionValueTensor;
  using ValueArray = Base::ValueArray;

  ActionValueTensor action_value_uncertainties;

  // For each player, the min and max win-rate ever observed during search.
  ValueArray min_win_rates;
  ValueArray max_win_rates;

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/betazero/SearchResults.inl"
