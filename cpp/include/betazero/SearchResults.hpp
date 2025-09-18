#pragma once

#include "alphazero/SearchResults.hpp"
#include "core/concepts/GameConcept.hpp"

#include <boost/json.hpp>

namespace beta0 {

template <core::concepts::Game Game>
struct SearchResults : alpha0::SearchResults<Game> {
  using Base = alpha0::SearchResults<Game>;
  using ActionValueTensor = Base::ActionValueTensor;

  ActionValueTensor action_value_uncertainties;

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/betazero/SearchResults.inl"
