#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "x0/SearchResults.hpp"

#include <boost/json.hpp>

namespace beta0 {

template <core::concepts::Game Game>
struct SearchResults : public x0::SearchResults<Game> {
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;

  ActionValueTensor AV;
  ActionValueTensor AW;
  ActionValueTensor AQ;
  ActionValueTensor AU;
  PolicyTensor pi;

  ValueArray Q_min;
  ValueArray Q_max;
  ValueArray W;

  core::seat_index_t seat;

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/beta0/SearchResults.inl"
