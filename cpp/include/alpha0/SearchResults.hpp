#pragma once

#include "core/concepts/GameConcept.hpp"
#include "x0/SearchResults.hpp"

#include <boost/json.hpp>

namespace alpha0 {

template <core::concepts::Game Game>
struct SearchResults : public x0::SearchResults<Game> {
  using Base = x0::SearchResults<Game>;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;

  PolicyTensor counts;
  PolicyTensor AQs;  // s indicates only for the current seat
  PolicyTensor AQs_sq;
  ActionValueTensor AV;
  bool provably_lost = false;

  boost::json::object to_json() const;
};

}  // namespace alpha0

#include "inline/alpha0/SearchResults.inl"
