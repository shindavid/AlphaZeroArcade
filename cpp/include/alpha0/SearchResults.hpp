#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "x0/SearchResults.hpp"

#include <boost/json.hpp>

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct SearchResults : public x0::SearchResults<EvalSpec> {
  using Base = x0::SearchResults<EvalSpec>;
  using Game = EvalSpec::Game;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;

  PolicyTensor policy_target;
  PolicyTensor counts;
  PolicyTensor AQs;  // s indicates only for the current seat
  PolicyTensor AQs_sq;
  ActionValueTensor AV;

  boost::json::object to_json() const;
};

}  // namespace alpha0

#include "inline/alpha0/SearchResults.inl"
