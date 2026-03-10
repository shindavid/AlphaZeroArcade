#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "x0/SearchResults.hpp"

#include <boost/json.hpp>

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct SearchResults : public x0::SearchResults<EvalSpec> {
  using Game = EvalSpec::Game;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;

  ActionValueTensor AV;
  ActionValueTensor AU;
  ActionValueTensor AQ;
  ActionValueTensor AQ_min;
  ActionValueTensor AQ_max;
  ActionValueTensor AW;
  PolicyTensor N;
  PolicyTensor RN;
  PolicyTensor pi;

  ValueArray Q_min;
  ValueArray Q_max;
  ValueArray W;

  core::seat_index_t seat;

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/beta0/SearchResults.inl"
