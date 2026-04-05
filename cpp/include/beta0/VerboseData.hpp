#pragma once

#include "beta0/SearchResults.hpp"
#include "core/ActionPrinter.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "search/VerboseDataBase.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct VerboseData : public generic::VerboseDataBase {
  using Game = EvalSpec::Game;
  using IO = Game::IO;
  using Move = Game::Move;
  using SearchResults = beta0::SearchResults<EvalSpec>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using ActionPrinter = core::ActionPrinter<Game>;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using PolicyTensor = PolicyEncoding::Tensor;

  VerboseData(const PolicyTensor& p, const SearchResults& s, int n)
      : action_policy(p), mcts_results(s), n_rows_to_display_(n) {}

  PolicyTensor action_policy;
  SearchResults mcts_results;

  boost::json::object to_json() const;
  void to_terminal_text() const;

 private:
  auto build_action_data(ActionPrinter&) const;

  const int n_rows_to_display_;
};

}  // namespace beta0

#include "inline/beta0/VerboseData.inl"
