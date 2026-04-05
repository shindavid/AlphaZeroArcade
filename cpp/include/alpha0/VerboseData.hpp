#pragma once

#include "alpha0/SearchResults.hpp"
#include "core/ActionPrinter.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "search/VerboseDataBase.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
struct VerboseData : public generic::VerboseDataBase {
  using Game = EvalSpec::Game;
  using Move = Game::Move;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using IO = Game::IO;
  using PolicyTensor = PolicyEncoding::Tensor;
  using SearchResults = alpha0::SearchResults<EvalSpec>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using ActionPrinter = core::ActionPrinter<Game>;

  VerboseData(const PolicyTensor& p, const SearchResults& s, int n)
      : action_policy(p), mcts_results(s), n_rows_to_display_(n) {}

  PolicyTensor action_policy;
  SearchResults mcts_results;

  boost::json::object to_json() const;
  void to_terminal_text() const;

 private:
  const int n_rows_to_display_;
  auto build_action_data(ActionPrinter&) const;
};

}  // namespace alpha0

#include "inline/alpha0/VerboseData.inl"
