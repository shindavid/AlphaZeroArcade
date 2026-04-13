#pragma once

#include "alpha0/SearchResults.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "core/ActionPrinter.hpp"
#include "search/VerboseDataBase.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
struct VerboseData : public generic::VerboseDataBase {
  using Game = Spec::Game;
  using Move = Game::Move;
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using IO = Game::IO;
  using PolicyTensor = PolicyEncoding::Tensor;
  using SearchResults = alpha0::SearchResults<Spec>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using ActionPrinter = core::ActionPrinter<Game>;

  VerboseData(const PolicyTensor& p, const SearchResults& s, int n)
      : action_policy(p), mcts_results(s), n_rows_to_display_(n) {}

  PolicyTensor action_policy;
  SearchResults mcts_results;

  boost::json::object to_json() const override;
  void to_terminal_text() const override;

 private:
  const int n_rows_to_display_;
  auto build_action_data(ActionPrinter&) const;
};

}  // namespace alpha0

#include "inline/alpha0/VerboseData.inl"
