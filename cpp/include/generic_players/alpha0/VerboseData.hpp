#pragma once

#include "core/BasicTypes.hpp"
#include "search/VerboseDataBase.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace generic::alpha0 {

struct VerboseRow {
  core::action_t action;
  float prior;
  float posterior;
  float counts;
  float modified;
};

template <search::concepts::Traits Traits_>
struct VerboseData : public VerboseDataBase {
  using Traits = Traits_;
  using Game = Traits::Game;
  using IO = Game::IO;
  using PolicyTensor = Game::Types::PolicyTensor;
  using SearchResults = ::alpha0::SearchResults<Game>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  VerboseData(int n_rows_to_display) : n_rows_to_display_(n_rows_to_display) {}

  PolicyTensor action_policy;
  SearchResults mcts_results;

  boost::json::object to_json() const;
  void to_terminal_text() const;
  void set(const PolicyTensor& policy, const SearchResults& results);

 private:
  struct Table {
    std::vector<VerboseRow> rows_sorted;
    core::action_mode_t action_mode{};
    std::vector<float> net_value_v;
    std::vector<float> win_rates_v;

    void clear() {
      rows_sorted.clear();
      action_mode = core::action_mode_t{};
      net_value_v.clear();
      win_rates_v.clear();
    }
  };

  mutable Table table_;
  int n_rows_to_display_ = -1;

  void build_table() const;
};

}  // namespace generic::alpha0

#include "inline/generic_players/alpha0/VerboseData.inl"
