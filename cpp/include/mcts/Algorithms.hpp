#pragma once

#include "mcts/ActionSelector.hpp"
#include "search/Constants.hpp"
#include "search/GeneralContext.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/TraitsTypes.hpp"

namespace mcts {

template <typename Traits>
class Algorithms {
 public:
  using Game = Traits::Game;
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using SearchResults = Traits::SearchResults;
  using ManagerParams = Traits::ManagerParams;
  using LookupTable = search::LookupTable<Traits>;
  using TraitsTypes = search::TraitsTypes<Traits>;

  using ActionSelector = mcts::ActionSelector<Traits>;
  using GeneralContext = search::GeneralContext<Traits>;
  using SearchContext = search::SearchContext<Traits>;

  using RootInfo = GeneralContext::RootInfo;
  using Visitation = TraitsTypes::Visitation;

  using GameResults = Game::GameResults;
  using IO = Game::IO;
  using StateHistory = Game::StateHistory;
  using Symmetries = Game::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;

  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using ValueArray = Game::Types::ValueArray;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  static void pure_backprop(SearchContext& context, const ValueArray& value);
  static void virtual_backprop(SearchContext& context);
  static void undo_virtual_backprop(SearchContext& context);
  static void standard_backprop(SearchContext& context, bool undo_virtual);
  static void short_circuit_backprop(SearchContext& context);

  static bool should_short_circuit(const Edge* edge, const Node* child);
  static bool more_search_iterations_needed(const GeneralContext&, const Node* root);
  static void init_root_info(GeneralContext&, search::RootInitPurpose);
  static int get_best_child_index(const SearchContext& context);

  static void to_results(const GeneralContext&, SearchResults&);
  static void print_visit_info(const SearchContext&);

 private:
  static void load_action_symmetries(const Node* root, core::action_t* actions, SearchResults&);
  static void prune_policy_target(group::element_t inv_sym, const GeneralContext&, SearchResults&);
  static void validate_search_path(const SearchContext& context);
  static void print_action_selection_details(const SearchContext& context,
                                             const ActionSelector& selector, int argmax_index);
};

}  // namespace mcts

#include "inline/mcts/Algorithms.inl"
