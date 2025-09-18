#pragma once

#include "search/Constants.hpp"
#include "search/GameLogBase.hpp"
#include "search/GameLogViewParams.hpp"
#include "search/GeneralContext.hpp"
#include "search/PuctCalculator.hpp"
#include "search/SearchContext.hpp"
#include "search/TrainingInfoParams.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <ostream>
#include <vector>

namespace search {

// For now, we have alpha0::Algorithms and beta0::Algorithms inherit from this base class to reduce
// code duplication. In the future, as we specialize the algorithms, we may want to move functions
// from here to the derived classes.
//
// TODO: add some signatures to concepts::Algorithms:
//
// - write_to_training_info()
// - to_record()
// - compactify_record()
// - to_view()
template <search::concepts::Traits Traits>
class AlgorithmsBase {
 public:
  using Game = Traits::Game;
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using SearchResults = Traits::SearchResults;
  using ManagerParams = Traits::ManagerParams;
  using TrainingInfo = Traits::TrainingInfo;
  using GameLogCompactRecord = Traits::GameLogCompactRecord;
  using GameLogFullRecord = Traits::GameLogFullRecord;
  using GameLogView = Traits::GameLogView;
  using TraitsTypes = search::TraitsTypes<Traits>;
  using LookupTable = TraitsTypes::LookupTable;

  using PuctCalculator = search::PuctCalculator<Traits>;
  using GeneralContext = search::GeneralContext<Traits>;
  using SearchContext = search::SearchContext<Traits>;
  using TrainingInfoParams = search::TrainingInfoParams<Traits>;
  using TensorData = search::GameLogBase<Traits>::TensorData;
  using GameLogViewParams = search::GameLogViewParams<Traits>;

  using RootInfo = GeneralContext::RootInfo;
  using Visitation = TraitsTypes::Visitation;

  using GameResults = Game::GameResults;
  using IO = Game::IO;
  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using Symmetries = Game::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;

  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueTensor = Game::Types::ValueTensor;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using ValueArray = Game::Types::ValueArray;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using player_bitset_t = Game::Types::player_bitset_t;

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
  static void load_evaluations(SearchContext& context);

  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord&, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);

  static void to_results(const GeneralContext&, SearchResults&);
  static void print_visit_info(const SearchContext&);
  static void print_mcts_results(std::ostream& ss, const PolicyTensor& action_policy,
                                 const SearchResults& results, int n_rows_to_display);

 private:
  template <typename MutexProtectedFunc>
  static void update_stats(Node* node, LookupTable& lookup_table, MutexProtectedFunc&&);

  static void write_results(const GeneralContext&, const Node* root, group::element_t inv_sym,
                            SearchResults& results);
  static void validate_state(LookupTable& lookup_table, Node* node);  // NO-OP in release builds
  static void transform_policy(SearchContext&, LocalPolicyArray& P);
  static void add_dirichlet_noise(GeneralContext&, LocalPolicyArray& P);
  static void load_action_symmetries(const GeneralContext&, const Node* root,
                                     core::action_t* actions, SearchResults&);
  static void prune_policy_target(group::element_t inv_sym, const GeneralContext&, SearchResults&);
  static void validate_search_path(const SearchContext& context);
  static void print_action_selection_details(const SearchContext& context,
                                             const PuctCalculator& selector, int argmax_index);
  static bool extract_policy_target(const SearchResults* mcts_results, PolicyTensor& target);
};

}  // namespace search

#include "inline/search/AlgorithmsBase.inl"
