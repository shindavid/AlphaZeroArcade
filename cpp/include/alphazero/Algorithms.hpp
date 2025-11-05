#pragma once

#include "search/Constants.hpp"
#include "search/GameLogBase.hpp"
#include "search/GameLogViewParams.hpp"
#include "search/GeneralContext.hpp"
#include "search/LookupTable.hpp"
#include "search/PuctCalculator.hpp"
#include "search/SearchContext.hpp"
#include "search/TrainingInfoParams.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <ostream>
#include <vector>

namespace alpha0 {

// CRTP base class
//
// This allows us to effectively have alpha0::Algorithms methods invoke beta0::Algorithms methods.
//
// To make this work, we need to follow the discipline of always invoking Derived::func() rather
// than simply func() within alpha0::Algorithms methods.
template <search::concepts::Traits Traits, typename Derived>
class AlgorithmsBase {
 public:
  using Game = Traits::Game;
  using Edge = Traits::Edge;
  using NodeStats = Traits::NodeStats;
  using SearchResults = Traits::SearchResults;
  using ManagerParams = Traits::ManagerParams;
  using TrainingInfo = Traits::TrainingInfo;
  using GameLogCompactRecord = Traits::GameLogCompactRecord;
  using GameLogFullRecord = Traits::GameLogFullRecord;
  using GameLogView = Traits::GameLogView;

  using TraitsTypes = search::TraitsTypes<Traits>;

  using GameLogViewParams = search::GameLogViewParams<Traits>;
  using Node = TraitsTypes::Node;
  using StateHistory = TraitsTypes::StateHistory;
  using Visitation = TraitsTypes::Visitation;

  using GeneralContext = search::GeneralContext<Traits>;
  using LookupTable = search::LookupTable<Traits>;
  using PuctCalculator = search::PuctCalculator<Traits>;
  using SearchContext = search::SearchContext<Traits>;
  using PolicyTensorData = search::GameLogBase<Traits>::PolicyTensorData;
  using ActionValueTensorData = search::GameLogBase<Traits>::ActionValueTensorData;
  using TrainingInfoParams = search::TrainingInfoParams<Traits>;

  using RootInfo = GeneralContext::RootInfo;

  using GameResults = Game::GameResults;
  using IO = Game::IO;
  using State = Game::State;
  using Symmetries = Game::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;

  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using GameResultTensor = Game::Types::GameResultTensor;
  using player_bitset_t = Game::Types::player_bitset_t;

  class Backpropagator {
   public:
    Backpropagator(LookupTable& lookup_table) : lookup_table_(lookup_table) {}

    template <typename MutexProtectedFunc>
    void run(Node* node, Edge* edge, MutexProtectedFunc&& func);

   private:
    LookupTable& lookup_table_;
  };

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  static void init_edge_from_child(const GeneralContext&, Node* parent, Edge* edge) {}
  static void init_node_stats_from_terminal(Node* node);
  static void init_node_stats_from_nn_eval(Node* node, bool undo_virtual);
  static void update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual);
  static void virtually_update_node_stats(Node* node);
  static void virtually_update_node_stats_and_edge(Node* node, Edge* edge);
  static void undo_virtual_update(Node* node, Edge* edge);

  static void validate_search_path(const SearchContext& context);
  static bool should_short_circuit(const Edge* edge, const Node* child);
  static bool more_search_iterations_needed(const GeneralContext&, const Node* root);
  static void init_root_info(GeneralContext&, search::RootInitPurpose);
  static int get_best_child_index(const SearchContext& context);
  static void load_evaluations(SearchContext& context);

  static void to_results(const GeneralContext&, SearchResults&);
  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord&, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);

  static void print_visit_info(const SearchContext&);

 protected:
  static void update_stats(NodeStats& stats, const Node* node, LookupTable& lookup_table);
  static void write_results(const GeneralContext&, const Node* root, group::element_t inv_sym,
                            SearchResults& results);
  static void validate_state(LookupTable& lookup_table, Node* node);  // NO-OP in release builds
  static void transform_policy(SearchContext&, LocalPolicyArray& P);
  static void add_dirichlet_noise(GeneralContext&, LocalPolicyArray& P);
  static void load_action_symmetries(const GeneralContext&, const Node* root,
                                     core::action_t* actions, SearchResults&);
  static void prune_policy_target(group::element_t inv_sym, const GeneralContext&, SearchResults&);
  static void print_action_selection_details(const SearchContext& context,
                                             const PuctCalculator& selector, int argmax_index);
  static bool extract_policy_target(const SearchResults* mcts_results, PolicyTensor& target);
};

template <search::concepts::Traits Traits>
struct Algorithms : public AlgorithmsBase<Traits, Algorithms<Traits>> {};

}  // namespace alpha0

#include "inline/alphazero/Algorithms.inl"
