#pragma once

#include "alpha0/Algorithms.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/Gaussian1D.hpp"

namespace beta0 {

// CRTP base class
//
// This allows us to effectively have other Algorithms classes methods invoke beta0::Algorithms
// methods.
//
// To make this work, we need to follow the discipline of always invoking Derived::func() rather
// than simply func() within beta0::Algorithms methods.
template <search::concepts::Traits Traits, typename Derived>
class AlgorithmsBase : public alpha0::AlgorithmsBase<Traits, Derived> {
 public:
  using Base = alpha0::AlgorithmsBase<Traits, Derived>;
  friend class alpha0::AlgorithmsBase<Traits, Derived>;

  using ActionValueTensorData = Base::ActionValueTensorData;
  using Edge = Base::Edge;
  using Game = Base::Game;
  using GameLogCompactRecord = Base::GameLogCompactRecord;
  using GameLogFullRecord = Base::GameLogFullRecord;
  using GameLogView = Base::GameLogView;
  using GameLogViewParams = Base::GameLogViewParams;
  using GameResultTensor = Base::GameResultTensor;
  using GeneralContext = Base::GeneralContext;
  using LocalActionValueArray = Base::LocalActionValueArray;
  using LocalPolicyArray = Base::LocalPolicyArray;
  using LookupTable = Base::LookupTable;
  using ManagerParams = Base::ManagerParams;
  using Node = Base::Node;
  using NodeStats = Base::NodeStats;
  using PolicyTensor = Base::PolicyTensor;
  using PolicyTensorData = Base::PolicyTensorData;
  using PuctCalculator = Base::PuctCalculator;
  using RootInfo = Base::RootInfo;
  using SearchContext = Base::SearchContext;
  using SearchResults = Base::SearchResults;
  using State = Base::State;
  using TrainingInfo = Base::TrainingInfo;
  using TrainingInfoParams = Base::TrainingInfoParams;
  using ValueArray = Base::ValueArray;
  using player_bitset_t = Base::player_bitset_t;

  using LogitValueArray = Game::Types::LogitValueArray;
  using NodeStableData = Traits::NodeStableData;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  template <typename MutexProtectedFunc>
  static void backprop(SearchContext& context, Node* node, Edge* edge, MutexProtectedFunc&& func);

  static void init_node_stats_from_terminal(Node* node);
  static void init_node_stats_from_nn_eval(Node* node, bool undo_virtual);
  static void update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual);
  static void virtually_update_node_stats(Node* node) {}
  static void virtually_update_node_stats_and_edge(Node* node, Edge* edge) {}
  static void undo_virtual_update(Node* node, Edge* edge) {}

  static void validate_search_path(const SearchContext& context) {}
  static bool should_short_circuit(const Edge* edge, const Node* child) { return false; }
  static bool more_search_iterations_needed(const GeneralContext&, const Node* root);

  static int get_best_child_index(const SearchContext& context);
  static void load_evaluations(SearchContext& context);

  static void to_results(const GeneralContext&, SearchResults&);
  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord& full_record, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);

 protected:
  static void populate_XC(SearchContext& context, int* XC, int n);
};

template <search::concepts::Traits Traits>
struct Algorithms : public AlgorithmsBase<Traits, Algorithms<Traits>> {};

}  // namespace beta0

#include "inline/beta0/Algorithms.inl"
