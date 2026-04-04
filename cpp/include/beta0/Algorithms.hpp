#pragma once

#include "beta0/Calculations.hpp"
#include "beta0/PuctCalculator.hpp"
#include "core/ActionPrinter.hpp"
#include "core/ActionSymmetryTable.hpp"
#include "search/Constants.hpp"
#include "search/GameLogBase.hpp"
#include "search/GameLogViewParams.hpp"
#include "search/GeneralContext.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/TrainingInfoParams.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/Exceptions.hpp"
#include "x0/Algorithms.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
class Algorithms : public x0::Algorithms<Traits> {
 public:
  using Game = Traits::Game;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
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
  using Visitation = TraitsTypes::Visitation;

  using GeneralContext = search::GeneralContext<Traits>;
  using LookupTable = search::LookupTable<Traits>;
  using PuctCalculator = beta0::PuctCalculator<Traits>;
  using SearchContext = search::SearchContext<Traits>;
  using PolicyTensorData = search::GameLogBase<Traits>::PolicyTensorData;
  using ActionValueTensorData = search::GameLogBase<Traits>::ActionValueTensorData;
  using TrainingInfoParams = search::TrainingInfoParams<Traits>;

  using RootInfo = GeneralContext::RootInfo;

  using GameResults = Game::GameResults;
  using IO = Game::IO;
  using State = Game::State;
  using EvalSpec = Traits::EvalSpec;
  using PolicyEncoding = EvalSpec::PolicyEncoding;
  using InputFrame = EvalSpec::InputFrame;
  using Symmetries = EvalSpec::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;

  using ActionPrinter = core::ActionPrinter<Game>;
  using ActionSymmetryTable = core::ActionSymmetryTable<EvalSpec>;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using GameResultTensor = Game::Types::GameResultTensor;
  using player_bitset_t = Game::Types::player_bitset_t;

  using LogitValueArray = Game::Types::LogitValueArray;
  using NodeStableData = Traits::NodeStableData;

  using Calculations = beta0::Calculations<Game>;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using Array1D = LocalPolicyArray;
  using Array2D = LocalActionValueArray;
  using Mask = eigen_util::DArray<kMaxBranchingFactor, bool>;

  template <typename MutexProtectedFunc>
  static void backprop(SearchContext& context, Node* node, Edge* edge, MutexProtectedFunc&& func);

  static void init_node_stats_from_terminal(Node* node);
  static void update_node_stats(Node* node, bool undo_virtual);
  static void update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual);
  static void virtually_update_node_stats(Node* node) {
    throw util::CleanException("beta0 virtual updates not implemented");
  }
  static void virtually_update_node_stats_and_edge(Node* node, Edge* edge) {
    throw util::CleanException("beta0 virtual updates not implemented");
  }
  static void undo_virtual_update(Node* node, Edge* edge) {
    throw util::CleanException("beta0 virtual updates not implemented");
  }

  static void validate_search_path(const SearchContext& context) {}
  static bool should_short_circuit(const Edge* edge, const Node* child);
  static bool more_search_iterations_needed(const GeneralContext&, const Node* root);
  static void init_root_info(GeneralContext&, search::RootInitPurpose);
  static void init_root_edges(GeneralContext&);
  static int get_best_child_index(const SearchContext& context);
  static void load_evaluations(SearchContext& context);

  static void to_results(const GeneralContext&, SearchResults&);
  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord& full_record, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);

 protected:
  static void update_stats(NodeStats& stats, const Node* node, SearchContext& context);
  static void transform_policy(SearchContext&, LocalPolicyArray& P);
  static void add_dirichlet_noise(GeneralContext&, LocalPolicyArray& P);
  static void prune_policy_target(const GeneralContext&, SearchResults&);
  static void print_action_selection_details(const SearchContext& context,
                                             const PuctCalculator& selector, int argmax_index);
};

}  // namespace beta0

#include "inline/beta0/Algorithms.inl"
