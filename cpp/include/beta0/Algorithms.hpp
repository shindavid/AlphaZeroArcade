#pragma once

#include "beta0/Calculations.hpp"
#include "beta0/PuctCalculator.hpp"
#include "core/ActionPrinter.hpp"
#include "core/ActionResponse.hpp"
#include "core/ActionSymmetryTable.hpp"
#include "search/Constants.hpp"
#include "search/GameLogBase.hpp"
#include "search/GameLogViewParams.hpp"
#include "search/GeneralContext.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/TrainingDataWriter.hpp"
#include "search/concepts/SearchSpecConcept.hpp"
#include "util/Exceptions.hpp"
#include "x0/Algorithms.hpp"

namespace beta0 {

template <search::concepts::SearchSpec SearchSpec>
class Algorithms : public x0::Algorithms<SearchSpec> {
 public:
  using Game = SearchSpec::Game;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using Edge = SearchSpec::Edge;
  using NodeStats = SearchSpec::NodeStats;
  using SearchResults = SearchSpec::SearchResults;
  using ManagerParams = SearchSpec::ManagerParams;
  using TrainingInfo = SearchSpec::TrainingInfo;
  using GameLogCompactRecord = SearchSpec::GameLogCompactRecord;
  using GameLogFullRecord = SearchSpec::GameLogFullRecord;
  using GameLogView = SearchSpec::GameLogView;

  using GameLogViewParams = search::GameLogViewParams<SearchSpec>;
  using Node = SearchSpec::Node;
  using Visitation = search::SearchContext<SearchSpec>::Visitation;

  using GeneralContext = search::GeneralContext<SearchSpec>;
  using LookupTable = search::LookupTable<SearchSpec>;
  using PuctCalculator = beta0::PuctCalculator<SearchSpec>;
  using SearchContext = search::SearchContext<SearchSpec>;
  using PolicyTensorData = search::GameLogBase<SearchSpec>::PolicyTensorData;
  using ActionValueTensorData = search::GameLogBase<SearchSpec>::ActionValueTensorData;

  using RootInfo = GeneralContext::RootInfo;

  using IO = Game::IO;
  using State = Game::State;
  using EvalSpec = SearchSpec::EvalSpec;
  using PolicyEncoding = EvalSpec::TensorEncodings::PolicyEncoding;
  using GameResultEncoding = EvalSpec::TensorEncodings::GameResultEncoding;
  using InputFrame = EvalSpec::InputFrame;
  using Symmetries = EvalSpec::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;

  using ActionResponse = core::ActionResponse<Game>;
  using ActionPrinter = core::ActionPrinter<Game>;
  using ActionSymmetryTable = core::ActionSymmetryTable<EvalSpec>;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using PolicyTensor = PolicyEncoding::Tensor;
  using ValueArray = Game::Types::ValueArray;
  using GameResultTensor = GameResultEncoding::Tensor;
  using player_bitset_t = Game::Types::player_bitset_t;

  using TrainingDataWriter = search::TrainingDataWriter<SearchSpec>;
  using GameWriteLog = TrainingDataWriter::GameWriteLog;
  using GameWriteLog_sptr = TrainingDataWriter::GameWriteLog_sptr;

  using LogitValueArray = Game::Types::LogitValueArray;
  using NodeStableData = SearchSpec::NodeStableData;

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
  static void write_to_training_info(bool use_for_training, const ActionResponse& response,
                                     const SearchResults*, core::seat_index_t seat,
                                     GameWriteLog_sptr, TrainingInfo& training_info);
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
