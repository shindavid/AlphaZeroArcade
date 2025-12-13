#pragma once

#include "alphazero/Algorithms.hpp"
#include "core/BasicTypes.hpp"
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
  using Node = Base::Node;
  using NodeStats = Base::NodeStats;
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

  using LocalPolicyArrayDouble = Game::Types::LocalPolicyArrayDouble;
  using LogitValueArray = Game::Types::LogitValueArray;
  using NodeStableData = Traits::NodeStableData;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  class Backpropagator {
   public:
    Backpropagator(SearchContext& context) : context_(context) {}

    template <typename MutexProtectedFunc>
    void run(Node* node, Edge* edge, MutexProtectedFunc&& func);

   private:
    SearchContext& context_;
  };

  static void init_node_stats_from_terminal(Node* node);
  static void undo_virtual_update(Node* node, Edge* edge);

  static int get_best_child_index(const SearchContext& context);
  static void load_evaluations(SearchContext& context);

  static void to_results(const GeneralContext&, SearchResults&);
  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord& full_record, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);

 protected:
  static void update_stats(SearchContext& context, NodeStats& stats, LocalPolicyArray& pi_arr,
                           const Node* node, const Edge* edge);

  // Updates pi_arr in-place to be the posterior policy
  static void update_policy(SearchContext& context, LocalPolicyArray& pi_arr, const Node* node,
                            const Edge* edge, LookupTable& lookup_table, int updated_edge_arr_index,
                            const LocalPolicyArray& prior_pi_arr,
                            const util::Gaussian1D* prior_logit_beliefs,
                            const util::Gaussian1D* cur_logit_beliefs);

  static void update_QW(NodeStats& stats, core::seat_index_t seat, const LocalPolicyArray& pi_arr,
                       const LocalActionValueArray& child_Q_arr,
                       const LocalActionValueArray& child_W_arr);

  static void populate_logit_value_beliefs(const ValueArray& Q, const ValueArray& W,
                                           LogitValueArray& logit_value_beliefs);
  static util::Gaussian1D compute_logit_value_belief(float Q, float W);
  static void normalize_policy(LocalPolicyArray& pi_arr);
};

template <search::concepts::Traits Traits>
struct Algorithms : public AlgorithmsBase<Traits, Algorithms<Traits>> {};

}  // namespace beta0

#include "inline/betazero/Algorithms.inl"
