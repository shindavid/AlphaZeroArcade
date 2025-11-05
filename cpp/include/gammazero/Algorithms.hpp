#pragma once

#include "alphazero/Algorithms.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace gamma0 {

// CRTP base class
//
// This allows us to effectively have other Algorithms classes methods invoke gamma0::Algorithms
// methods.
//
// To make this work, we need to follow the discipline of always invoking Derived::func() rather
// than simply func() within gamma0::Algorithms methods.
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
  using RootInfo = Base::RootInfo;
  using SearchContext = Base::SearchContext;
  using SearchResults = Base::SearchResults;
  using State = Base::State;
  using TrainingInfo = Base::TrainingInfo;
  using TrainingInfoParams = Base::TrainingInfoParams;
  using ValueArray = Base::ValueArray;
  using player_bitset_t = Base::player_bitset_t;

  using LocalPolicyArrayDouble = Game::Types::LocalPolicyArrayDouble;
  using NodeStableData = Traits::NodeStableData;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  class Backpropagator {
   public:
    Backpropagator(LookupTable& lookup_table) : lookup_table_(lookup_table) {}

    template <typename MutexProtectedFunc>
    void run(Node* node, Edge* edge, MutexProtectedFunc&& func);

   private:
    void update_edge_snapshots(Node* parent, Edge* edge, bool load_prev_snapshots = false);
    void update_edge_snapshots_helper(const NodeStats& child_stats, Edge* edge,
                                      bool load_edge_snapshots_before_update);

    LookupTable& lookup_table_;
    ValueArray last_child_Qgamma_snapshot_;
    ValueArray last_child_W_snapshot_;
    bool snapshots_set_ = false;
  };

  static void init_edge_from_child(const GeneralContext&, Node* parent, Edge* edge);
  static void init_node_stats_from_terminal(Node* node);
  static void undo_virtual_update(Node* node, Edge* edge);

  static void load_evaluations(SearchContext& context);

  static void to_results(const GeneralContext&, SearchResults&);
  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord& full_record, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);

 protected:
  static void update_stats(NodeStats& stats, LocalPolicyArray& pi_arr,
                           const ValueArray& last_child_Qgamma_snapshot,
                           const ValueArray& last_child_W_snapshot, const Node* node,
                           const Edge* edge, LookupTable& lookup_table);

  // Updates pi_arr in-place to be the posterior policy
  static void update_policy(LocalPolicyArray& pi_arr, const Node* node, const Edge* edge,
                            LookupTable& lookup_table, int updated_edge_arr_index,
                            float old_child_Qgamma, float old_child_W,
                            const LocalPolicyArray& child_Qgamma_arr,
                            const LocalPolicyArray& child_W_arr);

  static void compute_theta_omega_sq(double Qgamma, double W, double& theta, double& omega_sq);

  static void update_QW_fields(const NodeStableData& stable_data, const LocalPolicyArray& pi_arr,
                               const LocalActionValueArray& child_Qgamma_arr,
                               const LocalActionValueArray& child_W_arr, NodeStats& stats);
};

template <search::concepts::Traits Traits>
struct Algorithms : public AlgorithmsBase<Traits, Algorithms<Traits>> {};

}  // namespace gamma0

#include "inline/gammazero/Algorithms.inl"
