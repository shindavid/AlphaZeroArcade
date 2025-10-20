#pragma once

#include "alphazero/Algorithms.hpp"
#include "search/concepts/TraitsConcept.hpp"

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
  using GeneralContext = Base::GeneralContext;
  using LocalActionValueArray = Base::LocalActionValueArray;
  using LookupTable = Base::LookupTable;
  using Node = Base::Node;
  using NodeStats = Base::NodeStats;
  using PolicyTensorData = Base::PolicyTensorData;
  using RootInfo = Base::RootInfo;
  using SearchContext = Base::SearchContext;
  using SearchResults = Base::SearchResults;
  using TrainingInfo = Base::TrainingInfo;
  using TrainingInfoParams = Base::TrainingInfoParams;
  using ValueArray = Base::ValueArray;
  using player_bitset_t = Base::player_bitset_t;

  using EigenMapArrayXf = Eigen::Map<Eigen::ArrayXf>;
  using EigenMapArrayXd = Eigen::Map<Eigen::ArrayXd>;

  template <typename MutexProtectedFunc>
  static void backprop_helper(Node* node, Edge* edge, LookupTable& lookup_table,
                              MutexProtectedFunc&&);

  static void load_evaluations(SearchContext& context);

  static void to_results(const GeneralContext&, SearchResults&);
  static void write_to_training_info(const TrainingInfoParams&, TrainingInfo& training_info);
  static void to_record(const TrainingInfo&, GameLogFullRecord&);
  static void serialize_record(const GameLogFullRecord& full_record, std::vector<char>& buf);
  static void to_view(const GameLogViewParams&, GameLogView&);

 protected:
  static void update_stats(Node* node, Edge* edge, LookupTable& lookup_table,
                           const NodeStats& old_stats);

  // Updates pi_arr in-place to be the posterior policy
  static void update_policy(Node* node, Edge* edge, LookupTable& lookup_table,
                            const NodeStats& old_stats, const NodeStats* child_stats_arr,
                            EigenMapArrayXf pi_arr, int updated_edge_arr_index);

  static void compute_theta_omega_sq(const NodeStats& stats, core::seat_index_t seat, double& theta,
                                     double& omega_sq);
};

template <search::concepts::Traits Traits>
struct Algorithms : public AlgorithmsBase<Traits, Algorithms<Traits>> {};

}  // namespace beta0

#include "inline/betazero/Algorithms.inl"
