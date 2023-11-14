#include <mcts/PUCTStats.hpp>

#include <bitset>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline PUCTStats<GameState, Tensorizor>::PUCTStats(const ManagerParams& params,
                                                   const SearchParams& search_params,
                                                   TreeTraversalMode traversal_mode,
                                                   const Node* tree, bool is_root)
    : cp(tree->stable_data().current_player),
      P(tree->evaluation_data().local_policy_prob_distr),
      V(P.rows()),
      PW(P.rows()),
      PL(P.rows()),
      E(P.rows()),
      N(P.rows()),
      VN(P.rows()),
      PUCT(P.rows()) {
  V.setZero();
  PW.setZero();
  PL.setZero();
  E.setZero();
  N.setZero();
  VN.setZero();

  std::bitset<kMaxNumLocalActions> fpu_bits;
  fpu_bits.set();

  for (const auto& edge : tree->children_data()) {
    /*
     * NOTE: we do NOT grab mutexes here! This means that edge_stats/child_stats can contain
     * arbitrarily-partially-written data.
     */
    core::action_index_t i = edge.action_index();
    const auto& child_stats = edge.child()->stats(traversal_mode);

    V(i) = child_stats.virtualized_avg(cp);
    PW(i) = child_stats.provably_winning[cp];
    PL(i) = child_stats.provably_losing[cp];
    E(i) = edge.count(traversal_mode);
    N(i) = child_stats.real_count;
    VN(i) = child_stats.virtual_count;

    fpu_bits[i] = (N(i) == 0);
  }

  if (params.enable_first_play_urgency && fpu_bits.any()) {
    /*
     * Again, we do NOT grab the stats_mutex here!
     */
    const auto& stats = tree->stats(traversal_mode);
    dtype PV = stats.virtualized_avg(cp);

    bool disableFPU = is_root && params.dirichlet_mult > 0 && !search_params.disable_exploration;
    dtype cFPU = disableFPU ? 0.0 : params.cFPU;
    dtype v = PV - cFPU * sqrt((P * (N > 0).template cast<dtype>()).sum());
    for (int i : bitset_util::on_indices(fpu_bits)) {
      V(i) = v;
    }
  }

  /*
   * AlphaZero/KataGo defines V to be over a [-1, +1] range, but we use a [0, +1] range.
   *
   * We multiply V by 2 to account for this difference.
   *
   * This could have been accomplished also by multiplying cPUCT by 0.5, but this way maintains
   * better consistency with the AlphaZero/KataGo approach.
   */
  PUCT = 2 * V + params.cPUCT * P * sqrt(N.sum() + eps) / (N + 1);

  if (params.avoid_proven_losers && !PL.all()) {
    PUCT *= (1 - PL);  // zero out provably-losing actions
  }
  if (params.exploit_proven_winners && PW.any()) {
    PUCT *= PW;
  }
}

}  // namespace mcts
