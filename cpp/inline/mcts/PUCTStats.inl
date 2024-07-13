#include <mcts/PUCTStats.hpp>

#include <bitset>

namespace mcts {

template <core::concepts::Game Game>
inline PUCTStats<Game>::PUCTStats(const ManagerParams& params, const SearchParams& search_params,
                                  const Node* node, bool is_root)
    : cp(node->stable_data().current_player),
      P(node->num_representative_actions()),
      V(P.rows()),
      PW(P.rows()),
      PL(P.rows()),
      E(P.rows()),
      N(P.rows()),
      VN(P.rows()),
      FPU(P.rows()),
      PUCT(P.rows()) {
  P.setZero();
  V.setZero();
  PW.setZero();
  PL.setZero();
  E.setZero();
  N.setZero();
  VN.setZero();
  FPU.setZero();

  bool fpu_any = false;
  for (int e = 0; e < node->stable_data().num_valid_actions; ++e) {
    /*
     * NOTE: we do NOT grab mutexes here! This means that edge_stats/child_stats can contain
     * arbitrarily-partially-written data.
     */
    using edge_t = Node::edge_t;
    edge_t* edge = node->get_edge(e);
    if (edge->representative_edge_index != e) continue;

    int i = edge->collapsed_index;
    P(i) = edge->adjusted_policy_prior;
    E(i) = edge->RN;

    Node* child = node->get_child(edge);
    if (child) {
      const auto& child_stats = child->stats();
      V(i) = child_stats.VQ(cp);
      PW(i) = child_stats.provably_winning[cp];
      PL(i) = child_stats.provably_losing[cp];
      N(i) = child_stats.RN;
      VN(i) = child_stats.VN;
    }

    edge_indices(i) = e;

    bool fpu = N(i) == 0;
    FPU[i] = fpu;
    fpu_any |= fpu;
  }

  if (params.enable_first_play_urgency && fpu_any) {
    /*
     * Again, we do NOT grab the stats_mutex here!
     */
    const auto& stats = node->stats();  // no struct copy, not needed here
    float PV = stats.VQ(cp);

    bool disableFPU = is_root && params.dirichlet_mult > 0 && !search_params.disable_exploration;
    float cFPU = disableFPU ? 0.0 : params.cFPU;
    float v = PV - cFPU * sqrt((P * (N > 0).template cast<float>()).sum());
    V = (1 - FPU) * V + FPU * v;
  }

  /*
   * AlphaZero/KataGo defines V to be over a [-1, +1] range, but we use a [0, +1] range.
   *
   * We multiply V by 2 to account for this difference.
   *
   * This could have been accomplished also by multiplying cPUCT by 0.5, but this way maintains
   * better consistency with the AlphaZero/KataGo approach.
   */

  PUCT = 2 * V + params.cPUCT * P * sqrt(E.sum() + eps) / (E + 1);

  if (params.avoid_proven_losers && !PL.all()) {
    PUCT *= (1 - PL);  // zero out provably-losing actions
  }
  if (params.exploit_proven_winners && PW.any()) {
    PUCT *= PW;
  }
}

}  // namespace mcts
