#include <mcts/PUCTStats.hpp>

#include <bitset>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline PUCTStats<GameState, Tensorizor>::PUCTStats(
    const ManagerParams& params, const SearchParams& search_params, const Node* tree)
: cp(tree->stable_data().current_player)
, P(tree->evaluation_data().local_policy_prob_distr)
, V(P.rows())
, N(P.rows())
, VN(P.rows())
, PUCT(P.rows())
{
  V.setZero();
  N.setZero();
  VN.setZero();

  std::bitset<kMaxNumLocalActions> fpu_bits;

  for (child_index_t c = 0; c < tree->stable_data().num_valid_actions(); ++c) {
    /*
     * NOTE: we do NOT grab the child stats_mutex here! This means that child_stats can contain
     * arbitrarily-partially-written data.
     */
    Node* child = tree->get_child(c);
    if (!child) {
      fpu_bits[c] = true;
      continue;
    }
    auto child_stats = child->stats();  // struct copy to simplify reasoning about race conditions

    V(c) = child_stats.value_avg(cp);
    N(c) = child_stats.count;
    VN(c) = child_stats.virtual_count;

    fpu_bits[c] = (N(c) == 0);
  }

  if (params.enable_first_play_urgency && fpu_bits.any()) {
    /*
     * Again, we do NOT grab the stats_mutex here!
     */
    const auto& stats = tree->stats();  // no struct copy, not needed here
    dtype PV = stats.value_avg(cp);

    bool disableFPU = tree->is_root() && params.dirichlet_mult > 0 && !search_params.disable_exploration;
    dtype cFPU = disableFPU ? 0.0 : params.cFPU;
    dtype v = PV - cFPU * sqrt((P * (N > 0).template cast<dtype>()).sum());
    for (int c : bitset_util::on_indices(fpu_bits)) {
      V(c) = v;
    }
  }

  /*
   * AlphaZero/KataGo defines V to be over a [-1, +1] range, but we use a [0, +1] range.
   *
   * We multiply V by 2 to account for this difference.
   *
   * This could have been accomplished also by multiplying cPUCT by 0.5, but this way maintains better
   * consistency with the AlphaZero/KataGo approach.
   */
  PUCT = 2 * V + params.cPUCT * P * sqrt(N.sum() + eps) / (N + 1);
}

}  // namespace mcts
