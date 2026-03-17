#include "beta0/PuctCalculator.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
inline PuctCalculator<Traits>::PuctCalculator(const LookupTable& lookup_table,
                                              const ManagerParams& params,
                                              const search::SearchParams& search_params,
                                              const Node* node, bool is_root)
    : seat(node->stable_data().active_seat),
      P(node->stable_data().num_valid_actions),
      Q(P.rows()),
      W(P.rows()),
      E(P.rows()),
      mE(P.rows()),
      N(P.rows()),
      PUCT(P.rows()) {
  P.setZero();
  Q.setZero();
  W.setZero();
  E.setZero();
  mE.setZero();
  N.setZero();

  for (int i = 0; i < node->stable_data().num_valid_actions; ++i) {
    /*
     * NOTE: we do NOT grab mutexes here! This means that edge_stats/child_stats can contain
     * arbitrarily-partially-written data.
     */
    Edge* edge = lookup_table.get_edge(node, i);
    P(i) = edge->P_adjusted;
    E(i) = edge->E;

    Node* child = lookup_table.get_node(edge->child_index);
    if (child) {
      const auto child_stats = child->stats_safe();  // make a copy
      Q(i) = child_stats.Q(seat);
      W(i) = child_stats.W[seat];
      N(i) = child_stats.N;
    } else {
      Q(i) = edge->child_AV[seat];
      W(i) = edge->child_AU[seat];
    }
  }

  LocalPolicyArray mask(P.rows());
  mask.setConstant(1);

  constexpr float kMin = Game::GameResults::kMinValue;
  constexpr float kMax = Game::GameResults::kMaxValue;
  auto W0 = (W == 0.f).template cast<float>();
  auto Qmax = (Q == kMax).template cast<float>();
  auto Qmin = (Q == kMin).template cast<float>();
  auto PL = W0 * Qmin;  // provably-losing actions
  auto PW = W0 * Qmax;  // provably-winning actions

  mask *= (1 - PL);  // zero out provably-losing actions
  if (PW.any()) {
    mask *= PW;  // zero out non-provably-winning actions
  }
  if ((mask == 0).all()) {
    mask.setConstant(1);  // if all actions are masked out, unmask all actions
  }

  mE = mask * E;

  /*
   * AlphaZero/KataGo defines Q to be over a [-1, +1] range, but we use a [0, +1] range.
   *
   * We multiply Q by 2 to account for this difference.
   *
   * This could have been accomplished also by multiplying cPUCT by 0.5, but this way maintains
   * better consistency with the AlphaZero/KataGo approach.
   */

  PUCT = 2 * Q + params.cPUCT * P * sqrt(mE.sum() + eps) / (mE + 1);
  PUCT = mask * PUCT + (1 - mask) * -1e6;
}

}  // namespace beta0
