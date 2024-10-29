#include <mcts/ActionSelector.hpp>

#include <bitset>

namespace mcts {

template <core::concepts::Game Game>
inline ActionSelector<Game>::ActionSelector(const ManagerParams& params,
                                            const SearchParams& search_params, const Node* node,
                                            bool is_root)
    : cp(node->stable_data().current_player),
      P(node->stable_data().num_valid_actions),
      Q(P.rows()),
      QLB(P.rows()),
      QUB(P.rows()),
      PW(P.rows()),
      PL(P.rows()),
      E(P.rows()),
      mE(P.rows()),
      RN(P.rows()),
      VN(P.rows()),
      FPU(P.rows()),
      PUCT(P.rows()) {
  P.setZero();
  Q.setZero();
  QLB.setConstant(Game::GameResults::kMinValue);
  QUB.setConstant(Game::GameResults::kMaxValue);
  PW.setZero();
  PL.setZero();
  E.setZero();
  mE.setZero();
  RN.setZero();
  VN.setZero();
  FPU.setZero();

  for (int i = 0; i < node->stable_data().num_valid_actions; ++i) {
    /*
     * NOTE: we do NOT grab mutexes here! This means that edge_stats/child_stats can contain
     * arbitrarily-partially-written data.
     */
    using edge_t = Node::edge_t;
    edge_t* edge = node->get_edge(i);
    P(i) = edge->adjusted_policy_prior;
    E(i) = edge->E;
    mE(i) = edge->mE;

    Node* child = node->get_child(edge);
    if (child) {
      const auto& child_stats = child->stats();
      Q(i) = child_stats.Q(cp);
      QLB(i) = child_stats.Q_lower_bound(cp);
      QUB(i) = child_stats.Q_upper_bound(cp);
      PW(i) = QLB(i) == Game::GameResults::kMaxValue;
      PL(i) = QUB(i) == Game::GameResults::kMinValue;
      RN(i) = child_stats.RN;
      VN(i) = child_stats.VN;

      if (VN(i)) {
        Q(i) = (RN(i) * Q(i) + VN(i) * Game::GameResults::kMinValue) / (RN(i) + VN(i));
      }
    } else {
      Q(i) = edge->child_V_estimate;
    }
  }

  bool fpu_any = false;
  if (params.enable_first_play_urgency) {
    FPU = (E == 0).template cast<float>();
    fpu_any = FPU.any();
  }

  if (fpu_any) {
    /*
     * Again, we do NOT grab the stats_mutex here!
     */
    const auto& stats = node->stats();  // no struct copy, not needed here
    float PV = stats.Q(cp);

    bool disableFPU = is_root && params.dirichlet_mult > 0 && search_params.full_search;
    float cFPU = disableFPU ? 0.0 : params.cFPU;
    float v = PV - cFPU * sqrt((P * (E > 0).template cast<float>()).sum());
    Q = (1 - FPU) * Q + FPU * v;
  }

  auto mask = (mE > 0 || E == 0).template cast<float>();

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

}  // namespace mcts
