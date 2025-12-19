#include "x0/Algorithms.hpp"

#include "util/EigenUtil.hpp"

namespace x0 {

template <search::concepts::Traits Traits, typename Derived>
bool AlgorithmsBase<Traits, Derived>::validate_and_symmetrize_policy_target(
  const SearchResults* mcts_results, PolicyTensor& target) {
  float sum = eigen_util::sum(target);
  if (mcts_results->provably_lost || sum == 0 || mcts_results->trivial) {
    // python training code will ignore these rows for policy training.
    return false;
  } else {
    target = mcts_results->action_symmetry_table.symmetrize(target);
    target = target / eigen_util::sum(target);
    eigen_util::debug_assert_is_valid_prob_distr(target);
    return true;
  }
}

}  // namespace x0
