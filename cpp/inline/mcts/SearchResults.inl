#include "mcts/SearchResults.hpp"

#include "util/EigenUtil.hpp"

namespace mcts {

template <typename Traits>
boost::json::object SearchResults<Traits>::to_json() const {
  boost::json::object results_json;
  results_json["valid_actions"] = valid_actions.to_string();
  results_json["counts"] = eigen_util::to_json(counts);
  results_json["policy_target"] = eigen_util::to_json(policy_target);
  results_json["policy_prior"] = eigen_util::to_json(policy_prior);
  results_json["Q"] = eigen_util::to_json(Q);
  results_json["Q_sq"] = eigen_util::to_json(Q_sq);
  results_json["action_values"] = eigen_util::to_json(action_values);
  results_json["win_rates"] = eigen_util::to_json(win_rates);
  results_json["value_prior"] = eigen_util::to_json(value_prior);
  results_json["action_symmetry_table"] = action_symmetry_table.to_json();
  results_json["trivial"] = trivial;
  results_json["provably_lost"] = provably_lost;
  return results_json;
}

}  // namespace mcts
