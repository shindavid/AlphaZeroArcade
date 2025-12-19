#include "beta0/SearchResults.hpp"

#include "util/EigenUtil.hpp"

namespace beta0 {

template <core::concepts::Game Game>
boost::json::object SearchResults<Game>::to_json() const {
  boost::json::object results_json;
  results_json["valid_actions"] = valid_actions.to_string_natural();

  results_json["P"] = eigen_util::to_json(P);
  results_json["pi"] = eigen_util::to_json(pi);
  results_json["AQ"] = eigen_util::to_json(AQ);
  results_json["AW"] = eigen_util::to_json(AW);

  results_json["R"] = eigen_util::to_json(R);
  results_json["Q"] = eigen_util::to_json(Q);
  results_json["Q_min"] = eigen_util::to_json(Q_min);
  results_json["Q_max"] = eigen_util::to_json(Q_max);
  results_json["W"] = eigen_util::to_json(W);

  results_json["action_symmetry_table"] = action_symmetry_table.to_json();
  results_json["trivial"] = trivial;
  results_json["provably_lost"] = provably_lost;

  return results_json;
}

}  // namespace beta0
