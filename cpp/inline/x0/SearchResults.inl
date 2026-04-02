#include "x0/SearchResults.hpp"

#include "util/EigenUtil.hpp"

namespace x0 {

template <core::concepts::EvalSpec EvalSpec>
boost::json::object SearchResults<EvalSpec>::to_json() const {
  boost::json::object results_json;
  results_json["valid_actions"] = valid_moves.to_string();
  results_json["P"] = eigen_util::to_json(P);
  results_json["Q"] = eigen_util::to_json(Q);
  results_json["R"] = eigen_util::to_json(R);
  results_json["action_symmetry_table"] = action_symmetry_table.to_json();
  results_json["action_mode"] = game_phase;  // TODO: change key to "game_phase"
  results_json["trivial"] = trivial;
  results_json["provably_lost"] = provably_lost;
  return results_json;
}

}  // namespace x0
