#include "alpha0/SearchResults.hpp"

#include "util/EigenUtil.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
boost::json::object SearchResults<Spec>::to_json() const {
  boost::json::object results_json;
  results_json["P"] = eigen_util::to_json(P);
  results_json["Q"] = eigen_util::to_json(Q);
  results_json["R"] = eigen_util::to_json(R);
  results_json["action_symmetry_table"] = action_symmetry_table.to_json();
  results_json["trivial"] = trivial;
  results_json["provably_lost"] = provably_lost;
  results_json["policy_target"] = eigen_util::to_json(policy_target);
  results_json["counts"] = eigen_util::to_json(counts);
  results_json["AQs"] = eigen_util::to_json(AQs);
  results_json["AQs_sq"] = eigen_util::to_json(AQs_sq);
  results_json["AV"] = eigen_util::to_json(AV);
  return results_json;
}

}  // namespace alpha0
