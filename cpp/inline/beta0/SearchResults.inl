#include "beta0/SearchResults.hpp"

#include "util/EigenUtil.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
boost::json::object SearchResults<EvalSpec>::to_json() const {
  boost::json::object results_json;

  results_json["AV"] = eigen_util::to_json(AV);
  results_json["AU"] = eigen_util::to_json(AU);
  results_json["AQ"] = eigen_util::to_json(AQ);
  results_json["AQ_min"] = eigen_util::to_json(AQ_min);
  results_json["AQ_max"] = eigen_util::to_json(AQ_max);
  results_json["AW"] = eigen_util::to_json(AW);
  results_json["N"] = eigen_util::to_json(N);
  results_json["R"] = eigen_util::to_json(RN);
  results_json["policy_target"] = eigen_util::to_json(policy_target);
  results_json["policy"] = eigen_util::to_json(policy);

  results_json["Q_min"] = eigen_util::to_json(Q_min);
  results_json["Q_max"] = eigen_util::to_json(Q_max);
  results_json["W"] = eigen_util::to_json(W);

  return results_json;
}

}  // namespace beta0
