#include "beta0/SearchResults.hpp"

#include "util/EigenUtil.hpp"

namespace beta0 {

template <core::concepts::Game Game>
boost::json::object SearchResults<Game>::to_json() const {
  boost::json::object results_json;

  results_json["AQ"] = eigen_util::to_json(AQ);
  results_json["AW"] = eigen_util::to_json(AW);
  results_json["pi"] = eigen_util::to_json(pi);

  results_json["Q_min"] = eigen_util::to_json(Q_min);
  results_json["Q_max"] = eigen_util::to_json(Q_max);
  results_json["W"] = eigen_util::to_json(W);

  return results_json;
}

}  // namespace beta0
