#include "alpha0/SearchResults.hpp"

#include "util/EigenUtil.hpp"

namespace alpha0 {

template <core::concepts::Game Game>
boost::json::object SearchResults<Game>::to_json() const {
  boost::json::object results_json = Base::to_json();
  results_json["counts"] = eigen_util::to_json(counts);
  results_json["AQs"] = eigen_util::to_json(AQs);
  results_json["AQs_sq"] = eigen_util::to_json(AQs_sq);
  results_json["AV"] = eigen_util::to_json(AV);
  return results_json;
}

}  // namespace alpha0
