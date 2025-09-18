#include "betazero/SearchResults.hpp"

#include "util/EigenUtil.hpp"

namespace beta0 {

template <core::concepts::Game Game>
boost::json::object SearchResults<Game>::to_json() const {
  boost::json::object results_json = Base::to_json();
  results_json["action_value_uncertainties"] = eigen_util::to_json(action_value_uncertainties);
  return results_json;
}

}  // namespace beta0
