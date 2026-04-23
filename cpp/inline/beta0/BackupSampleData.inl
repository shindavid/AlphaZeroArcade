#include "beta0/BackupSampleData.hpp"

#include "util/EigenUtil.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
boost::json::object BackupSampleData<Spec>::to_json() const {
  boost::json::object obj;
  obj["valid"] = valid;
  if (valid) {
    obj["N"] = eigen_util::to_json(N);
    obj["Q"] = eigen_util::to_json(Q);
    obj["W"] = eigen_util::to_json(W);
  }
  return obj;
}

}  // namespace beta0
