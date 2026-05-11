#include "beta0/BackupSampleData.hpp"

#include "util/EigenUtil.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
boost::json::object BackupSampleData<Spec>::to_json() const {
  boost::json::object obj;
  obj["valid"] = valid;
  if (valid) {
    obj["N"] = eigen_util::to_json(N);
    obj["Qs"] = eigen_util::to_json(Qs);
    obj["Ws"] = eigen_util::to_json(Ws);
    obj["P"] = eigen_util::to_json(P);
    obj["AVs"] = eigen_util::to_json(AVs);
    obj["AUs"] = eigen_util::to_json(AUs);
    obj["Qs_star"] = Qs_star;
    obj["Ws_star"] = Ws_star;
  }
  return obj;
}

}  // namespace beta0
