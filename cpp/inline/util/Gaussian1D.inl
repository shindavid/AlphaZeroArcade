#include "util/Gaussian1D.hpp"

namespace util {

inline std::string Gaussian1D::fmt_mean(float mean, float variance) {
  if (variance == kVarianceNegInf) {
    return "-inf";
  } else if (variance == kVariancePosInf) {
    return "+inf";
  } else if (variance == kVarianceUnset) {
    return "???";
  } else {
    return util::float_to_str8(mean);
  }
}

inline std::string Gaussian1D::fmt_variance(float variance) {
  return util::float_to_str8(std::max(0.f, variance));
}

inline std::string Gaussian1D::fmt_mean0(float mean, float variance) {
  if (variance == kVarianceNegInf) {
    return "-inf";
  } else if (variance == kVariancePosInf) {
    return "+inf";
  } else if (variance == kVarianceUnset) {
    return "???";
  } else {
    return util::float_to_str8(mean, false);
  }
}

inline std::string Gaussian1D::fmt_variance0(float variance) {
  return util::float_to_str8(std::max(0.f, variance), false);
}

}  // namespace util
