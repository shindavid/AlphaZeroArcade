#include <util/PrintUtil.hpp>
#include <util/Random.hpp>

namespace util {
namespace detail {

_xprintf_helper _xprintf_helper::instance_;

}  // namespace detail

Random* Random::instance_ = nullptr;

}  // namespace util
