#include <util/CppUtil.hpp>
#include <util/Random.hpp>
#include <util/ScreenUtil.hpp>

namespace util {

TtyMode* TtyMode::instance_ = nullptr;
Random* Random::instance_ = nullptr;
ScreenClearer* ScreenClearer::instance_ = nullptr;

}  // namespace util
