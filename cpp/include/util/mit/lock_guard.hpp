#pragma once

#include <mutex>

#ifndef MIT_TEST_MODE
static_assert(false, "This file is not intended to be #include'd directly.");
#endif  // MIT_TEST_MODE

namespace mit {

template <typename Mutex>
using lock_guard = std::lock_guard<Mutex>;

}  // namespace mit
