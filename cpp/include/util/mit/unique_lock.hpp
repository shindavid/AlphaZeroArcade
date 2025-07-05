#pragma once

#include <mutex>

#ifndef MIT_TEST_MODE
static_assert(false, "This file is not intended to be #include'd directly.");
#endif  // MIT_TEST_MODE

namespace mit {

template <typename Mutex>
using unique_lock = std::unique_lock<Mutex>;

}  // namespace mit
