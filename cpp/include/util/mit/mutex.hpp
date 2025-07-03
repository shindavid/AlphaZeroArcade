#pragma once

#include <mutex>

#ifndef MIT_TEST_MODE
static_assert(false, "This file is not intended to be #include'd directly.");
#endif  // MIT_TEST_MODE

// TODO: implement MIT_TEST_MODE overrides

namespace mit {

using mutex = std::mutex;

template <typename Mutex>
using lock_guard = std::lock_guard<Mutex>;

template <typename Mutex>
using unique_lock = std::unique_lock<Mutex>;

}  // namespace mit
