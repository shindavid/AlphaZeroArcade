#pragma once

#ifdef MIT_TEST_MODE

// TODO: implement MIT_TEST_MODE overrides

#include <mutex>

namespace mit {
using mutex = std::mutex;

template <typename Mutex>
using lock_guard = std::lock_guard<Mutex>;

template <typename Mutex>
using unique_lock = std::unique_lock<Mutex>;
}  // namespace mit

#else  // MIT_TEST_MODE

#include <mutex>

namespace mit {
using mutex = std::mutex;

template <typename Mutex>
using lock_guard = std::lock_guard<Mutex>;

template <typename Mutex>
using unique_lock = std::unique_lock<Mutex>;
}  // namespace mit

#endif  // MIT_TEST_MODE
