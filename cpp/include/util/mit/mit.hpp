#pragma once

// The Multithreading-Is-Testable (MIT) library provides drop-in replacements for several std
// classes. When the macro MIT_TEST_MODE is defined, these classes are replaced with custom
// versions that behave deterministically, making them suitable for unit testing. Otherwise, they
// simply map to the standard library equivalents.
//
// This file is the main header for the MIT library.

#ifdef MIT_TEST_MODE

#include <util/mit/condition_variable.hpp>
#include <util/mit/lock_guard.hpp>
#include <util/mit/mutex.hpp>
#include <util/mit/thread.hpp>
#include <util/mit/unique_lock.hpp>

#else  // MIT_TEST_MODE

// Use standard library equivalents when not in test mode
#include <condition_variable>
#include <mutex>
#include <thread>

namespace mit {
using condition_variable = std::condition_variable;
using mutex = std::mutex;
using thread = std::thread;

template <typename Mutex>
using lock_guard = std::lock_guard<Mutex>;

template <typename Mutex>
using unique_lock = std::unique_lock<Mutex>;
}  // namespace mit

#endif  // MIT_TEST_MODE
