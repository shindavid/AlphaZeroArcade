#pragma once

#include "util/Exceptions.hpp"

namespace mit {

// The mit framework detects multithreading bugs like mutex deadlock. When such a bug is detected,
// it throws an mit::BugDetectedError exception.
struct BugDetectedError : public util::Exception {
  using util::Exception::Exception;
};

// For unit tests for the mit framework itself, we want to run intentionally-buggy code, catch the
// resultant mit::BugDetectedError, and assert that it was thrown. The difficulty is that those
// exceptions can be thrown within spawned std::thread's, and c++ generally performs an
// uncatchable std::terminate() when an exception is thrown in a non-main-thread. Attempting to
// catch such an exception from within a spawned thread and rethrowing in the main thread can be
// cumbersome.
//
// To illustrate, how exactly should you recover from a mutex-deadlock scenario? Arbitrarily
// unlocking mutexes in order to continue the spawned threads could lead to further exceptions,
// potentially at the application level, rather than at the mit framework level.
//
// mit::BugDetectGuard is the solution to this problem. While an mit::BugDetectGuard is active, any
// mit::BugDetectedError exceptions thrown in spawned threads will be caught and stored. The
// mit::scheduler then enters an "orderly shutdown" mode, where it attempts to interrupt and exit
// all threads gracefully, silently swallowing any further exceptions. Finally, in the
// mit::BugDetectGuard destructor, the original mit::BugDetectedError is rethrown, allowing the
// unit test to assert that the bug was indeed caught.
struct BugDetectGuard {
  BugDetectGuard();
  ~BugDetectGuard() noexcept(false);
};

// Exception used by mit::scheduler when performing an orderly shutdown after a bug is detected.
// See mit::BugDetectGuard for more details.
struct OrderlyShutdownException : public util::Exception {
  using util::Exception::Exception;
};

}  // namespace mit
