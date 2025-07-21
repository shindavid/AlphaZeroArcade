#pragma once

#include "util/mit/exceptions.hpp"
#include "util/mit/scheduler.hpp"

namespace mit {

inline BugDetectGuard::BugDetectGuard() { scheduler::instance().enable_bug_catching_mode(); }

inline BugDetectGuard::~BugDetectGuard() noexcept(false) {
  scheduler::instance().disable_bug_catching_mode();
}

}  // namespace mit
