#pragma once

#include "util/mit/functions.hpp"

#include "util/mit/scheduler.hpp"

namespace mit {

inline void seed(int s) { scheduler::instance().seed(s); }

inline void reset() {
  auto& sched = scheduler::instance();
  sched.reset();
}

}  // namespace mit
