#pragma once

#include <util/mit/mutex.hpp>

#include <util/Asserts.hpp>
#include <util/LoggingUtil.hpp>
#include <util/mit/scheduler.hpp>

namespace mit {

inline mutex::mutex() {
  scheduler::instance().register_mutex(this);
}

inline mutex::~mutex() {
  scheduler::instance().unregister_mutex(this);
}

inline void mutex::lock() {
  scheduler::instance().lock_mutex(this);
}

inline void mutex::unlock() {
  scheduler::instance().unlock_mutex(this);
}

}  // namespace mit
