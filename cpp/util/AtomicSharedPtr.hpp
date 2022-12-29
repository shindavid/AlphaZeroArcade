#pragma once

#include <atomic>
#include <memory>

namespace util {

/*
 * Same as std::atomic<std::shared_ptr<T>>, but with copy constructor and assignment operator.
 */
template<typename T>
class AtomicSharedPtr : public std::atomic<std::shared_ptr<T>> {
public:
  AtomicSharedPtr() = default;

  AtomicSharedPtr(const AtomicSharedPtr& ptr) {
    this->store(ptr.load());
  }

  AtomicSharedPtr& operator=(const AtomicSharedPtr& ptr) {
    this->store(ptr.load());
    return *this;
  }
};

}  // namespace util
