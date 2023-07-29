#pragma once

#include <atomic>
#include <memory>

namespace util {

/*
 * Same as std::atomic<std::shared_ptr<T>>, but with various convenience methods to make AtomicSharedPtr usage
 * similar to a raw pointer.
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

  AtomicSharedPtr& operator=(const std::shared_ptr<T>& t) {
    this->store(t);
    return *this;
  }

  operator T*() const {
    return this->load().get();
  }

  operator bool() const {
    return this->load().get() != nullptr;
  }

  T* operator->() const {
    return this->load().get();
  }
};

}  // namespace util
