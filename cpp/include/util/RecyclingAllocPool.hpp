#pragma once

#include "util/AllocPool.hpp"

#include <functional>
#include <vector>

// RecyclingAllocPool uses util::AllocPool<T> in conjunction with a vector of recycling T* pointers.
// It supports alloc and free operations, freeing onto the recycling vector, and allocating from
// either the recycling vector or the AllocPool.
//
// By design, it is NOT thread-safe. It is the caller's responsibility to ensure thread-safety.

namespace util {

template <typename T, int N = 10>
class RecyclingAllocPool {
 public:
  using recycle_func_t = std::function<void(T*)>;

  ~RecyclingAllocPool() { clear(); }

  // Set a function that will be called whenever a recycled object is allocated.
  void set_recycle_func(recycle_func_t f) {
    recycle_func_ = std::move(f);
  }

  void clear() {
    recycling_.clear();
    pool_.clear();
  }

  T* alloc() {
    if (recycling_.empty()) {
      T* t = &pool_[pool_.alloc(1)];
      new (t) T();
      return t;
    } else {
      T* ptr = recycling_.back();
      recycling_.pop_back();
      recycle_func_(ptr);
      return ptr;
    }
  }

  void free(T* ptr) { recycling_.push_back(ptr); }

 private:
  AllocPool<T, N, false> pool_;  // false = not thread-safe
  std::vector<T*> recycling_;
  recycle_func_t recycle_func_ = [](T* ptr) {};
};

}  // namespace util
