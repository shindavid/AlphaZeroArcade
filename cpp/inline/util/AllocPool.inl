#include "util/AllocPool.hpp"
#include "util/Asserts.hpp"
#include "util/mit/mit.hpp"

#include <type_traits>

namespace util {

namespace detail {

template <int N>
int get_block_index(pool_index_t i) {
  return std::max(0, 64 - N - std::countl_zero(uint64_t(i)));
}

}  // namespace detail

template <typename T, int N, bool ThreadSafe>
AllocPool<T, N, ThreadSafe>::AllocPool() {
  blocks_[0] = new char[sizeof(T) * (1 << N)];
  blocks_[1] = new char[sizeof(T) * (1 << N)];
}

template <typename T, int N, bool ThreadSafe>
AllocPool<T, N, ThreadSafe>::~AllocPool() {
  // TODO: call destructor on all elements?
  for (int i = 0; i < kNumBlocks; ++i) {
    delete[] blocks_[i];
  }
}

template <typename T, int N, bool ThreadSafe>
void AllocPool<T, N, ThreadSafe>::clear() {
  mit::unique_lock lock(mutex_);
  size_ = 0;
}

template <typename T, int N, bool ThreadSafe>
pool_index_t AllocPool<T, N, ThreadSafe>::alloc(int n) {
  uint64_t old_size = fetch_add_to_size(n);
  uint64_t new_size = old_size + n;

  int block_index = detail::get_block_index<N>(new_size - 1);
  add_blocks_if_necessary(block_index);
  return old_size;
}

template <typename T, int N, bool ThreadSafe>
T& AllocPool<T, N, ThreadSafe>::operator[](pool_index_t i) {
  DEBUG_ASSERT(i >= 0, "Index out of bounds: {}", i);
  int block_index = detail::get_block_index<N>(i);
  int offset = block_index == 0 ? i : (i - std::bit_floor(uint64_t(i)));

  T* block = reinterpret_cast<T*>(blocks_[block_index]);
  return block[offset];
}

template <typename T, int N, bool ThreadSafe>
const T& AllocPool<T, N, ThreadSafe>::operator[](pool_index_t i) const {
  DEBUG_ASSERT(i >= 0 && size_t(i) < size_, "Index out of bounds: {}", i);
  int block_index = detail::get_block_index<N>(i);
  int offset = block_index == 0 ? i : (i - std::bit_floor(uint64_t(i)));

  const T* block = reinterpret_cast<const T*>(blocks_[block_index]);
  return block[offset];
}

template <typename T, int N, bool ThreadSafe>
std::vector<T> AllocPool<T, N, ThreadSafe>::to_vector() const {
  std::vector<T> vec(size_);
  for (uint64_t i = 0; i < size_; ++i) {
    vec[i] = (*this)[i];
  }
  return vec;
}

template <typename T, int N, bool ThreadSafe>
void AllocPool<T, N, ThreadSafe>::defragment(const boost::dynamic_bitset<>& used_indices) {
  // The below static_assert() incorrectly fails for some fixed-size Eigen types,
  // So I comment it out for now. When c++ reflection comes out, I might be able to resurrect it.
  // static_assert(std::is_trivially_constructible_v<T>);

  static_assert(std::is_trivially_destructible_v<T>);
  RELEASE_ASSERT(used_indices.size() == size_);

  uint64_t r = 0;
  uint64_t w = 0;

  // TODO: find contiguous blocks, and copy block-wise
  while (r < size_) {
    if (used_indices[r]) {
      if (r != w) {
        (*this)[w] = (*this)[r];
      }
      ++w;
    }
    ++r;
  }
  size_ = w;
}

template <typename T, int N, bool ThreadSafe>
uint64_t AllocPool<T, N, ThreadSafe>::fetch_add_to_size(uint64_t n) {
  if constexpr (ThreadSafe) {
    return size_.fetch_add(n, std::memory_order_relaxed);
  } else {
    uint64_t old_size = size_;
    size_ += n;
    return old_size;
  }
}

template <typename T, int N, bool ThreadSafe>
void AllocPool<T, N, ThreadSafe>::add_blocks_if_necessary(int block_index) {
  if (block_index >= num_blocks_) {
    mit::unique_lock lock(mutex_);
    while (num_blocks_ <= block_index) {
      blocks_[num_blocks_] = new char[sizeof(T) * (1 << (N + num_blocks_ - 1))];
      ++num_blocks_;
    }
  }
}

}  // namespace util
