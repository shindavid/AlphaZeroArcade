#include <util/AllocPool.hpp>

#include <util/Asserts.hpp>

namespace util {

namespace detail {

template<int N>
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
  std::unique_lock lock(mutex_);
  size_ = 0;
}

template <typename T, int N, bool ThreadSafe>
pool_index_t AllocPool<T, N, ThreadSafe>::alloc(int n) {
  // TODO: use std::atomic rather than a mutex here. If an overflow occurs, we should grab the mutex
  // at that point before calling add_block(), with the awareness that another thread might jump in
  // and perform the add_block() before we do.
  std::unique_lock lock(mutex_);
  uint64_t old_size = size_;
  size_ += n;

  int block_index = detail::get_block_index<N>(size_ - 1);
  for (int i = num_blocks_; i <= block_index; ++i) {
    add_block();
  }
  return old_size;
}

template <typename T, int N, bool ThreadSafe>
T& AllocPool<T, N, ThreadSafe>::operator[](pool_index_t i) {
  debug_assert(i >= 0, "Index out of bounds: %ld", i);
  int block_index = detail::get_block_index<N>(i);
  int offset = block_index == 0 ? i : (i - std::bit_floor(uint64_t(i)));

  T* block = reinterpret_cast<T*>(blocks_[block_index]);
  return block[offset];
}

template <typename T, int N, bool ThreadSafe>
const T& AllocPool<T, N, ThreadSafe>::operator[](pool_index_t i) const {
  debug_assert(i >= 0 && size_t(i) < size_, "Index out of bounds: %ld", i);
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
  util::release_assert(used_indices.size() == size_);

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
void AllocPool<T, N, ThreadSafe>::add_block() {
  if (!blocks_[num_blocks_]) {
    blocks_[num_blocks_] = new char[sizeof(T) * (1 << (N + num_blocks_ - 1))];
  }
  ++num_blocks_;
}

}  // namespace util
