#include <util/AllocPool.hpp>

#include <util/Asserts.hpp>

namespace util {

namespace detail {

template<int N>
int get_block_index(pool_index_t i) {
  return std::max(0, 64 - N - std::countl_zero(uint64_t(i)));
}

}  // namespace detail

template <typename T, int N>
AllocPool<T, N>::AllocPool() {
  blocks_[0] = new char[sizeof(T) * (1 << N)];
  blocks_[1] = new char[sizeof(T) * (1 << N)];
}

template <typename T, int N>
AllocPool<T, N>::~AllocPool() {
  for (int i = 0; i < kNumBlocks; ++i) {
    delete[] blocks_[i];
  }
}

template <typename T, int N>
void AllocPool<T, N>::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  size_ = 0;
  num_blocks_ = 2;
}

template <typename T, int N>
pool_index_t AllocPool<T, N>::alloc(int n) {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t old_size = size_;
  size_ += n;

  int block_index = detail::get_block_index<N>(size_ - 1);
  for (int i = num_blocks_; i <= block_index; ++i) {
    add_block();
  }
  return old_size;
}

template <typename T, int N>
T& AllocPool<T, N>::operator[](pool_index_t i) {
  debug_assert(i >= 0, "Index out of bounds: %ld", i);
  int block_index = detail::get_block_index<N>(i);
  int offset = block_index == 0 ? i : (i - std::bit_floor(uint64_t(i)));

  T* block = reinterpret_cast<T*>(blocks_[block_index]);
  if (!block) {
    // Race-condition
    std::lock_guard<std::mutex> lock(mutex_);
    block = reinterpret_cast<T*>(blocks_[block_index]);
  }
  return block[offset];
}

template <typename T, int N>
const T& AllocPool<T, N>::operator[](pool_index_t i) const {
  debug_assert(i >= 0 && i < size_, "Index out of bounds: %ld", i);
  int block_index = detail::get_block_index<N>(i);
  int offset = block_index == 0 ? i : (i - std::bit_floor(uint64_t(i)));

  const T* block = reinterpret_cast<const T*>(blocks_[block_index]);
  if (!block) {
    // Race-condition
    std::lock_guard<std::mutex> lock(mutex_);
    block = reinterpret_cast<const T*>(blocks_[block_index]);
  }
  return block[offset];
}

template <typename T, int N>
std::vector<T> AllocPool<T, N>::to_vector() const {
  std::vector<T> vec(size_);
  for (uint64_t i = 0; i < size_; ++i) {
    vec[i] = (*this)[i];
  }
  return vec;
}

template <typename T, int N>
void AllocPool<T, N>::defragment(const boost::dynamic_bitset<>& used_indices) {
  util::release_assert(used_indices.size() == size_);

  uint64_t r = 0;
  uint64_t w = 0;
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

template <typename T, int N>
void AllocPool<T, N>::add_block() {
  if (!blocks_[num_blocks_]) {
    blocks_[num_blocks_] = new char[sizeof(T) * (1 << (N + num_blocks_ - 1))];
  }
  ++num_blocks_;
}

}  // namespace util
