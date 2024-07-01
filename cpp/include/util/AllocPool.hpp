#pragma once

#include <bit>
#include <cstdint>
#include <mutex>
#include <vector>

#include <boost/dynamic_bitset.hpp>

namespace util {

using pool_index_t = int64_t;

/*
 * An AllocPool is a thread-safe pool of memory that supports single or block alloc() calls. Each
 * alloc() call returns a pool_index_t that can be used to access the allocated memory via
 * operator[].
 *
 * The alloc() call is mutex-protected, while operator[] is usually lockfree.
 *
 * AllocPool does NOT support free() calls. Memory is freed when the pool is destroyed.
 *
 * The underlying implementation relies on an array of blocks of type T[]. The first two blocks are
 * of size 2^N, and each subsequent block is twice the size of the previous block.
 */
template<typename T, int N=10>
class AllocPool {
 public:
  static_assert(std::is_trivially_destructible_v<T>);
  static_assert(std::is_trivially_copyable_v<T>);

  AllocPool();
  AllocPool(const AllocPool&) = delete;
  AllocPool& operator=(const AllocPool&) = delete;
  ~AllocPool();

  void clear();
  pool_index_t alloc(int n);
  T& operator[](pool_index_t i);
  const T& operator[](pool_index_t i) const;
  uint64_t size() const { return size_; }
  std::vector<T> to_vector() const;  // for debugging
  void defragment(const boost::dynamic_bitset<>& used_indices);

 private:
  void add_block();

  static constexpr int kNumBlocks = 64 - N;
  using block_t = char*;

  uint64_t size_ = 0;
  int num_blocks_ = 2;
  mutable std::mutex mutex_;
  block_t blocks_[kNumBlocks] = {};
};

}  // namespace util

#include <inline/util/AllocPool.inl>
