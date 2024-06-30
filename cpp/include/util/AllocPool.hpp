#pragma once

#include <bit>
#include <cstdint>
#include <mutex>
#include <vector>

namespace util {

using pool_index_t = uint64_t;

/*
 * An AllocPool is a thread-safe pool of memory that supports single or block alloc() calls. Each
 * alloc() call returns a pool_index_t that can be used to access the allocated memory via
 * operator[].
 *
 * It is assumed that all reads via operator[] occur after the corresponding alloc() call has
 * returned. This assumption can be easily relaxed, but we should not need to do so in our
 * expected use cases.
 *
 * AllocPool does NOT support free() calls. Memory is freed when the pool is destroyed.
 *
 * The underlying implementation relies on an array of blocks of type T[]. The first two blocks are
 * of size 2^N, and each subsequent block is twice the size of the previous block.
 *
 * TODO: add defragmentation functionality.
 */
template<typename T, int N=10>
class AllocPool {
 public:
  static_assert(std::is_trivially_destructible_v<T>);

  AllocPool();
  AllocPool(const AllocPool&) = delete;
  AllocPool& operator=(const AllocPool&) = delete;
  ~AllocPool();

  void clear();
  pool_index_t alloc(int n);
  T& operator[](pool_index_t i);
  const T& operator[](pool_index_t i) const;
  std::vector<T> to_vector() const;  // for debugging

 private:
  void add_block();

  static constexpr int kNumBlocks = 65 - N;
  using block_t = char*;

  block_t blocks_[kNumBlocks] = {};
  uint64_t size_ = 0;
  int num_blocks_ = 2;
  std::mutex mutex_;
};

}  // namespace util

#include <inline/util/AllocPool.inl>
