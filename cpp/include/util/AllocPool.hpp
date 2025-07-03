#pragma once

#include <util/CppUtil.hpp>
#include <util/mit/mit.hpp>

#include <boost/dynamic_bitset.hpp>

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <vector>

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
template<typename T, int N=10, bool ThreadSafe=true>
class AllocPool {
 public:
  using atomic_size_t = std::conditional_t<ThreadSafe, std::atomic<uint64_t>, uint64_t>;
  using mutex_t = std::conditional_t<ThreadSafe, mit::mutex, dummy_mutex>;

  // The below static_assert fails currently because Eigen::Array incorrectly reports itself as
  // non-trivially copyable. So we leave it commented out for now.
  // static_assert(std::is_trivially_copyable_v<T>);

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
  uint64_t fetch_add_to_size(uint64_t n);  // does size_ += n and returns the old size, atomically
  void add_blocks_if_necessary(int block_index);

  static constexpr int kNumBlocks = 64 - N;
  using Block = char*;

  atomic_size_t size_ = 0;
  int num_blocks_ = 2;
  mutable mutex_t mutex_;
  Block blocks_[kNumBlocks] = {};
};

}  // namespace util

#include <inline/util/AllocPool.inl>
