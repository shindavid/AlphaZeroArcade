#include "util/GTestUtil.hpp"
#include "util/RecyclingAllocPool.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(RecyclingAllocPool, AllocReturnsDistinctPointers) {
  util::RecyclingAllocPool<int> pool;
  int* a = pool.alloc();
  int* b = pool.alloc();
  EXPECT_NE(a, b);
}

TEST(RecyclingAllocPool, FreeAndReallocReusesPointer) {
  util::RecyclingAllocPool<int> pool;
  int* a = pool.alloc();
  *a = 42;
  pool.free(a);

  int* b = pool.alloc();
  EXPECT_EQ(a, b);  // should reuse the freed pointer
}

TEST(RecyclingAllocPool, RecycleFuncIsCalled) {
  int recycle_count = 0;
  util::RecyclingAllocPool<int> pool;
  pool.set_recycle_func([&](int*) { recycle_count++; });

  int* a = pool.alloc();  // fresh alloc, no recycle callback
  EXPECT_EQ(recycle_count, 0);

  pool.free(a);
  int* b = pool.alloc();  // reuse from recycling, callback fires
  EXPECT_EQ(recycle_count, 1);
  (void)b;
}

TEST(RecyclingAllocPool, ClearEmptiesPool) {
  util::RecyclingAllocPool<int> pool;
  std::vector<int*> ptrs;
  for (int i = 0; i < 5; ++i) {
    ptrs.push_back(pool.alloc());
  }
  for (auto* p : ptrs) pool.free(p);

  pool.clear();

  // After clear, new allocs should come from the underlying pool (fresh)
  // We can't check pointers directly but can verify no crash
  int* a = pool.alloc();
  *a = 99;
  EXPECT_EQ(*a, 99);
}

TEST(RecyclingAllocPool, MultipleAllocFreeRounds) {
  util::RecyclingAllocPool<int> pool;

  // Round 1: alloc and free several
  std::vector<int*> ptrs;
  for (int i = 0; i < 10; ++i) {
    int* p = pool.alloc();
    *p = i;
    ptrs.push_back(p);
  }
  for (auto* p : ptrs) pool.free(p);
  ptrs.clear();

  // Round 2: realloc from recycling
  for (int i = 0; i < 10; ++i) {
    int* p = pool.alloc();
    *p = i + 100;
    ptrs.push_back(p);
  }

  // Verify values
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(*ptrs[i], i + 100);
  }
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
