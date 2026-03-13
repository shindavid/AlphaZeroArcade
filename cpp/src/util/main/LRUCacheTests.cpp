#include "util/GTestUtil.hpp"
#include "util/LRUCache.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

TEST(LRUCache, EmptyCache) {
  util::LRUCache<int, int> cache(4);
  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.size(), 0);
  EXPECT_FALSE(cache.contains(42));
}

TEST(LRUCache, InsertAndRetrieve) {
  util::LRUCache<int, int> cache(4);
  int& v1 = cache.insert_if_missing(1, [] { return 10; });
  EXPECT_EQ(v1, 10);
  EXPECT_EQ(cache.size(), 1);
  EXPECT_TRUE(cache.contains(1));

  // Inserting the same key returns the existing value
  int& v2 = cache.insert_if_missing(1, [] { return 99; });
  EXPECT_EQ(v2, 10);
  EXPECT_EQ(cache.size(), 1);
}

TEST(LRUCache, EvictsLeastRecentlyUsed) {
  util::LRUCache<int, int> cache(3);
  cache.insert_if_missing(1, [] { return 10; });
  cache.insert_if_missing(2, [] { return 20; });
  cache.insert_if_missing(3, [] { return 30; });
  EXPECT_EQ(cache.size(), 3);

  // Inserting a 4th item should evict key=1 (LRU)
  cache.insert_if_missing(4, [] { return 40; });
  EXPECT_EQ(cache.size(), 3);
  EXPECT_FALSE(cache.contains(1));
  EXPECT_TRUE(cache.contains(2));
  EXPECT_TRUE(cache.contains(3));
  EXPECT_TRUE(cache.contains(4));
}

TEST(LRUCache, AccessRefreshesOrder) {
  util::LRUCache<int, int> cache(3);
  cache.insert_if_missing(1, [] { return 10; });
  cache.insert_if_missing(2, [] { return 20; });
  cache.insert_if_missing(3, [] { return 30; });

  // Access key=1 to make it most recently used
  cache.insert_if_missing(1, [] { return 99; });

  // Inserting a 4th item should evict key=2 (now the LRU)
  cache.insert_if_missing(4, [] { return 40; });
  EXPECT_TRUE(cache.contains(1));
  EXPECT_FALSE(cache.contains(2));
  EXPECT_TRUE(cache.contains(3));
  EXPECT_TRUE(cache.contains(4));
}

TEST(LRUCache, EvictionHandler) {
  std::vector<int> evicted;
  util::LRUCache<int, int> cache(2);
  cache.set_eviction_handler([&](int& v) { evicted.push_back(v); });

  cache.insert_if_missing(1, [] { return 10; });
  cache.insert_if_missing(2, [] { return 20; });
  cache.insert_if_missing(3, [] { return 30; });  // evicts key=1 (value=10)

  EXPECT_EQ(evicted.size(), 1);
  EXPECT_EQ(evicted[0], 10);
}

TEST(LRUCache, Clear) {
  std::vector<int> evicted;
  util::LRUCache<int, int> cache(4);
  cache.set_eviction_handler([&](int& v) { evicted.push_back(v); });

  cache.insert_if_missing(1, [] { return 10; });
  cache.insert_if_missing(2, [] { return 20; });
  cache.clear();

  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.size(), 0);
  EXPECT_FALSE(cache.contains(1));
  EXPECT_FALSE(cache.contains(2));
  EXPECT_EQ(evicted.size(), 2);
}

TEST(LRUCache, ReinsertAfterEviction) {
  util::LRUCache<int, int> cache(2);
  cache.insert_if_missing(1, [] { return 10; });
  cache.insert_if_missing(2, [] { return 20; });
  cache.insert_if_missing(3, [] { return 30; });  // evicts key=1

  // Re-insert key=1 with a new value
  int& v = cache.insert_if_missing(1, [] { return 100; });
  EXPECT_EQ(v, 100);
  EXPECT_TRUE(cache.contains(1));
  // key=2 should have been evicted
  EXPECT_FALSE(cache.contains(2));
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
