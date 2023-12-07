#pragma once

/*
 * This is adapted from boost's lru_cache.
 *
 * The main difference is that this implementation is powered by a std::unordered_map, while boost's
 * is powered by a std::map.
 *
 * See: https://www.boost.org/doc/libs/1_67_0/boost/compute/detail/lru_cache.hpp
 */

#include <list>
#include <unordered_map>
#include <utility>

#include <boost/optional.hpp>

namespace util {

// a cache which evicts the least recently used item when it is full
template <class Key_, class Value_>
class LRUCache {
 public:
  using Key = Key_;
  using Value = Value_;
  using KeyList = std::list<Key>;
  using MapValue = std::pair<Value, typename KeyList::iterator>;
  using Map = std::unordered_map<Key, MapValue>;

  LRUCache(size_t capacity) : map_(capacity), capacity_(capacity) {}

  size_t size() const { return map_.size(); }
  size_t capacity() const { return capacity_; }
  bool empty() const { return map_.empty(); }
  bool contains(const Key& key) { return map_.find(key) != map_.end(); }
  void insert(const Key& key, const Value& value);
  boost::optional<Value> get(const Key& key);
  void clear();
  float get_hash_balance_factor()
      const;  // (1 + size(largest_bucket)) / (1 + size(smallest_bucket))

 private:
  void evict();

  Map map_;
  KeyList list_;
  size_t capacity_;
};

}  // namespace util

#include <inline/util/LRUCache.inl>
