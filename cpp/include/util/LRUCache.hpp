#pragma once

/*
 * This is adapted from boost's lru_cache.
 *
 * The main difference is that this implementation is powered by a std::unordered_map, while boost's
 * is powered by a std::map.
 *
 * See: https://www.boost.org/doc/libs/1_67_0/boost/compute/detail/lru_cache.hpp
 */
#include <util/Asserts.hpp>

#include <boost/intrusive/list.hpp>

#include <functional>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace util {

// a cache which evicts the least recently used item when it is full
template <class Key_, class Value_, class Hasher = std::hash<Key_>>
class LRUCache {
 public:
  using Key = Key_;
  using Value = Value_;

  static_assert(std::is_trivially_default_constructible_v<Key>);
  static_assert(std::is_trivially_default_constructible_v<Value>);

  static_assert(std::is_trivially_destructible_v<Key>);
  static_assert(std::is_trivially_destructible_v<Value>);

  static_assert(std::is_move_assignable_v<Key>);
  static_assert(std::is_move_assignable_v<Value>);

 private:
  struct Node : public boost::intrusive::list_base_hook<> {
    Key key;
    Value value;

    Node() = default;
    Node(Key k, Value v) : key(std::move(k)), value(std::move(v)) {}
  };

  using List = boost::intrusive::list<Node>;
  using Map = std::unordered_map<Key, Node*, Hasher>;

 public:
  using value_creation_func_t = std::function<Value()>;
  using eviction_func_t = std::function<void(Value&)>;

  explicit LRUCache(size_t capacity);
  ~LRUCache() { clear(); }

  // Sets the eviction handler. This function is called when an item is evicted from the cache.
  // The default handler does nothing.
  void set_eviction_handler(eviction_func_t f) { eviction_handler_ = std::move(f); }

  // If key is not in the cache, adds a mapping of key -> f(). If the cache is full, evicts the
  // least recently used item.
  //
  // Updates key to the most recently used item regardless of whether it was already in the cache or
  // not.
  //
  // Returns the value that key maps to.
  Value& insert_if_missing(const Key& key, value_creation_func_t f);

  size_t size() const { return map_.size(); }
  bool empty() const { return map_.empty(); }
  bool contains(const Key& key) const { return map_.find(key) != map_.end(); }

  void clear();

 private:
  void evict();

  List list_;
  Map map_;
  std::vector<Node> pool_;        // preallocated nodes
  std::vector<Node*> free_list_;  // available node pointers
  size_t capacity_;
  eviction_func_t eviction_handler_ = [](Value&) {};
};

}  // namespace util

#include <inline/util/LRUCache.inl>
