#include <util/LRUCache.hpp>

#include <util/Asserts.hpp>

namespace util {

template <class Key, class Value, class Hasher>
void LRUCache<Key, Value, Hasher>::set_capacity(size_t capacity) {
  RELEASE_ASSERT(empty(), "Cannot set capacity on a non-empty cache");
  capacity_ = capacity;
  pool_.resize(capacity_);
  free_list_.resize(capacity_);
  for (size_t i = 0; i < capacity_; ++i) {
    free_list_[i] = &pool_[i];
  }
}

template <class Key, class Value, class Hasher>
Value& LRUCache<Key, Value, Hasher>::insert_if_missing(const Key& key, value_creation_func_t f) {
  auto it = map_.find(key);
  if (it == map_.end()) {
    if (size() >= capacity_) {
      evict();
    }
    RELEASE_ASSERT(!free_list_.empty());
    Node* node = free_list_.back();
    free_list_.pop_back();
    node->key = key;
    node->value = f();
    list_.push_front(*node);
    map_[key] = node;
    return node->value;
  } else {
    Node* node = it->second;
    if (&list_.front() != node) {
      list_.erase(List::s_iterator_to(*node));
      list_.push_front(*node);
    }
    return node->value;
  }
}

template <class Key, class Value, class Hasher>
void LRUCache<Key, Value, Hasher>::clear() {
  for (Node& node : list_) {
    eviction_handler_(node.value);
    free_list_.push_back(&node);
  }
  list_.clear();
  map_.clear();
}

template <class Key, class Value, class Hasher>
void LRUCache<Key, Value, Hasher>::evict() {
  RELEASE_ASSERT(!list_.empty());
  Node& node = list_.back();
  eviction_handler_(node.value);
  map_.erase(node.key);
  list_.pop_back();
  free_list_.push_back(&node);
}

}  // namespace util
