#include <util/LRUCache.hpp>

namespace util {

template<class Key_, class Value_>
void LRUCache<Key_, Value_>::insert(const Key& key, const Value& value)
{
  typename Map::iterator i = map_.find(key);
  if (i == map_.end()) {
    // insert item into the cache, but first check if it is full
    if (size() >= capacity_) {
      // cache is full, evict the least recently used item
      evict();
    }

    // insert the new item
    list_.push_front(key);
    map_[key] = std::make_pair(value, list_.begin());
  }
}

template<class Key_, class Value_>
boost::optional<Value_> LRUCache<Key_, Value_>::get(const Key& key)
{
  // lookup value in the cache
  typename Map::iterator i = map_.find(key);
  if (i == map_.end()) {
    // value not in cache
    return boost::none;
  }

  // return the value, but first update its place in the most
  // recently used list
  typename KeyList ::iterator j = i->second.second;
  if (j != list_.begin()) {
    // move item to the front of the most recently used list
    list_.erase(j);
    list_.push_front(key);

    // update iterator in map
    j = list_.begin();
    const Value& value = i->second.first;
    map_[key] = std::make_pair(value, j);

    // return the value
    return value;
  } else {
    // the item is already at the front of the most recently
    // used list so just return it
    return i->second.first;
  }
}

template<class Key_, class Value_>
void LRUCache<Key_, Value_>::clear()
{
  map_.clear();
  list_.clear();
}

template<class Key_, class Value_>
void LRUCache<Key_, Value_>::evict()
{
  // evict item from the end of most recently used list
  typename KeyList::iterator i = --list_.end();
  map_.erase(*i);
  list_.erase(i);
}

}  // namespace util
