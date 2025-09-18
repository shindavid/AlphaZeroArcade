#include "core/Node.hpp"

#include "util/Asserts.hpp"

namespace core {

template <typename StableData, typename Stats>
bool Node<StableData, Stats>::increment_child_expand_count(int n) {
  if (n <= 0) return false;
  child_expand_count_ += n;
  DEBUG_ASSERT(child_expand_count_ <= this->stable_data_.num_valid_actions);
  return child_expand_count_ == this->stable_data_.num_valid_actions;
}

template <typename StableData, typename Stats>
Stats Node<StableData, Stats>::stats_safe() const {
  // NOTE[dshin]: I attempted a version of this that attempted a lock-free read, resorting to a
  // the mutex only when a set dirty-bit was found on the copied stats. Contrary to my expectations,
  // this was slightly but clearly slower than the current version. I don't really understand why
  // this might be, but it's not worth investigating further at this time.
  mit::unique_lock lock(this->mutex());
  return stats_;
}

}  // namespace core
