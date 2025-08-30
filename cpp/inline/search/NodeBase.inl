#include "search/NodeBase.hpp"

namespace search {

template <typename Traits>
template <typename... Ts>
NodeBase<Traits>::NodeBase(mit::mutex* mutex, Ts&&... args)
    : NodeBaseCore(std::forward<Ts>(args)...),
      mutex_(mutex) {}


template <typename Traits>
bool NodeBase<Traits>::increment_child_expand_count(int n) {
  if (n <= 0) return false;
  child_expand_count_ += n;
  DEBUG_ASSERT(child_expand_count_ <= this->stable_data_.num_valid_actions);
  return child_expand_count_ == this->stable_data_.num_valid_actions;
}

}  // namespace search
