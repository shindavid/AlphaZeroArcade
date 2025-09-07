#include "core/NodeBase.hpp"

#include "util/Asserts.hpp"

namespace core {

template <core::concepts::EvalSpec EvalSpec>
bool NodeBase<EvalSpec>::increment_child_expand_count(int n) {
  if (n <= 0) return false;
  child_expand_count_ += n;
  DEBUG_ASSERT(child_expand_count_ <= this->stable_data_.num_valid_actions);
  return child_expand_count_ == this->stable_data_.num_valid_actions;
}

}  // namespace core
