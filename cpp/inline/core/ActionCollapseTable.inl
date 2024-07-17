#include <core/ActionCollapseTable.hpp>

#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>

namespace core {

template <concepts::GameConstants GameConstants, group::concepts::FiniteGroup Group>
void ActionCollapseTable<GameConstants, Group>::load(const lookup_table_t& lookup) {
  lookup_table_ = lookup;
  count_table_.fill(0);
  for (int i = 0; i < GameConstants::kNumActions; ++i) {
    int rep = lookup[i];
    if (rep < 0) continue;
    count_table_[rep]++;
    util::debug_assert(lookup[rep] == rep);
  }
}

template <concepts::GameConstants GameConstants, group::concepts::FiniteGroup Group>
typename ActionCollapseTable<GameConstants, Group>::PolicyTensor
ActionCollapseTable<GameConstants, Group>::uncollapse(const PolicyTensor& policy) const {
  PolicyTensor output = policy;
  for (int i = 0; i < GameConstants::kNumActions; ++i) {
    core::action_t action = lookup_table_[i];
    if (action < 0) continue;
    float policy_value = policy(action);
    util::debug_assert(policy_value >= policy(i), "%f < %f", policy_value, policy(i));
    int count = count_table_[action];
    util::debug_assert(count > 0);
    if (count > Group::kOrder) {
      output(i) = policy_value / count;
    } else {
      output(i) = policy_value * util::ReciprocalTable<Group::kOrder>::values[count - 1];
    }
  }
  util::debug_assert(eigen_util::sum(output) > 0);
  return output;
}

}  // namespace core
