#include <core/ActionCollapseTable.hpp>

#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>

#include <algorithm>
#include <cstring>
#include <ranges>

namespace core {

template <concepts::GameConstants GameConstants, group::concepts::FiniteGroup Group>
typename ActionSymmetryTable<GameConstants, Group>::PolicyTensor
ActionSymmetryTable<GameConstants, Group>::symmetrize(const PolicyTensor& policy) const {
  // int i = 0;
  // while (i < GameConstants::kNumActions && action_array_[i] >= 0) {
  //   core::action_t action = action_array_[i];

  //   action_array_t equivalence_class;
  //   int j = 0;
  //   equivalence_class[j++] = action;

  //   // TODO
  // }
  throw util::Exception("Not implemented");
}

template <concepts::GameConstants GameConstants, group::concepts::FiniteGroup Group>
typename ActionSymmetryTable<GameConstants, Group>::PolicyTensor
ActionSymmetryTable<GameConstants, Group>::collapse(const PolicyTensor& policy) const {
  throw util::Exception("Not implemented");
}

}  // namespace core
