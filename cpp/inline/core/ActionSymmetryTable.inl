#include <core/ActionSymmetryTable.hpp>

#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>

#include <algorithm>
#include <cstring>
#include <ranges>

namespace core {

template <concepts::GameConstants GameConstants, group::concepts::FiniteGroup Group>
typename ActionSymmetryTable<GameConstants, Group>::PolicyTensor
ActionSymmetryTable<GameConstants, Group>::symmetrize(const PolicyTensor& policy) const {
  PolicyTensor out;
  out.setZero();
  int i = 0;
  while (i < GameConstants::kNumActions) {
    core::action_t action = action_array_[i];
    if (action < 0) break;

    int start_i = i;
    float sum = 0;
    while (i < GameConstants::kNumActions && action_array_[i] >= action) {
      sum += policy(action_array_[i++]);
    }

    int end_i = i;
    int count = end_i - start_i;

    float inv_count = util::ReciprocalTable<Group::kOrder>::get(count);
    float avg = sum * inv_count;
    for (int j = start_i; j < end_i; ++j) {
      out(action_array_[j]) = avg;
    }
  }
  return out;
}

template <concepts::GameConstants GameConstants, group::concepts::FiniteGroup Group>
typename ActionSymmetryTable<GameConstants, Group>::PolicyTensor
ActionSymmetryTable<GameConstants, Group>::collapse(const PolicyTensor& policy) const {
  PolicyTensor out;
  out.setZero();
  int i = 0;
  while (i < GameConstants::kNumActions) {
    core::action_t action = action_array_[i];
    if (action < 0) break;

    int start_i = i;
    float sum = 0;
    while (i < GameConstants::kNumActions && action_array_[i] >= action) {
      sum += policy(action_array_[i++]);
    }

    out(action_array_[start_i]) = sum;
  }
  return out;
}

template <concepts::GameConstants GameConstants, group::concepts::FiniteGroup Group>
boost::json::array ActionSymmetryTable<GameConstants, Group>::to_json() const {
  boost::json::array action_array_json;
  for (const auto& action : action_array_) {
    if (action < 0) break;
    action_array_json.push_back(action);
  }
  return action_array_json;
}

}  // namespace core
