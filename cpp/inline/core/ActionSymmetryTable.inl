#include <core/ActionSymmetryTable.hpp>

#include <util/Asserts.hpp>
#include <util/CppUtil.hpp>

#include <algorithm>
#include <cstring>
#include <ranges>

namespace core {

template <int kMaxNumActions, group::concepts::FiniteGroup Group>
void ActionSymmetryTable<kMaxNumActions, Group>::load(std::vector<item_t>& items) {
  int num_items = items.size();
  std::sort(items.begin(), items.begin() + num_items);

  // items is now a pseudo-list of sets [S_1, S_2, ...], where S_i is a set of symmetrically
  // equivalent actions, and where each S_i is sorted in increasing order

  struct pair_t {
    auto operator<=>(const pair_t&) const = default;
    core::action_t action;
    int cluster_start_index;
  };
  using pair_array_t = std::array<pair_t, kMaxNumActions>;

  pair_array_t pair_array;
  int num_pairs = 0;
  int last_equivalence_class = -1;
  for (int i = 0; i < num_items; ++i) {
    auto& item = items[i];
    if (item.equivalence_class != last_equivalence_class) {
      pair_array[num_pairs++] = {item.action, i};
      last_equivalence_class = item.equivalence_class;
    }
  }

  std::sort(pair_array.begin(), pair_array.begin() + num_pairs, std::greater{});

  // now pair_array is a pseudo-map of min(S) -> &S for each set S in items

  action_array_t action_array;
  int i = 0;
  for (int p = 0; p < num_pairs; ++p) {
    int start_index = pair_array[p].cluster_start_index;
    auto equivalence_class = items[start_index].equivalence_class;
    for (int index = start_index; index < num_items && items[index].equivalence_class == equivalence_class; ++index) {
      action_array[i++] = items[index].action;
    }
  }
  util::debug_assert(i == num_items);

  if (num_items < kMaxNumActions) {
    action_array[num_items] = -1;
  }

  // now action_array is the same as items, but with the sets themselves sorted in decreasing order
  // by their minimum element
  action_array_ = action_array;
}

template <int kMaxNumActions, group::concepts::FiniteGroup Group>
typename ActionSymmetryTable<kMaxNumActions, Group>::PolicyTensor
ActionSymmetryTable<kMaxNumActions, Group>::symmetrize(const PolicyTensor& policy) const {
  PolicyTensor out;
  out.setZero();
  int i = 0;
  while (i < kMaxNumActions) {
    core::action_t action = action_array_[i];
    if (action < 0) break;

    int start_i = i;
    float sum = 0;
    while (i < kMaxNumActions && action_array_[i] >= action) {
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

template <int kMaxNumActions, group::concepts::FiniteGroup Group>
typename ActionSymmetryTable<kMaxNumActions, Group>::PolicyTensor
ActionSymmetryTable<kMaxNumActions, Group>::collapse(const PolicyTensor& policy) const {
  PolicyTensor out;
  out.setZero();
  int i = 0;
  while (i < kMaxNumActions) {
    core::action_t action = action_array_[i];
    if (action < 0) break;

    int start_i = i;
    float sum = 0;
    while (i < kMaxNumActions && action_array_[i] >= action) {
      sum += policy(action_array_[i++]);
    }

    out(action_array_[start_i]) = sum;
  }
  return out;
}

template <int kMaxNumActions, group::concepts::FiniteGroup Group>
boost::json::array ActionSymmetryTable<kMaxNumActions, Group>::to_json() const {
  boost::json::array action_array_json;
  boost::json::array equivalence_class_json;
  int i = 0;
  while (i < kMaxNumActions) {
    core::action_t action = action_array_[i];
    if (action < 0) break;

    equivalence_class_json = {};
    while (i < kMaxNumActions && action_array_[i] >= action) {
      equivalence_class_json.push_back(action_array_[i++]);
    }
    action_array_json.push_back(equivalence_class_json);
  }
  return action_array_json;
}

}  // namespace core
