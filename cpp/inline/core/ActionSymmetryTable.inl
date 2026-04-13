#include "core/ActionSymmetryTable.hpp"

#include "util/Asserts.hpp"
#include "util/CppUtil.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace core {

template <::alpha0::concepts::EvalSpec EvalSpec>
void ActionSymmetryTable<EvalSpec>::load(std::vector<Item>& items) {
  int num_items = items.size();
  std::sort(items.begin(), items.begin() + num_items);

  // items is now a pseudo-list of sets [S_1, S_2, ...], where S_i is a set of symmetrically
  // equivalent moves, and where each S_i is sorted in increasing order

  struct Pair {
    auto operator<=>(const Pair&) const = default;
    Move move;
    int cluster_start_index;
  };
  using pair_array_t = std::array<Pair, kMaxNumActions>;

  pair_array_t pair_array;
  int num_pairs = 0;
  int last_equivalence_class = std::numeric_limits<int>::min();
  for (int i = 0; i < num_items; ++i) {
    auto& item = items[i];
    if (item.equivalence_class != last_equivalence_class) {
      pair_array[num_pairs++] = {item.move, i};
      last_equivalence_class = item.equivalence_class;
    }
  }

  std::sort(pair_array.begin(), pair_array.begin() + num_pairs, std::greater{});

  // now pair_array is a pseudo-map of min(S) -> &S for each set S in items

  move_array_t move_array;
  int i = 0;
  for (int p = 0; p < num_pairs; ++p) {
    int start_index = pair_array[p].cluster_start_index;
    auto equivalence_class = items[start_index].equivalence_class;
    for (int index = start_index;
         index < num_items && items[index].equivalence_class == equivalence_class; ++index) {
      move_array[i++] = items[index].move;
    }
  }
  DEBUG_ASSERT(i == num_items);

  // now move_array is the same as items, but with the sets themselves sorted in decreasing order
  // by their minimum element
  move_array_ = move_array;
  num_moves_ = num_items;
}

template <::alpha0::concepts::EvalSpec EvalSpec>
typename ActionSymmetryTable<EvalSpec>::PolicyTensor ActionSymmetryTable<EvalSpec>::symmetrize(
  const InputFrame& frame, const PolicyTensor& policy) const {
  PolicyTensor out;
  out.setZero();
  int i = 0;
  while (i < num_moves_) {
    Move move = move_array_[i];

    int start_i = i;
    float sum = 0;
    while (i < num_moves_ && move_array_[i] >= move) {
      sum += policy.coeff(PolicyEncoding::to_index(frame, move_array_[i++]));
    }

    int end_i = i;
    int count = end_i - start_i;

    float inv_count = util::ReciprocalTable<Group::kOrder>::get(count);
    float avg = sum * inv_count;
    for (int j = start_i; j < end_i; ++j) {
      out.coeffRef(PolicyEncoding::to_index(frame, move_array_[j])) = avg;
    }
  }
  return out;
}

template <::alpha0::concepts::EvalSpec EvalSpec>
typename ActionSymmetryTable<EvalSpec>::PolicyTensor ActionSymmetryTable<EvalSpec>::collapse(
  const InputFrame& frame, const PolicyTensor& policy) const {
  PolicyTensor out;
  out.setZero();
  int i = 0;
  while (i < num_moves_) {
    Move move = move_array_[i];

    int start_i = i;
    float sum = 0;
    while (i < num_moves_ && move_array_[i] >= move) {
      sum += policy.coeff(PolicyEncoding::to_index(frame, move_array_[i++]));
    }

    out.coeffRef(PolicyEncoding::to_index(frame, move_array_[start_i])) = sum;
  }
  return out;
}

template <::alpha0::concepts::EvalSpec EvalSpec>
boost::json::array ActionSymmetryTable<EvalSpec>::to_json() const {
  boost::json::array move_array_json;
  boost::json::array equivalence_class_json;
  int i = 0;
  while (i < num_moves_) {
    Move move = move_array_[i];

    equivalence_class_json = {};
    while (i < num_moves_ && move_array_[i] >= move) {
      equivalence_class_json.emplace_back(Game::IO::move_to_json_value(move_array_[i++]));
    }
    move_array_json.push_back(equivalence_class_json);
  }
  return move_array_json;
}

}  // namespace core
