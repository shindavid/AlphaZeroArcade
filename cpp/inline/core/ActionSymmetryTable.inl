#include "core/ActionSymmetryTable.hpp"

#include <algorithm>
#include <cstring>

namespace core {

template <::alpha0::concepts::Spec Spec>
void ActionSymmetryTable<Spec>::Builder::add(const Move& move, const InputFrame& frame) {
  group::element_t sym = Symmetries::get_canonical_symmetry(frame);
  InputFrame frame_copy = frame;
  Symmetries::apply(frame_copy, sym);

  int move_index = moves_.size();

  auto [it, inserted] = frame_map_.try_emplace(frame_copy, frame_map_.size());
  moves_.push_back(move);
  items_.emplace_back(it->second, move_index);
}

template <::alpha0::concepts::Spec Spec>
void ActionSymmetryTable<Spec>::Builder::clear() {
  moves_.clear();
  frame_map_.clear();
  items_.clear();
}

template <::alpha0::concepts::Spec Spec>
void ActionSymmetryTable<Spec>::load(Builder& builder) {
  num_equivalence_classes_ = builder.frame_map_.size();
  size_ = builder.moves_.size();

  std::sort(builder.items_.begin(), builder.items_.end());

  int i = 0;
  for (const auto& item : builder.items_) {
    move_array_[i] = builder.moves_[item.move_index];
    equivalence_class_array_[i] = item.equivalence_class;
    ++i;
  }

  builder.clear();
}

template <::alpha0::concepts::Spec Spec>
typename ActionSymmetryTable<Spec>::PolicyTensor ActionSymmetryTable<Spec>::symmetrize(
  const InputFrame& frame, const PolicyTensor& policy) const {
  PolicyTensor out;
  out.setZero();
  int i = 0;
  while (i < size_) {
    equivalence_class_t ec = equivalence_class_array_[i];

    int start_i = i;
    float sum = 0;
    while (i < size_ && equivalence_class_array_[i] == ec) {
      sum += policy.coeff(PolicyEncoding::to_index(frame, move_array_[i++]));
    }

    int end_i = i;
    int count = end_i - start_i;

    float avg = sum / count;
    for (int j = start_i; j < end_i; ++j) {
      out.coeffRef(PolicyEncoding::to_index(frame, move_array_[j])) = avg;
    }
  }
  return out;
}

template <::alpha0::concepts::Spec Spec>
typename ActionSymmetryTable<Spec>::PolicyTensor ActionSymmetryTable<Spec>::collapse(
  const InputFrame& frame, const PolicyTensor& policy) const {
  PolicyTensor out;
  out.setZero();
  int i = 0;
  while (i < size_) {
    equivalence_class_t ec = equivalence_class_array_[i];

    int start_i = i;
    float sum = 0;
    while (i < size_ && equivalence_class_array_[i] == ec) {
      sum += policy.coeff(PolicyEncoding::to_index(frame, move_array_[i++]));
    }

    out.coeffRef(PolicyEncoding::to_index(frame, move_array_[start_i])) = sum;
  }
  return out;
}

template <::alpha0::concepts::Spec Spec>
boost::json::array ActionSymmetryTable<Spec>::to_json() const {
  boost::json::array move_array_json;
  boost::json::array equivalence_class_json;
  int i = size_ - 1;
  while (i >= 0) {
    equivalence_class_t ec = equivalence_class_array_[i];

    std::vector<Move> moves;
    while (i >= 0 && equivalence_class_array_[i] == ec) {
      moves.push_back(move_array_[i--]);
    }

    std::reverse(moves.begin(), moves.end());
    equivalence_class_json = {};
    for (const auto& move : moves) {
      equivalence_class_json.emplace_back(Game::IO::move_to_json_value(move));
    }
    move_array_json.push_back(equivalence_class_json);
  }
  return move_array_json;
}

}  // namespace core
