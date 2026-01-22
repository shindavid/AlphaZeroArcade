#include "core/StateIterator.hpp"

namespace core {

template <concepts::Game Game>
StateIterator<Game>& StateIterator<Game>::operator++() {
  index_ = tree_->get_parent_index(index_);
  return *this;
}

template <concepts::Game Game>
StateIterator<Game> StateIterator<Game>::operator++(int) {
  StateIterator temp = *this;
  ++(*this);
  return temp;
}

template <concepts::Game Game>
VerboseDataIterator<Game>& VerboseDataIterator<Game>::operator++() {
  index_ = tree_->get_parent_index(index_);
  return *this;
}

template <concepts::Game Game>
VerboseDataIterator<Game> VerboseDataIterator<Game>::operator++(int) {
  VerboseDataIterator temp = *this;
  ++(*this);
  return temp;
}

template <concepts::Game Game>
typename VerboseDataIterator<Game>::VerboseData_sptr VerboseDataIterator<Game>::most_recent_data() const {
  auto ix = index_;
  while (ix >= 0) {
    VerboseData_sptr data = tree_->verbose_data(ix);
    if (data != nullptr) {
      return data;
    }
    ix = tree_->get_parent_index(ix);
  }
  return nullptr;
}

}  // namespace core
