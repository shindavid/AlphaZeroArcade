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

}  // namespace core
