#include "core/InfoSetIterator.hpp"

namespace core {

template <concepts::Game Game>
InfoSetIterator<Game>& InfoSetIterator<Game>::operator++() {
  index_ = tree_->get_parent_index(index_);
  return *this;
}

template <concepts::Game Game>
InfoSetIterator<Game> InfoSetIterator<Game>::operator++(int) {
  InfoSetIterator temp = *this;
  ++(*this);
  return temp;
}

template <concepts::Game Game>
game_tree_node_aux_t InfoSetIterator<Game>::get_player_aux() const {
  auto parent_index = tree_->get_parent_index(index_);
  if (parent_index >= 0) {
    return tree_->get_player_aux(parent_index, tree_->get_active_seat(parent_index));
  }
  return 0;
}

}  // namespace core
