#include "search/RootInfo.hpp"

namespace search {

template <search::concepts::Traits Traits>
void RootInfoBase<Traits>::clear() {
  State state;
  Game::Rules::init_state(state);
  history.clear();
  history.update(state);

  node_index = -1;
  active_seat = -1;
  add_noise = false;
}

template <search::concepts::Traits Traits>
void RootInfoImpl<Traits, core::kSymmetryTranspositions>::clear() {
  RootInfoBase<Traits>::clear();
  canonical_sym = Game::Symmetries::get_canonical_symmetry(this->history.current());
}

}  // namespace search
