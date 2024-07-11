#include <core/DefaultCanonicalizer.hpp>

namespace core {

template <concepts::Game Game>
group::element_t DefaultCanonicalizer<Game>::get(const BaseState& state) {
  group::element_t best_sym = 0;
  BaseState best_state = state;

  auto mask = Game::Symmetries::get_mask(state);
  for (group::element_t sym : bitset_util::on_indices(mask)) {
    BaseState transformed_state = state;
    Game::Symmetries::apply(transformed_state, sym);
    if (transformed_state < best_state) {
      best_sym = sym;
      best_state = transformed_state;
    }
  }
  return best_sym;
}

}  // namespace core
