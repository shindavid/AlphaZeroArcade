#include "core/DefaultCanonicalizer.hpp"

namespace core {

template <concepts::Game Game, typename Symmetries>
group::element_t DefaultCanonicalizer<Game, Symmetries>::get(const State& state) {
  group::element_t best_sym = 0;
  State best_state = state;

  auto mask = Symmetries::get_mask(state);
  for (group::element_t sym : mask.on_indices()) {
    State transformed_state = state;
    Symmetries::apply(transformed_state, sym);
    if (transformed_state < best_state) {
      best_sym = sym;
      best_state = transformed_state;
    }
  }
  return best_sym;
}

}  // namespace core
