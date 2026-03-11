#include "core/DefaultCanonicalizer.hpp"

namespace core {

template <typename InputFrame, typename Symmetries>
group::element_t DefaultCanonicalizer<InputFrame, Symmetries>::get(const InputFrame& frame) {
  group::element_t best_sym = 0;
  InputFrame best_frame = frame;

  auto mask = Symmetries::get_mask(frame);
  for (group::element_t sym : mask.on_indices()) {
    InputFrame transformed_frame = frame;
    Symmetries::apply(transformed_frame, sym);
    if (transformed_frame < best_frame) {
      best_sym = sym;
      best_frame = transformed_frame;
    }
  }
  return best_sym;
}

}  // namespace core
