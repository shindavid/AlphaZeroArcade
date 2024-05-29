#include <core/Symmetries.hpp>

namespace core {

template <GameStateConcept GameState>
typename Transforms<GameState>::transform_tuple_t Transforms<GameState>::transforms_;

template <GameStateConcept GameState>
typename Transforms<GameState>::Transform* Transforms<GameState>::get(
    core::symmetry_index_t index) {
  Transform* transform = nullptr;
  constexpr size_t N = std::tuple_size_v<transform_tuple_t>;
  mp::constexpr_for<0, N, 1>([&](auto i) {
    if (i == index) {
      transform = &std::get<i>(transforms_);
    }
  });

  return transform;
}

}  // namespace core
