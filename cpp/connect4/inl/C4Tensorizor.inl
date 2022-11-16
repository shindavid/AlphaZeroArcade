#include <connect4/C4Tensorizor.hpp>

#include <util/Random.hpp>

namespace c4 {

inline void ReflectionTransform::transform_input(torch::Tensor input) {
  throw std::exception();
}

inline void ReflectionTransform::transform_policy(torch::Tensor input) {
  throw std::exception();
}

inline Tensorizor::Tensorizor()
: transforms_{&identity_transform_, &reflection_transform_}
{}

inline common::AbstractSymmetryTransform* Tensorizor::get_random_symmetry(const GameState&) const {
  return *(transforms_.begin() + util::Random::uniform_draw(0, transforms_.size()));
}

}  // namespace c4
