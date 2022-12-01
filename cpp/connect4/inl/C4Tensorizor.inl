#include <connect4/C4Tensorizor.hpp>

#include <util/Random.hpp>

namespace c4 {

inline void Tensorizor::ReflectionTransform::transform_input(Tensorizor::InputTensor&) {
  // TODO
}

inline void Tensorizor::ReflectionTransform::transform_policy(Tensorizor::PolicyVector&) {
  // TODO
}

inline Tensorizor::Tensorizor()
: transforms_{&identity_transform_, &reflection_transform_}
{}

inline Tensorizor::SymmetryTransform* Tensorizor::get_random_symmetry(const GameState&) const {
  return *(transforms_.begin() + util::Random::uniform_sample(0, transforms_.size()));
}

}  // namespace c4
