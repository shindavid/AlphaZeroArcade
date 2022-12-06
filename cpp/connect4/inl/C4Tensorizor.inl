#include <connect4/C4Tensorizor.hpp>

#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

namespace c4 {

inline void Tensorizor::ReflectionTransform::transform_input(Tensorizor::InputTensor& tensor) {
  tensor = eigen_util::reverse(tensor, 2).eval();  // axis 2 corresponds to columns
}

inline void Tensorizor::ReflectionTransform::transform_policy(Tensorizor::PolicyVector& vector) {
  std::reverse(vector.begin(), vector.end());
}

inline Tensorizor::Tensorizor()
: transforms_{&identity_transform_, &reflection_transform_}
{}

inline common::symmetry_index_t Tensorizor::get_random_symmetry_index(const GameState&) const {
  return util::Random::uniform_sample(0, transforms_.size());
}

inline Tensorizor::SymmetryTransform* Tensorizor::get_symmetry(const GameState&, common::symmetry_index_t index) const {
  return *(transforms_.begin() + index);
}

}  // namespace c4
