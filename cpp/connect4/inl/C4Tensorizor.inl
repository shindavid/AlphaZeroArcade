#include <connect4/C4Tensorizor.hpp>

#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

namespace c4 {

inline void Tensorizor::ReflectionTransform::transform_input(InputEigenSlab& tensor) {
  tensor = eigen_util::reverse(tensor, 2).eval();  // axis 2 corresponds to columns
}

inline void Tensorizor::ReflectionTransform::transform_policy(PolicyEigenSlab& vector) {
  std::reverse(vector.begin(), vector.end());
}

inline Tensorizor::transform_array_t Tensorizor::transforms() {
  transform_array_t arr{&identity_transform_, &reflection_transform_};
  return arr;
}

inline common::symmetry_index_t Tensorizor::get_random_symmetry_index(const GameState&) const {
  return util::Random::uniform_sample(0, transforms().size());
}

inline Tensorizor::SymmetryTransform* Tensorizor::get_symmetry(common::symmetry_index_t index) const {
  return *(transforms().begin() + index);
}

}  // namespace c4
