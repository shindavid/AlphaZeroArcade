#include <connect4/Tensorizor.hpp>

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

inline Tensorizor::SymmetryIndexSet Tensorizor::get_symmetry_indices(const GameState&) const {
  SymmetryIndexSet set;
  set.set();
  return set;
}

inline Tensorizor::SymmetryTransform* Tensorizor::get_symmetry(common::symmetry_index_t index) const {
  return *(transforms().begin() + index);
}

}  // namespace c4
