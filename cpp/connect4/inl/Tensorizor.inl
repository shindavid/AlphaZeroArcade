#include <connect4/Tensorizor.hpp>

#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

namespace c4 {

inline void Tensorizor::ReflectionTransform::transform_input(InputTensor& tensor) {
  eigen_util::packed_fixed_tensor_cp(tensor, eigen_util::reverse(tensor, 1).eval());  // axis 1 corresponds to columns
}

inline void Tensorizor::ReflectionTransform::transform_policy(PolicyTensor& policy) {
  eigen_util::packed_fixed_tensor_cp(policy, eigen_util::reverse(policy, 0).eval());  // axis 0 corresponds to columns
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

inline Tensorizor::SymmetryTransform* Tensorizor::get_symmetry(core::symmetry_index_t index) const {
  return *(transforms().begin() + index);
}

}  // namespace c4
