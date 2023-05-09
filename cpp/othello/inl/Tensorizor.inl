#include <othello/Tensorizor.hpp>

#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

namespace othello {

inline void Tensorizor::Rotation90Transform::transform_input(InputEigenTensor& tensor) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::Rotation90Transform::transform_policy(PolicyEigenTensor& vector) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::Rotation180Transform::transform_input(InputEigenTensor& tensor) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::Rotation180Transform::transform_policy(PolicyEigenTensor& vector) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::Rotation270Transform::transform_input(InputEigenTensor& tensor) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::Rotation270Transform::transform_policy(PolicyEigenTensor& vector) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::ReflectionOverHorizontalTransform::transform_input(InputEigenTensor& tensor) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::ReflectionOverHorizontalTransform::transform_policy(PolicyEigenTensor& vector) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation90Transform::transform_input(InputEigenTensor& tensor) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation90Transform::transform_policy(PolicyEigenTensor& vector) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation180Transform::transform_input(InputEigenTensor& tensor) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation180Transform::transform_policy(PolicyEigenTensor& vector) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation270Transform::transform_input(InputEigenTensor& tensor) {
  throw std::runtime_error("Not implemented");
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation270Transform::transform_policy(PolicyEigenTensor& vector) {
  throw std::runtime_error("Not implemented");
}

inline Tensorizor::transform_array_t Tensorizor::transforms() {
  transform_array_t arr{
      &transforms_struct_.identity_transform_,
      &transforms_struct_.rotation90_transform_,
      &transforms_struct_.rotation180_transform_,
      &transforms_struct_.rotation270_transform_,
      &transforms_struct_.reflection_over_horizontal_transform_,
      &transforms_struct_.reflection_over_horizontal_with_rotation90_transform_,
      &transforms_struct_.reflection_over_horizontal_with_rotation180_transform_,
      &transforms_struct_.reflection_over_horizontal_with_rotation270_transform_,
  };
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

}  // namespace othello
