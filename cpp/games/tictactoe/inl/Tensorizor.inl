#include <games/tictactoe/Tensorizor.hpp>

#include <games/tictactoe/Constants.hpp>
#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

namespace tictactoe {

template<typename Scalar>
inline auto& Tensorizor::slice_as_matrix(InputTensorX<Scalar>& input, int row) {
  return eigen_util::reinterpret_as_matrix<MatrixT<Scalar>>(eigen_util::slice(input, row));
}

inline Tensorizor::MatrixT<Tensorizor::PolicyScalar>& Tensorizor::as_matrix(PolicyTensor& policy) {
  return eigen_util::reinterpret_as_matrix<MatrixT<PolicyScalar>>(policy);
}

template<typename Scalar>
inline void Tensorizor::Rotation90Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.transposeInPlace();
    matrix.rowwise().reverseInPlace();
  }
}

inline void Tensorizor::Rotation90Transform::transform_policy(PolicyTensor& policy) {
  auto& matrix = as_matrix(policy);
  matrix.rowwise().reverseInPlace();
  matrix.transposeInPlace();
}

template<typename Scalar>
inline void Tensorizor::Rotation180Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.rowwise().reverseInPlace();
    matrix.colwise().reverseInPlace();
  }
}

inline void Tensorizor::Rotation180Transform::transform_policy(PolicyTensor& policy) {
  auto& matrix = as_matrix(policy);
  matrix.rowwise().reverseInPlace();
  matrix.colwise().reverseInPlace();
}

template<typename Scalar>
inline void Tensorizor::Rotation270Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.transposeInPlace();
    matrix.colwise().reverseInPlace();
  }
}

inline void Tensorizor::Rotation270Transform::transform_policy(PolicyTensor& policy) {
  auto& matrix = as_matrix(policy);
  matrix.colwise().reverseInPlace();
  matrix.transposeInPlace();
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalTransform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.colwise().reverseInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalTransform::transform_policy(PolicyTensor& policy) {
  auto& matrix = as_matrix(policy);
  matrix.colwise().reverseInPlace();
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalWithRotation90Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.transposeInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation90Transform::transform_policy(PolicyTensor& policy) {
  auto& matrix = as_matrix(policy);
  matrix.transposeInPlace();
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalWithRotation180Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.rowwise().reverseInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation180Transform::transform_policy(PolicyTensor& policy) {
  auto& matrix = as_matrix(policy);
  matrix.rowwise().reverseInPlace();
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalWithRotation270Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.transposeInPlace();
    matrix.rowwise().reverseInPlace();
    matrix.colwise().reverseInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation270Transform::transform_policy(PolicyTensor& policy) {
  auto& matrix = as_matrix(policy);
  matrix.transposeInPlace();
  matrix.rowwise().reverseInPlace();
  matrix.colwise().reverseInPlace();
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

inline Tensorizor::SymmetryTransform* Tensorizor::get_symmetry(core::symmetry_index_t index) const {
  return *(transforms().begin() + index);
}

}  // namespace tictactoe
