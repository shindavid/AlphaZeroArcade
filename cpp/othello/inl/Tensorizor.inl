#include <othello/Tensorizor.hpp>

#include <othello/Constants.hpp>
#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

namespace othello {

inline Tensorizor::CenterFourSquares Tensorizor::get_center_four_squares(const PolicyTensor& policy) {
  CenterFourSquares center;
  center.starting_white1 = policy.data()[kStartingWhite1];
  center.starting_white2 = policy.data()[kStartingWhite2];
  center.starting_black1 = policy.data()[kStartingBlack1];
  center.starting_black2 = policy.data()[kStartingBlack2];
  return center;
}

inline void Tensorizor::set_center_four_squares(PolicyTensor& policy, const CenterFourSquares& center_four_squares)
{
  policy.data()[kStartingWhite1] = center_four_squares.starting_white1;
  policy.data()[kStartingWhite2] = center_four_squares.starting_white2;
  policy.data()[kStartingBlack1] = center_four_squares.starting_black1;
  policy.data()[kStartingBlack2] = center_four_squares.starting_black2;
}

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
  auto center = get_center_four_squares(policy);
  auto& matrix = as_matrix(policy);
  matrix.rowwise().reverseInPlace();
  matrix.transposeInPlace();
  set_center_four_squares(policy, center);
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
  auto center = get_center_four_squares(policy);
  auto& matrix = as_matrix(policy);
  matrix.rowwise().reverseInPlace();
  matrix.colwise().reverseInPlace();
  set_center_four_squares(policy, center);
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
  auto center = get_center_four_squares(policy);
  auto& matrix = as_matrix(policy);
  matrix.colwise().reverseInPlace();
  matrix.transposeInPlace();
  set_center_four_squares(policy, center);
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalTransform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.colwise().reverseInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalTransform::transform_policy(PolicyTensor& policy) {
  auto center = get_center_four_squares(policy);
  auto& matrix = as_matrix(policy);
  matrix.colwise().reverseInPlace();
  set_center_four_squares(policy, center);
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalWithRotation90Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.transposeInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation90Transform::transform_policy(PolicyTensor& policy) {
  auto center = get_center_four_squares(policy);
  auto& matrix = as_matrix(policy);
  matrix.transposeInPlace();
  set_center_four_squares(policy, center);
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalWithRotation180Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    auto& matrix = slice_as_matrix(input, row);
    matrix.rowwise().reverseInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation180Transform::transform_policy(PolicyTensor& policy) {
  auto center = get_center_four_squares(policy);
  auto& matrix = as_matrix(policy);
  matrix.rowwise().reverseInPlace();
  set_center_four_squares(policy, center);
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
  auto center = get_center_four_squares(policy);
  auto& matrix = as_matrix(policy);
  matrix.transposeInPlace();
  matrix.rowwise().reverseInPlace();
  matrix.colwise().reverseInPlace();
  set_center_four_squares(policy, center);
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

}  // namespace othello
