#include <games/othello/Tensorizor.hpp>

#include <games/othello/Constants.hpp>
#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

namespace othello {

template<typename Scalar>
inline void Tensorizor::Rotation90Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < input.dimension(0); ++row) {
    MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
    slice.transposeInPlace();
    slice.rowwise().reverseInPlace();
  }
}

inline void Tensorizor::Rotation90Transform::transform_policy(PolicyTensor& policy) {
  PolicyMatrixSlice slice(policy.data());
  slice.rowwise().reverseInPlace();
  slice.transposeInPlace();
}

template<typename Scalar>
inline void Tensorizor::Rotation180Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
    slice.rowwise().reverseInPlace();
    slice.colwise().reverseInPlace();
  }
}

inline void Tensorizor::Rotation180Transform::transform_policy(PolicyTensor& policy) {
  PolicyMatrixSlice slice(policy.data());
  slice.rowwise().reverseInPlace();
  slice.colwise().reverseInPlace();
}

template<typename Scalar>
inline void Tensorizor::Rotation270Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
    slice.transposeInPlace();
    slice.colwise().reverseInPlace();
  }
}

inline void Tensorizor::Rotation270Transform::transform_policy(PolicyTensor& policy) {
  PolicyMatrixSlice slice(policy.data());
  slice.colwise().reverseInPlace();
  slice.transposeInPlace();
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalTransform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
    slice.colwise().reverseInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalTransform::transform_policy(PolicyTensor& policy) {
  PolicyMatrixSlice slice(policy.data());
  slice.colwise().reverseInPlace();
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalWithRotation90Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
    slice.transposeInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation90Transform::transform_policy(PolicyTensor& policy) {
  PolicyMatrixSlice slice(policy.data());
  slice.transposeInPlace();
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalWithRotation180Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
    slice.rowwise().reverseInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation180Transform::transform_policy(PolicyTensor& policy) {
  PolicyMatrixSlice slice(policy.data());
  slice.rowwise().reverseInPlace();
}

template<typename Scalar>
inline void Tensorizor::ReflectionOverHorizontalWithRotation270Transform::transform_input(InputTensorX<Scalar>& input) {
  for (int row = 0; row < 2; ++row) {
    MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
    slice.transposeInPlace();
    slice.rowwise().reverseInPlace();
    slice.colwise().reverseInPlace();
  }
}

inline void Tensorizor::ReflectionOverHorizontalWithRotation270Transform::transform_policy(PolicyTensor& policy) {
  PolicyMatrixSlice slice(policy.data());
  slice.transposeInPlace();
  slice.rowwise().reverseInPlace();
  slice.colwise().reverseInPlace();
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
