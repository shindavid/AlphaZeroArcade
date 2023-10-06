#pragma once

#include <core/AbstractSymmetryTransform.hpp>
#include <core/DerivedTypes.hpp>
#include <core/IdentityTransform.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace common {

/*
 * SquareBoardSymmetries can be used as a base class for tensorizors for games played on a square
 * board with 8 symmetries, corresponding to 90-degree rotations and reflections.
 *
 * BoardShape_ is the shape of the board, and is expected to be of the form Eigen::Sizes<B, B>. If
 * it is not specified, then it will be inferred from Tensor_.
 *
 * Tensor_ is the type of the tensor that will be transformed. It will typically be of shape
 * Eigen::Sizes<K, B, B> or Eigen::Sizes<B, B>. If it is of either of these forms, then
 * BoardShape_ can be inferred from the last 2 dimensions. If it is of the Eigen::Sizes<K, B, B>
 * form, then the 2D-transform is performed on each of the K slices.
 *
 * In some cases, Tensor_'s shape may not be of the form Eigen::Sizes<K, B, B> or
 * Eigen::Sizes<B, B>. In such cases, BoardShape_ must be specified. The transform will then be
 * applied by implicitly reshaping the tensor to BoardShape_. This case occurs, for instance, for
 * games where the policy tensor is "flattened", which we might do if there is a "pass" move, which
 * is true in games like Othello or go. In such cases, we typically use a policy tensor of shape
 * Eigen::Sizes<B*B + 1>, with the +1 for the pass move. The implicit reshape treats the first B*B
 * entries of the tensor as a tensor of shape Eigen::Sizes<B, B>.
 */
template <eigen_util::FixedTensorConcept Tensor_,
          eigen_util::ShapeConcept BoardShape_ = eigen_util::Shape<>>
class SquareBoardSymmetries {
 public:
  using Tensor = Tensor_;
  using Scalar = typename Tensor::Scalar;
  using TensorShape = typename Tensor::Dimensions;

  static constexpr int kNumSymmetries = 8;
  static constexpr int kTensorRank = eigen_util::rank_v<TensorShape>;
  static constexpr int kTensorDim0 = eigen_util::extract_dim_v<0, TensorShape>;
  static constexpr int kTensorDimX =
      eigen_util::extract_dim_v<(kTensorRank >= 2 ? kTensorRank - 2 : 0), TensorShape>;
  static constexpr int kTensorDimY =
      eigen_util::extract_dim_v<(kTensorRank >= 2 ? kTensorRank - 1 : 0), TensorShape>;

  using BoardShape = std::conditional_t<(eigen_util::rank_v<BoardShape_> > 0), BoardShape_,
                                        eigen_util::Shape<kTensorDimX, kTensorDimY>>;

  static constexpr int kBoardDimX = eigen_util::extract_dim_v<0, BoardShape>;
  static constexpr int kBoardDimY = eigen_util::extract_dim_v<1, BoardShape>;
  static_assert(kBoardDimX == kBoardDimY, "BoardShape must be of form Eigen::Sizes<B, B>");

  static constexpr int kBoardLength = kBoardDimX;
  static constexpr bool kShapeMatch = kTensorDimX == kBoardDimX && kTensorDimY == kBoardDimY;
  static_assert(kBoardLength > 0, "Unexpected BoardShape");
  static_assert(!kShapeMatch || kTensorRank == 2 || kTensorRank == 3, "Unexpected TensorShape");

  static constexpr int kNumRows = (kShapeMatch && kTensorRank == 3) ? kTensorDim0 : 1;

  using Transform = core::AbstractSymmetryTransform<Tensor>;
  using Identity = core::IdentityTransform<Tensor>;
  using MatrixSlice = Eigen::Map<
      Eigen::Matrix<Scalar, kBoardLength, kBoardLength, Eigen::RowMajor | Eigen::DontAlign>>;

  using transform_array_t = std::array<Transform*, kNumSymmetries>;

  static void rot90(Tensor& t) {
    for (int row = 0; row < kNumRows; ++row) {
      MatrixSlice slice(t.data() + row * kBoardLength * kBoardLength);
      slice.transposeInPlace();
      slice.rowwise().reverseInPlace();
    }
  }

  static void rot180(Tensor& t) {
    for (int row = 0; row < kNumRows; ++row) {
      MatrixSlice slice(t.data() + row * kBoardLength * kBoardLength);
      slice.rowwise().reverseInPlace();
      slice.colwise().reverseInPlace();
    }
  }

  static void rot270(Tensor& t) {
    for (int row = 0; row < kNumRows; ++row) {
      MatrixSlice slice(t.data() + row * kBoardLength * kBoardLength);
      slice.transposeInPlace();
      slice.colwise().reverseInPlace();
    }
  }

  static void refl(Tensor& t) {
    for (int row = 0; row < kNumRows; ++row) {
      MatrixSlice slice(t.data() + row * kBoardLength * kBoardLength);
      slice.colwise().reverseInPlace();
    }
  }

  static void refl_rot90(Tensor& t) {
    for (int row = 0; row < kNumRows; ++row) {
      MatrixSlice slice(t.data() + row * kBoardLength * kBoardLength);
      slice.transposeInPlace();
    }
  }

  static void refl_rot180(Tensor& t) {
    for (int row = 0; row < kNumRows; ++row) {
      MatrixSlice slice(t.data() + row * kBoardLength * kBoardLength);
      slice.rowwise().reverseInPlace();
    }
  }

  static void refl_rot270(Tensor& t) {
    for (int row = 0; row < kNumRows; ++row) {
      MatrixSlice slice(t.data() + row * kBoardLength * kBoardLength);
      slice.transposeInPlace();
      slice.rowwise().reverseInPlace();
      slice.colwise().reverseInPlace();
    }
  }

  struct Rot90 : public Transform {
    void apply(Tensor& t) override { rot90(t); }
    void undo(Tensor& t) override { rot270(t); }
  };

  struct Rot180 : public Transform {
    void apply(Tensor& t) override { rot180(t); }
    void undo(Tensor& t) override { rot180(t); }
  };

  struct Rot270 : public Transform {
    void apply(Tensor& t) override { rot270(t); }
    void undo(Tensor& t) override { rot90(t); }
  };

  struct Refl : public Transform {
    void apply(Tensor& t) override { refl(t); }
    void undo(Tensor& t) override { refl(t); }
  };

  struct ReflRot90 : public Transform {
    void apply(Tensor& t) override { refl_rot90(t); }
    void undo(Tensor& t) override { refl_rot90(t); }
  };

  struct ReflRot180 : public Transform {
    void apply(Tensor& t) override { refl_rot180(t); }
    void undo(Tensor& t) override { refl_rot180(t); }
  };

  struct ReflRot270 : public Transform {
    void apply(Tensor& t) override { refl_rot270(t); }
    void undo(Tensor& t) override { refl_rot270(t); }
  };

  static Transform* get_symmetry(core::symmetry_index_t index) {
    return *(transforms().begin() + index);
  }

 protected:
  static transform_array_t transforms() {
    transform_array_t arr{&transforms_struct_.identity_,    &transforms_struct_.rot90_,
                          &transforms_struct_.rot180_,      &transforms_struct_.rot270_,
                          &transforms_struct_.refl,         &transforms_struct_.refl_rot90_,
                          &transforms_struct_.refl_rot180_, &transforms_struct_.refl_rot270_};
    return arr;
  }

  struct transforms_struct_t {
    Identity identity_;
    Rot90 rot90_;
    Rot180 rot180_;
    Rot270 rot270_;
    Refl refl;
    ReflRot90 refl_rot90_;
    ReflRot180 refl_rot180_;
    ReflRot270 refl_rot270_;
  };

  static transforms_struct_t transforms_struct_;
};

template <eigen_util::FixedTensorConcept Tensor, eigen_util::ShapeConcept BoardShape>
SquareBoardSymmetries<Tensor, BoardShape>::transforms_struct_t
    SquareBoardSymmetries<Tensor, BoardShape>::transforms_struct_;

}  // namespace common
