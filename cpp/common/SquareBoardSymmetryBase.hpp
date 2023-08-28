#pragma once

#include <core/AbstractSymmetryTransform.hpp>
#include <core/GameStateConcept.hpp>
#include <core/DerivedTypes.hpp>
#include <core/IdentityTransform.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace common {

/*
 * SquareBoardSymmetryBase can be used as a base class for tensorizors for games played on a
 * square board with 8 symmetries, corresponding to 90-degree rotations and reflections.
 */
template<core::GameStateConcept GameState, eigen_util::ShapeConcept InputShape_>
class SquareBoardSymmetryBase {
public:
  using InputShape = InputShape_;
  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kMaxNumSymmetries = 8;
  static constexpr int kBoardDimension = eigen_util::extract_dim_v<1, InputShape>;
  static constexpr int kNumCells = kBoardDimension * kBoardDimension;
  static_assert(kBoardDimension == eigen_util::extract_dim_v<2, InputShape>, "InputShape must be square");

  using InputTensor = Eigen::TensorFixedSize<bool, InputShape, Eigen::RowMajor>;
  template<typename Scalar> using InputTensorX = Eigen::TensorFixedSize<Scalar, InputShape, Eigen::RowMajor>;

  using GameStateTypes = core::GameStateTypes<GameState>;
  using SymmetryIndexSet = std::bitset<kMaxNumSymmetries>;
  using PolicyTensor = GameStateTypes::PolicyTensor;
  using SymmetryTransform = core::AbstractSymmetryTransform<InputTensor, PolicyTensor>;
  using Identity = core::IdentityTransform<InputTensor, PolicyTensor>;
  using transform_array_t = std::array<SymmetryTransform*, kMaxNumSymmetries>;

  using InputScalar = typename InputTensor::Scalar;
  using PolicyScalar = typename PolicyTensor::Scalar;

  template <typename Scalar>
  using MatrixSliceX =
      Eigen::Map<Eigen::Matrix<Scalar, kBoardDimension, kBoardDimension,
                               Eigen::RowMajor | Eigen::DontAlign>>;

  using PolicyMatrixSlice = MatrixSliceX<PolicyScalar>;

  struct Rot90 : public SymmetryTransform {
    template<typename Scalar>
    void transform_input(InputTensorX<Scalar>& input) {
      for (int row = 0; row < input.dimension(0); ++row) {
        MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
        slice.transposeInPlace();
        slice.rowwise().reverseInPlace();
      }
    }

    void transform_input(InputTensor& input) override {
      transform_input<InputScalar>(input);
    }

    void transform_policy(PolicyTensor& policy) override {
      PolicyMatrixSlice slice(policy.data());
      slice.rowwise().reverseInPlace();
      slice.transposeInPlace();
    }
  };

  struct Rot180 : public SymmetryTransform {
    template <typename Scalar>
    void transform_input(InputTensorX<Scalar>& input) {
      for (int row = 0; row < input.dimension(0); ++row) {
        MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
        slice.rowwise().reverseInPlace();
        slice.colwise().reverseInPlace();
      }
    }

    void transform_input(InputTensor& input) override {
      transform_input<InputScalar>(input);
    }

    void transform_policy(PolicyTensor& policy) override {
      PolicyMatrixSlice slice(policy.data());
      slice.rowwise().reverseInPlace();
      slice.colwise().reverseInPlace();
    }
  };

  struct Rot270 : public SymmetryTransform {
    template<typename Scalar>
    void transform_input(InputTensorX<Scalar>& input) {
      for (int row = 0; row < input.dimension(0); ++row) {
        MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
        slice.transposeInPlace();
        slice.colwise().reverseInPlace();
      }
    }

    void transform_input(InputTensor& input) override {
      transform_input<InputScalar>(input);
    }

    void transform_policy(PolicyTensor& policy) override {
      PolicyMatrixSlice slice(policy.data());
      slice.colwise().reverseInPlace();
      slice.transposeInPlace();
    }
  };

  struct Refl : public SymmetryTransform {
    template<typename Scalar>
    void transform_input(InputTensorX<Scalar>& input) {
      for (int row = 0; row < input.dimension(0); ++row) {
        MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
        slice.colwise().reverseInPlace();
      }
    }

    void transform_input(InputTensor& input) override {
      transform_input<InputScalar>(input);
    }

    void transform_policy(PolicyTensor& policy) override {
      PolicyMatrixSlice slice(policy.data());
      slice.colwise().reverseInPlace();
    }
  };

  struct ReflRot90 : public SymmetryTransform {
    template<typename Scalar>
    void transform_input(InputTensorX<Scalar>& input) {
      for (int row = 0; row < input.dimension(0); ++row) {
        MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
        slice.transposeInPlace();
      }
    }

    void transform_input(InputTensor& input) override {
      transform_input<InputScalar>(input);
    }

    void transform_policy(PolicyTensor& policy) override {
      PolicyMatrixSlice slice(policy.data());
      slice.transposeInPlace();
    }
  };

  struct ReflRot180 : public SymmetryTransform {
    template<typename Scalar>
    void transform_input(InputTensorX<Scalar>& input) {
      for (int row = 0; row < input.dimension(0); ++row) {
        MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
        slice.rowwise().reverseInPlace();
      }
    }

    void transform_input(InputTensor& input) override {
      transform_input<InputScalar>(input);
    }

    void transform_policy(PolicyTensor& policy) override {
      PolicyMatrixSlice slice(policy.data());
      slice.rowwise().reverseInPlace();
    }
  };

  struct ReflRot270 : public SymmetryTransform {
    template<typename Scalar>
    void transform_input(InputTensorX<Scalar>& input) {
      for (int row = 0; row < input.dimension(0); ++row) {
        MatrixSliceX<Scalar> slice(input.data() + row * kNumCells);
        slice.transposeInPlace();
        slice.rowwise().reverseInPlace();
        slice.colwise().reverseInPlace();
      }
    }

    void transform_input(InputTensor& input) override {
      transform_input<InputScalar>(input);
    }

    void transform_policy(PolicyTensor& policy) override {
      PolicyMatrixSlice slice(policy.data());
      slice.transposeInPlace();
      slice.rowwise().reverseInPlace();
      slice.colwise().reverseInPlace();
    }
  };

  SymmetryIndexSet get_symmetry_indices(const GameState&) const {
    SymmetryIndexSet set;
    set.set();
    return set;
  }

  SymmetryTransform* get_symmetry(core::symmetry_index_t index) const {
    return *(transforms().begin() + index);
  }

  void clear() {}
  void receive_state_change(const GameState& state, core::action_t action) {}

 protected:
  static transform_array_t transforms() {
    transform_array_t arr {
      &transforms_struct_.identity_,
      &transforms_struct_.rot90_,
      &transforms_struct_.rot180_,
      &transforms_struct_.rot270_,
      &transforms_struct_.refl,
      &transforms_struct_.refl_rot90_,
      &transforms_struct_.refl_rot180_,
      &transforms_struct_.refl_rot270_
    };
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

template <core::GameStateConcept GameState, eigen_util::ShapeConcept InputShape>
SquareBoardSymmetryBase<GameState, InputShape>::transforms_struct_t
    SquareBoardSymmetryBase<GameState, InputShape>::transforms_struct_;

}  // namespace common
