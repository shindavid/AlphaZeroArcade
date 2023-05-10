#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

#include <common/AbstractSymmetryTransform.hpp>
#include <common/DerivedTypes.hpp>
#include <common/IdentityTransform.hpp>
#include <common/TensorizorConcept.hpp>
#include <othello/Constants.hpp>
#include <othello/GameState.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

namespace othello {

class Tensorizor {
public:
  static constexpr int kMaxNumSymmetries = 8;
  using InputShape = eigen_util::Shape<kNumPlayers, kBoardDimension, kBoardDimension>;
  using GameStateTypes = common::GameStateTypes<GameState>;
  using TensorizorTypes = common::TensorizorTypes<Tensorizor>;
  using SymmetryIndexSet = TensorizorTypes::SymmetryIndexSet;
  using InputEigenTensor = TensorizorTypes::InputTensor::EigenType;
  using PolicyEigenTensor = GameStateTypes::PolicyTensor::EigenType;
  using SymmetryTransform = common::AbstractSymmetryTransform<GameState, Tensorizor>;
  using IdentityTransform = common::IdentityTransform<GameState, Tensorizor>;
  using transform_array_t = std::array<SymmetryTransform*, kMaxNumSymmetries>;

  using MatrixT = Eigen::Matrix<torch_util::dtype, kBoardDimension, kBoardDimension, Eigen::RowMajor>;

  static MatrixT& slice_as_matrix(InputEigenTensor& input, int row);
  static MatrixT& as_matrix(PolicyEigenTensor& policy);

  struct Rotation90Transform : public SymmetryTransform {
    void transform_input(InputEigenTensor& input) override;
    void transform_policy(PolicyEigenTensor& policy) override;
  };

  struct Rotation180Transform : public SymmetryTransform {
    void transform_input(InputEigenTensor& input) override;
    void transform_policy(PolicyEigenTensor& policy) override;
  };

  struct Rotation270Transform : public SymmetryTransform {
    void transform_input(InputEigenTensor& input) override;
    void transform_policy(PolicyEigenTensor& policy) override;
  };

  struct ReflectionOverHorizontalTransform : public SymmetryTransform {
    void transform_input(InputEigenTensor& input) override;
    void transform_policy(PolicyEigenTensor& policy) override;
  };

  struct ReflectionOverHorizontalWithRotation90Transform : public SymmetryTransform {
    void transform_input(InputEigenTensor& input) override;
    void transform_policy(PolicyEigenTensor& policy) override;
  };

  struct ReflectionOverHorizontalWithRotation180Transform : public SymmetryTransform {
    void transform_input(InputEigenTensor& input) override;
    void transform_policy(PolicyEigenTensor& policy) override;
  };

  struct ReflectionOverHorizontalWithRotation270Transform : public SymmetryTransform {
    void transform_input(InputEigenTensor& input) override;
    void transform_policy(PolicyEigenTensor& policy) override;
  };

  void clear() {}
  void receive_state_change(const GameState& state, common::action_index_t action_index) {}
  void tensorize(InputEigenTensor& tensor, const GameState& state) const { state.tensorize(tensor); }

  SymmetryIndexSet get_symmetry_indices(const GameState&) const;
  SymmetryTransform* get_symmetry(common::symmetry_index_t index) const;

private:
  static transform_array_t transforms();

  struct transforms_struct_t {
    IdentityTransform identity_transform_;
    Rotation90Transform rotation90_transform_;
    Rotation180Transform rotation180_transform_;
    Rotation270Transform rotation270_transform_;
    ReflectionOverHorizontalTransform reflection_over_horizontal_transform_;
    ReflectionOverHorizontalWithRotation90Transform reflection_over_horizontal_with_rotation90_transform_;
    ReflectionOverHorizontalWithRotation180Transform reflection_over_horizontal_with_rotation180_transform_;
    ReflectionOverHorizontalWithRotation270Transform reflection_over_horizontal_with_rotation270_transform_;
  };

  static transforms_struct_t transforms_struct_;
};

}  // namespace othello

static_assert(common::TensorizorConcept<othello::Tensorizor, othello::GameState>);

#include <othello/inl/Tensorizor.inl>
