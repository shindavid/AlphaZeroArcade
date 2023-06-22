#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

#include <core/AbstractSymmetryTransform.hpp>
#include <core/DerivedTypes.hpp>
#include <core/IdentityTransform.hpp>
#include <core/TensorizorConcept.hpp>
#include <connect4/Constants.hpp>
#include <connect4/GameState.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

namespace c4 {

class Tensorizor {
public:
  static constexpr int kMaxNumSymmetries = 2;
  using InputShape = eigen_util::Shape<kNumPlayers, kNumColumns, kNumRows>;
  using InputTensor = Eigen::TensorFixedSize<bool, InputShape, Eigen::RowMajor>;

  using GameStateTypes = common::GameStateTypes<GameState>;
  using TensorizorTypes = common::TensorizorTypes<Tensorizor>;
  using SymmetryIndexSet = TensorizorTypes::SymmetryIndexSet;
  using PolicyTensor = GameStateTypes::PolicyTensor;
  using SymmetryTransform = common::AbstractSymmetryTransform<GameState, Tensorizor>;
  using IdentityTransform = common::IdentityTransform<GameState, Tensorizor>;
  using transform_array_t = std::array<SymmetryTransform*, 2>;

  class ReflectionTransform : public SymmetryTransform {
  public:
    void transform_input(InputTensor& input) override;
    void transform_policy(PolicyTensor& policy) override;
  };

  void clear() {}
  void receive_state_change(const GameState& state, common::action_index_t action_index) {}
  void tensorize(InputTensor& tensor, const GameState& state) const { state.tensorize(tensor); }

  SymmetryIndexSet get_symmetry_indices(const GameState&) const;
  SymmetryTransform* get_symmetry(common::symmetry_index_t index) const;

private:
  static transform_array_t transforms();

  static IdentityTransform identity_transform_;
  static ReflectionTransform reflection_transform_;
};

}  // namespace c4

static_assert(common::TensorizorConcept<c4::Tensorizor, c4::GameState>);

#include <connect4/inl/Tensorizor.inl>
