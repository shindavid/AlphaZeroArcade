#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

#include <common/AbstractSymmetryTransform.hpp>
#include <common/DerivedTypes.hpp>
#include <common/IdentityTransform.hpp>
#include <common/TensorizorConcept.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>
#include <util/CppUtil.hpp>

namespace c4 {

class Tensorizor {
public:
  using Shape = util::int_sequence<kNumPlayers, kNumColumns, kNumRows>;
  using PolicyVector = common::GameStateTypes<GameState>::PolicyVector;
  using InputTensor = common::TensorizorTypes<Tensorizor>::InputTensor;
  using SymmetryTransform = common::AbstractSymmetryTransform<GameState, Tensorizor>;
  using IdentityTransform = common::IdentityTransform<GameState, Tensorizor>;

  class ReflectionTransform : public SymmetryTransform {
  public:
    void transform_input(InputTensor& input) override;
    void transform_policy(PolicyVector& policy) override;
  };

  Tensorizor();
  void clear() {}
  void receive_state_change(const GameState& state, common::action_index_t action_index) {}

  void tensorize(int slice, InputTensor& tensor, const GameState& state) { state.tensorize(slice, tensor); }

  SymmetryTransform* get_random_symmetry(const GameState&) const;

private:
  IdentityTransform identity_transform_;
  ReflectionTransform reflection_transform_;
  std::array<SymmetryTransform*, 2> transforms_;
};

}  // namespace c4

static_assert(common::TensorizorConcept<c4::Tensorizor, c4::GameState>);

#include <connect4/inl/C4Tensorizor.inl>
