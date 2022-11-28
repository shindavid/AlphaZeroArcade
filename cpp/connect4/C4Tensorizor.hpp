#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

#include <common/AbstractSymmetryTransform.hpp>
#include <common/IdentityTransform.hpp>
#include <common/TensorizorConcept.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>
#include <util/CppUtil.hpp>

namespace c4 {

class ReflectionTransform : public common::AbstractSymmetryTransform {
public:
  void transform_input(torch::Tensor input) override;
  void transform_policy(torch::Tensor policy) override;
};

class Tensorizor {
public:
  using Shape = util::int_sequence<kNumPlayers, kNumColumns, kNumRows>;

  Tensorizor();
  void clear() {}
  void receive_state_change(const GameState& state, common::action_index_t action_index) {}

  void tensorize(torch::Tensor tensor, const GameState& state) { state.tensorize(tensor); }

  common::AbstractSymmetryTransform* get_random_symmetry(const GameState&) const;

private:
  common::IdentityTransform identity_transform_;
  ReflectionTransform reflection_transform_;
  std::array<common::AbstractSymmetryTransform*, 2> transforms_;
};

}  // namespace c4

static_assert(common::TensorizorConcept<c4::Tensorizor, c4::GameState>);

#include <connect4/inl/C4Tensorizor.inl>
