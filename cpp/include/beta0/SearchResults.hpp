#pragma once

#include "beta0/concepts/SpecConcept.hpp"
#include "core/ActionSymmetryTable.hpp"

#include <boost/json.hpp>

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct SearchResults {
  using Game = Spec::Game;
  using MoveSet = Game::Types::MoveSet;
  using ActionSymmetryTable = core::ActionSymmetryTable<Spec>;
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ValueArray = Game::Types::ValueArray;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameResultTensor = GameResultEncoding::Tensor;
  using InputEncoder = TensorEncodings::InputEncoder;
  using InputFrame = Spec::InputFrame;

  InputFrame frame;
  MoveSet valid_moves;
  PolicyTensor P;
  ValueArray Q;
  ValueArray W;  // LoTV uncertainty from the root node
  GameResultTensor R;
  PolicyTensor pre_expanded_moves;

  ActionSymmetryTable action_symmetry_table;
  bool trivial;  // all moves are symmetrically equivalent
  bool provably_lost = false;

  PolicyTensor policy_target;
  PolicyTensor counts;
  PolicyTensor AQs;    // s indicates only for the current seat
  PolicyTensor AQs_sq;
  ActionValueTensor AV;
  ActionValueTensor AU;  // action-value uncertainty, per-action per-player

  boost::json::object to_json() const;
};

}  // namespace beta0

#include "inline/beta0/SearchResults.inl"
