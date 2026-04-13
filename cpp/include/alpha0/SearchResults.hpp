#pragma once

#include "core/ActionSymmetryTable.hpp"
#include "core/BasicTypes.hpp"
#include "alpha0/concepts/SpecConcept.hpp"

#include <boost/json.hpp>

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
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
  GameResultTensor R;
  PolicyTensor pre_expanded_moves;

  ActionSymmetryTable action_symmetry_table;
  bool trivial;  // all moves are symmetrically equivalent
  bool provably_lost = false;

  PolicyTensor policy_target;
  PolicyTensor counts;
  PolicyTensor AQs;  // s indicates only for the current seat
  PolicyTensor AQs_sq;
  ActionValueTensor AV;

  boost::json::object to_json() const;
};

}  // namespace alpha0

#include "inline/alpha0/SearchResults.inl"
