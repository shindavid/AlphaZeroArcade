#pragma once

#include <core/concepts/Game.hpp>
#include <core/Symmetries.hpp>
#include <util/MetaProgramming.hpp>

namespace core {

template <concepts::Game Game>
class Transforms {
 public:
  using BaseState = Game::BaseState;
  using PolicyTensor = Game::Types::PolicyTensor;
  using Transform = Game::Types::Transform;
  using TransformList = Game::TransformList;
  static_assert(mp::Length_v<TransformList> == Game::Constants::kNumSymmetries);

  static Transform* get(core::symmetry_index_t sym);

 private:
  Transforms();
  static Transforms* instance();

  static Transforms* instance_;
  Transform* transforms_[Game::Constants::kNumSymmetries];
};

}  // namespace core

#include <inline/core/Transforms.inl>
