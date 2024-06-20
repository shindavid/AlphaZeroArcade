#pragma once

#include <core/concepts/Game.hpp>
#include <core/Symmetries.hpp>
#include <util/MetaProgramming.hpp>

namespace core {

template <concepts::Game Game>
class Transforms {
 public:
  using BaseState = typename Game::BaseState;
  using PolicyTensor = typename Game::Types::PolicyTensor;
  using Transform = core::Transform<BaseState, PolicyTensor>;
  using TransformList = typename Game::TransformList;
  using transform_tuple_t = mp::TypeListToTuple_t<TransformList>;
  static constexpr size_t kNumTransforms = mp::Length_v<TransformList>;

  static Transform* get(core::symmetry_index_t sym);

 private:
  Transforms();
  static Transforms* instance();

  static Transforms* instance_;
  Transform* transforms_[kNumTransforms];
};

}  // namespace core

#include <inline/core/Transforms.inl>
