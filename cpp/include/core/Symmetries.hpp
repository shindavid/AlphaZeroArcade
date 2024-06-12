#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <util/MetaProgramming.hpp>
#include <util/EigenUtil.hpp>

namespace core {

template <typename BaseState, eigen_util::concepts::FTensor PolicyTensor>
struct Transform {
  virtual ~Transform() {}

  virtual void apply(BaseState& pos) = 0;
  virtual void apply(PolicyTensor&) = 0;
  virtual void undo(BaseState& pos) = 0;
  virtual void undo(PolicyTensor&) = 0;
};

template <typename BaseState, eigen_util::concepts::FTensor PolicyTensor>
struct ReflexiveTransform : public Transform<BaseState, PolicyTensor> {
  virtual ~ReflexiveTransform() {}
  void undo(BaseState& pos) override { this->apply(pos); }
  void undo(PolicyTensor& tensor) override { this->apply(tensor); }
};

template <typename BaseState, eigen_util::concepts::FTensor PolicyTensor>
struct IdentityTransform : public ReflexiveTransform<BaseState, PolicyTensor> {
  void apply(BaseState&) override {}
  void apply(PolicyTensor&) override {}
};

template <concepts::Game Game>
class Transforms {
 public:
  using BaseState = typename Game::BaseState;
  using PolicyTensor = typename Game::PolicyTensor;
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

#include <inline/core/Symmetries.inl>
