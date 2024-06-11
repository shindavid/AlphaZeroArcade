#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <util/MetaProgramming.hpp>
#include <util/EigenUtil.hpp>

namespace core {

template <typename StateSnapshot, eigen_util::concepts::FTensor PolicyTensor>
struct Transform {
  virtual ~Transform() {}

  virtual void apply(StateSnapshot& pos) = 0;
  virtual void apply(PolicyTensor&) = 0;
  virtual void undo(StateSnapshot& pos) = 0;
  virtual void undo(PolicyTensor&) = 0;
};

template <typename StateSnapshot, eigen_util::concepts::FTensor PolicyTensor>
struct ReflexiveTransform : public Transform<StateSnapshot, PolicyTensor> {
  virtual ~ReflexiveTransform() {}
  void undo(StateSnapshot& pos) override { this->apply(pos); }
  void undo(PolicyTensor& tensor) override { this->apply(tensor); }
};

template <typename StateSnapshot, eigen_util::concepts::FTensor PolicyTensor>
struct IdentityTransform : public ReflexiveTransform<StateSnapshot, PolicyTensor> {
  void apply(StateSnapshot&) override {}
  void apply(PolicyTensor&) override {}
};

template <concepts::Game Game>
class Transforms {
 public:
  using StateSnapshot = typename Game::StateSnapshot;
  using PolicyTensor = typename Game::PolicyTensor;
  using Transform = core::Transform<StateSnapshot, PolicyTensor>;
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
