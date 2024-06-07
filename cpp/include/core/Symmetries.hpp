#pragma once

#include <core/BasicTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <util/MetaProgramming.hpp>
#include <util/EigenUtil.hpp>

namespace core {

template <typename FullState, eigen_util::concepts::FTensor PolicyTensor>
struct Transform {
  virtual ~Transform() {}

  virtual void apply(FullState&) = 0;
  virtual void apply(PolicyTensor&) = 0;
  virtual void undo(FullState&) = 0;
  virtual void undo(PolicyTensor&) = 0;
};

template <typename FullState, eigen_util::concepts::FTensor PolicyTensor>
struct ReflexiveTransform : public Transform<FullState, PolicyTensor> {
  virtual ~ReflexiveTransform() {}
  void undo(FullState& state) override { this->apply(state); }
  void undo(PolicyTensor& tensor) override { this->apply(tensor); }
};

template <typename FullState, eigen_util::concepts::FTensor PolicyTensor>
struct IdentityTransform : public ReflexiveTransform<FullState, PolicyTensor> {
  void apply(FullState&) override {}
  void apply(PolicyTensor&) override {}
};

template <concepts::Game Game>
class Transforms {
 public:
  using FullState = typename Game::FullState;
  using PolicyTensor = typename Game::PolicyTensor;
  using Transform = core::Transform<FullState, PolicyTensor>;
  using TransformList = typename GameState::TransformList;
  using transform_tuple_t = mp::TypeListToTuple_t<TransformList>;
  static constexpr size_t kNumTransforms = mp::Length_v<TransformList>;

  static Transform* get(core::symmetry_index_t sym);

 private:
  Transforms();
  Transforms* instance();

  static Transforms* instance_;
  Transform* transforms_[kNumTransforms];
};

}  // namespace core

#include <inline/core/Symmetries.inl>
