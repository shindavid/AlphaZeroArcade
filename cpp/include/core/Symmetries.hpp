#pragma once

#include <core/BasicTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <util/MetaProgramming.hpp>

namespace core {

template <typename Data, typename PolicyTensor>
struct Transform {
  virtual ~Transform() {}

  virtual void apply(Data&) = 0;
  virtual void apply(PolicyTensor&) = 0;
  virtual void undo(Data&) = 0;
  virtual void undo(PolicyTensor&) = 0;
};

template <typename Data, typename PolicyTensor>
struct ReflexiveTransform : public Transform<Data, PolicyTensor> {
  virtual ~ReflexiveTransform() {}
  void undo(Data& data) override { this->apply(data); }
  void undo(PolicyTensor& tensor) override { this->apply(tensor); }
};

template <typename Data, typename PolicyTensor>
struct IdentityTransform : public ReflexiveTransform<Data, PolicyTensor> {
  void apply(Data&) override {}
  void apply(PolicyTensor&) override {}
};

template <GameStateConcept GameState>
class Transforms {
 public:
  using Data = typename GameState::Data;
  using PolicyTensor = typename GameState::PolicyTensor;
  using Transform = core::Transform<Data, PolicyTensor>;
  using TransformList = typename GameState::TransformList;
  using transform_tuple_t = mp::TypeListToTuple_t<TransformList>;

  static Transform* get(core::symmetry_index_t sym);

 private:
  static transform_tuple_t transforms_;
};

}  // namespace core

#include <inline/core/Symmetries.inl>
