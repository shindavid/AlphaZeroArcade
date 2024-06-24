#pragma once

#include <core/BasicTypes.hpp>
#include <util/EigenUtil.hpp>

namespace core {

template <typename _BaseState, eigen_util::concepts::FTensor _PolicyTensor>
struct Transform {
  using BaseState = _BaseState;
  using PolicyTensor = _PolicyTensor;

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

}  // namespace core
