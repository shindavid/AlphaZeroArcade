#pragma once

#include <core/AbstractSymmetryTransform.hpp>

namespace core {

template <typename InputTensor, typename PolicyTensor>
class IdentityTransform : public AbstractSymmetryTransform<InputTensor, PolicyTensor> {
 public:
  using base_t = AbstractSymmetryTransform<InputTensor, PolicyTensor>;

  template<typename InputTensorT> void transform_input(InputTensorT&) {}  // unit tests need general scalar type
  void transform_input(InputTensor&) override {}
  void transform_policy(PolicyTensor&) override {}
};
}
