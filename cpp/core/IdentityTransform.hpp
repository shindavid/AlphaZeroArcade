#pragma once

#include <core/AbstractSymmetryTransform.hpp>

namespace core {

template<typename GameState, typename Tensorizor>
class IdentityTransform : public AbstractSymmetryTransform<GameState, Tensorizor> {
public:
  using base_t = AbstractSymmetryTransform<GameState, Tensorizor>;
  using InputTensor = typename base_t::InputTensor;
  using PolicyTensor = typename base_t::PolicyTensor;

  template<typename InputTensorT> void transform_input(InputTensorT&) {}  // unit tests need general scalar type
  void transform_input(InputTensor&) override {}
  void transform_policy(PolicyTensor&) override {}
};

}
