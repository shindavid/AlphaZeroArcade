#pragma once

#include <common/AbstractSymmetryTransform.hpp>

namespace common {

template<typename GameState, typename Tensorizor>
class IdentityTransform : public AbstractSymmetryTransform<GameState, Tensorizor> {
public:
  using base_t = AbstractSymmetryTransform<GameState, Tensorizor>;
  using InputTensor = base_t::InputTensor;
  using PolicyTensor = base_t::PolicyTensor;

  void transform_input(InputTensor& input) override {}
  void transform_policy(PolicyTensor& policy) override {}
};

}
