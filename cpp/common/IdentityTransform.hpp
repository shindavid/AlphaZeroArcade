#pragma once

#include <common/AbstractSymmetryTransform.hpp>

namespace common {

template<typename GameState, typename Tensorizor>
class IdentityTransform : public AbstractSymmetryTransform<GameState, Tensorizor> {
public:
  using base_t = AbstractSymmetryTransform<GameState, Tensorizor>;
  using InputEigenTensor = typename base_t::InputEigenTensor;
  using PolicyEigenTensor = typename base_t::PolicyEigenTensor;

  void transform_input(InputEigenTensor& input) override {}
  void transform_policy(PolicyEigenTensor& policy) override {}
};

}
