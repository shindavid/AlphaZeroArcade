#pragma once

#include <common/AbstractSymmetryTransform.hpp>

namespace common {

template<typename GameState, typename Tensorizor>
class IdentityTransform : public AbstractSymmetryTransform<GameState, Tensorizor> {
public:
  using base_t = AbstractSymmetryTransform<GameState, Tensorizor>;
  using InputEigenTensor = typename base_t::InputEigenTensor;
  using PolicyEigenVector = typename base_t::PolicyEigenVector;

  void transform_input(InputEigenTensor& input) override {}
  void transform_policy(PolicyEigenVector& policy) override {}
};

}
