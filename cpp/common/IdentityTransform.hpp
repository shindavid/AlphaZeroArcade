#pragma once

#include <common/AbstractSymmetryTransform.hpp>

namespace common {

template<typename GameState, typename Tensorizor>
class IdentityTransform : public AbstractSymmetryTransform<GameState, Tensorizor> {
public:
  using base_t = AbstractSymmetryTransform<GameState, Tensorizor>;
  using InputEigenTensor = typename base_t::InputEigenTensor;
  using PolicyEigenSlab = typename base_t::PolicyEigenSlab;

  void transform_input(InputEigenTensor& input) override {}
  void transform_policy(PolicyEigenSlab& policy) override {}
};

}
