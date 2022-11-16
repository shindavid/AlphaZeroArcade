#pragma once

#include <common/AbstractSymmetryTransform.hpp>

namespace common {

class IdentityTransform : public AbstractSymmetryTransform {
public:
  void transform_input(torch::Tensor input) override {}
  void transform_policy(torch::Tensor input) override {}
};

}
