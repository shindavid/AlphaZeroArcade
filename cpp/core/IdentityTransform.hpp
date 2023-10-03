#pragma once

#include <core/AbstractSymmetryTransform.hpp>

namespace core {

template <eigen_util::FixedTensorConcept Tensor>
class IdentityTransform : public AbstractSymmetryTransform<Tensor> {
 public:
  void apply(Tensor& t) override {}
  void undo(Tensor& t) override {}
};

}  // namespace core
