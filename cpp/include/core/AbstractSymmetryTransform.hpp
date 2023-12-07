#pragma once

#include <util/EigenUtil.hpp>

namespace core {

/*
 * SymmetryTransform's provide a mechanism to effectively apply a symmetry transform to a GameState
 * or to an associated tensor target (such as a policy). In typical square board games like go,
 * there are 8 transforms: 4 rotations and 4 reflections. See AlphaGo papers for more details.
 */
template<eigen_util::FixedTensorConcept Tensor_>
class AbstractSymmetryTransform {
public:
  using Tensor = Tensor_;

  virtual ~AbstractSymmetryTransform() {}
  virtual void apply(Tensor& t) = 0;
  virtual void undo(Tensor& t) = 0;
};

}  // namespace core
