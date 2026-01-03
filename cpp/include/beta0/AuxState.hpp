#pragma once

#include "search/concepts/ManagerParamsConcept.hpp"
#include "util/EigenUtil.hpp"

#include <EigenRand/EigenRand>

namespace beta0 {

template <search::concepts::ManagerParams ManagerParams>
struct AuxState {
  AuxState(const ManagerParams& params) {}

  void clear() {}
  void step() {}

  mutable eigen_util::UniformDirichletGen<float> dirichlet_gen;
  mutable Eigen::Rand::P8_mt19937_64 rng;
};

}  // namespace beta0
