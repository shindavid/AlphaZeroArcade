#pragma once

#include "search/concepts/ManagerParamsConcept.hpp"
#include "util/EigenUtil.hpp"
#include "util/Math.hpp"

#include <EigenRand/EigenRand>

namespace beta0 {

// For now, beta0::AuxState is identical to alpha0::AuxState. Later we will specialize it. We
// expect that beta0 shouldn't need Dirichlet noise, for instance.
template <search::concepts::ManagerParams ManagerParams>
struct AuxState {
  AuxState(const ManagerParams& params)
      : root_softmax_temperature(params.starting_root_softmax_temperature,
                                 params.ending_root_softmax_temperature,
                                 params.root_softmax_temperature_half_life) {}

  void clear() { root_softmax_temperature.reset(); }
  void step() { root_softmax_temperature.step(); }

  mutable eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  mutable Eigen::Rand::P8_mt19937_64 rng;
};

}  // namespace beta0
