#pragma once

#include "search/concepts/ManagerParamsConcept.hpp"
#include "util/EigenUtil.hpp"
#include "util/Math.hpp"

#include <EigenRand/EigenRand>

namespace alpha0 {

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

}  // namespace alpha0
