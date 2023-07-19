#pragma once

#include <EigenRand/EigenRand>

#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <util/EigenUtil.hpp>
#include <util/Math.hpp>

namespace mcts {

/*
 * SharedData is owned by the Manager and shared by other threads/services.
 *
 * It is separated from Manager to avoid circular dependencies.
 */
template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
struct SharedData {
  using Node = mcts::Node<GameState, Tensorizor>;

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  Node* root_node = nullptr;
  int manager_id = -1;
  bool search_active = false;
};

}  // namespace mcts
