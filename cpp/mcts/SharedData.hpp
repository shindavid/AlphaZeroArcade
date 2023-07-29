#pragma once

#include <EigenRand/EigenRand>

#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/NodeCache.hpp>
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
  using NodeCache = mcts::NodeCache<GameState, Tensorizor>;

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  NodeCache node_cache;
  Node::asptr root_node;
  int manager_id = -1;
  move_number_t move_number = 0;
  bool search_active = false;
};

}  // namespace mcts
