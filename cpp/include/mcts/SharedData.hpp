#pragma once

#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/NodeCache.hpp>
#include <mcts/SearchParams.hpp>
#include <util/EigenUtil.hpp>
#include <util/Math.hpp>

#include <boost/dynamic_bitset.hpp>
#include <EigenRand/EigenRand>

#include <condition_variable>
#include <mutex>

namespace mcts {

/*
 * SharedData is owned by the Manager and shared by other threads/services.
 *
 * It is separated from Manager to avoid circular dependencies.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
struct SharedData {
  using Node = mcts::Node<GameState, Tensorizor>;
  using NodeCache = mcts::NodeCache<GameState, Tensorizor>;

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  std::mutex search_mutex;
  std::condition_variable cv_search_on, cv_search_off;
  boost::dynamic_bitset<> active_search_threads;
  NodeCache node_cache;
  GameState root_state;
  Tensorizor root_tensorizor;
  Node::sptr root_node;
  SearchParams search_params;
  int manager_id = -1;
  move_number_t move_number = 0;
  bool shutting_down = false;
};

}  // namespace mcts
