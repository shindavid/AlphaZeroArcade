#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/NodeCache.hpp>
#include <mcts/SearchParams.hpp>
#include <util/EigenUtil.hpp>
#include <util/Math.hpp>

#include <boost/dynamic_bitset.hpp>
#include <EigenRand/EigenRand>

#include <condition_variable>
#include <mutex>
#include <vector>

namespace mcts {

/*
 * SharedData is owned by the Manager and shared by other threads/services.
 *
 * It is separated from Manager to avoid circular dependencies.
 */
template <core::concepts::Game Game>
struct SharedData {
  using Node = mcts::Node<Game>;
  using NodeCache = mcts::NodeCache<Game>;
  using BaseState = typename Game::BaseState;
  using FullState = typename Game::FullState;

  using base_state_vec_t = std::vector<BaseState>;

  void clear() {
    move_number = 0;
    root_softmax_temperature.reset();
    node_cache.clear();
    root_node = nullptr;

    root_state = FullState();
    root_state_history.clear();
    util::stuff_back<Game::kHistorySize>(root_state_history, root_state);
  }

  void update_state(const FullState& state) {
    root_state = state;
    util::stuff_back<Game::kHistorySize>(root_state_history, state);
  }

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  std::mutex search_mutex;
  std::condition_variable cv_search_on, cv_search_off;
  boost::dynamic_bitset<> active_search_threads;
  NodeCache node_cache;
  FullState root_state;
  base_state_vec_t root_state_history;

  Node::sptr root_node;
  SearchParams search_params;
  int manager_id = -1;
  move_number_t move_number = 0;
  bool shutting_down = false;
};

}  // namespace mcts
