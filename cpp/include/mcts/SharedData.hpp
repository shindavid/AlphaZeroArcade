#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/Node.hpp>
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
  using BaseState = Game::BaseState;
  using FullState = Game::FullState;
  using LookupTable = Node::LookupTable;

  using base_state_vec_t = std::vector<BaseState>;
  using node_pool_index_t = Node::node_pool_index_t;

  void clear() {
    root_softmax_temperature.reset();
    lookup_table.clear();
    root_node_index = -1;

    Game::Rules::init_state(root_state);
    root_state_history.clear();
    util::stuff_back<Game::Constants::kHistorySize>(root_state_history, root_state);
  }

  void update_state(const FullState& state) {
    root_state = state;
    util::stuff_back<Game::Constants::kHistorySize>(root_state_history, state);
  }

  Node* get_root_node() { return lookup_table.get_node(root_node_index); }

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  std::mutex init_root_mutex;
  std::mutex search_mutex;
  std::condition_variable cv_search_on, cv_search_off;
  boost::dynamic_bitset<> active_search_threads;
  LookupTable lookup_table;
  FullState root_state;
  base_state_vec_t root_state_history;

  node_pool_index_t root_node_index = -1;
  SearchParams search_params;
  int manager_id = -1;
  bool shutting_down = false;
};

}  // namespace mcts
