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
  using SymmetryGroup = Game::SymmetryGroup;

  using base_state_vec_t = std::vector<BaseState>;
  using node_pool_index_t = Node::node_pool_index_t;

  using FullStateArray = std::array<FullState, SymmetryGroup::kOrder>;
  using base_state_vec_array_t = std::array<base_state_vec_t, SymmetryGroup::kOrder>;

  struct root_info_t {
    FullStateArray state;
    base_state_vec_array_t state_history;

    group::element_t canonical_sym = -1;
    node_pool_index_t node_index = -1;
  };

  SharedData(bool multithreaded_mode) : lookup_table(multithreaded_mode) {}
  SharedData(const SharedData&) = delete;
  SharedData& operator=(const SharedData&) = delete;

  void clear();
  void update_state(core::action_t action);
  Node* get_root_node() { return lookup_table.get_node(root_info.node_index); }

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  std::mutex init_root_mutex;
  std::mutex search_mutex;
  std::condition_variable cv_search_on, cv_search_off;
  boost::dynamic_bitset<> active_search_threads;
  LookupTable lookup_table;
  root_info_t root_info;

  SearchParams search_params;
  int manager_id = -1;
  bool shutting_down = false;
};

}  // namespace mcts

#include <inline/mcts/SharedData.inl>
