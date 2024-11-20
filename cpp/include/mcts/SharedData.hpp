#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/ReachableSet.hpp>
#include <mcts/SearchParams.hpp>
#include <util/EigenUtil.hpp>
#include <util/Math.hpp>

#include <boost/dynamic_bitset.hpp>
#include <EigenRand/EigenRand>

#include <barrier>
#include <condition_variable>
#include <functional>
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
  using Rules = Game::Rules;
  using ManagerParams = mcts::ManagerParams<Game>;
  using Node = mcts::Node<Game>;
  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using LookupTable = Node::LookupTable;
  using SymmetryGroup = Game::SymmetryGroup;
  using ReachableSet = mcts::ReachableSet<Game>;

  using node_pool_index_t = Node::node_pool_index_t;

  using StateHistoryArray = std::array<StateHistory, SymmetryGroup::kOrder>;

  struct root_info_t {
    StateHistoryArray history_array;

    group::element_t canonical_sym = -1;
    node_pool_index_t node_index = -1;
  };

  SharedData(const ManagerParams& manager_params, int mgr_id);
  SharedData(const SharedData&) = delete;
  SharedData& operator=(const SharedData&) = delete;

  /*
   * Called by each SearchThread. Signals to each SearchThread to break out of its search-loop.
   *
   * Returns when all SearchThread's have reached the break-point.
   */
  void break_search_threads();

  void reset_reachable_set();
  void add_to_reachable_set(const std::vector<node_pool_index_t>& indices);
  bool more_visits_needed();
  void clear();
  void update_state(core::action_t action);
  void init_root_info(bool add_noise);
  Node* get_root_node() { return lookup_table.get_node(root_info.node_index); }

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  std::mutex init_root_mutex;
  std::mutex search_mutex;
  std::condition_variable cv_search_on;
  std::condition_variable cv_search_off;
  std::condition_variable cv_search_thread_break;
  std::barrier<std::function<void()>> search_barrier;

  boost::dynamic_bitset<> active_search_threads;
  LookupTable lookup_table;
  root_info_t root_info;
  ReachableSet reachable_set;

  SearchParams search_params;
  const int manager_id = -1;
  bool search_threads_broken = false;
  bool shutting_down = false;
};

}  // namespace mcts

#include <inline/mcts/SharedData.inl>
