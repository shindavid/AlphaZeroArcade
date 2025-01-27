#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/TypeDefs.hpp>
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
  using Rules = Game::Rules;
  using ManagerParams = mcts::ManagerParams<Game>;
  using Node = mcts::Node<Game>;
  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using LookupTable = Node::LookupTable;
  using SymmetryGroup = Game::SymmetryGroup;

  using node_pool_index_t = Node::node_pool_index_t;

  using StateHistoryArray = std::array<StateHistory, SymmetryGroup::kOrder>;

  struct RootInfo {
    StateHistoryArray history_array;

    group::element_t canonical_sym = -1;
    node_pool_index_t node_index = -1;
    core::seat_index_t active_seat = -1;
  };

  SharedData(mutex_cv_vec_sptr_t mutex_pool, const ManagerParams& manager_params, int mgr_id);
  SharedData(const SharedData&) = delete;
  SharedData& operator=(const SharedData&) = delete;

  void clear();
  void update_state(core::action_t action);
  void init_root_info(bool add_noise);
  core::action_mode_t get_current_action_mode() const;
  Node* get_root_node() { return lookup_table.get_node(root_info.node_index); }

  eigen_util::UniformDirichletGen<float> dirichlet_gen;
  math::ExponentialDecay root_softmax_temperature;
  Eigen::Rand::P8_mt19937_64 rng;

  std::mutex init_root_mutex;
  std::mutex search_mutex;
  std::condition_variable cv_search_on, cv_search_off;
  boost::dynamic_bitset<> active_search_threads;
  LookupTable lookup_table;
  RootInfo root_info;

  SearchParams search_params;
  const int manager_id = -1;
  bool shutting_down = false;
};

}  // namespace mcts

#include <inline/mcts/SharedData.inl>
