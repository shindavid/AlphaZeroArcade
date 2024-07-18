#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/SearchThread.hpp>
#include <mcts/SharedData.hpp>

namespace mcts {

/*
 * The Manager class is the main entry point for doing MCTS searches.
 *
 * It maintains the search-tree and manages the threads and services that perform the search.
 */
template <core::concepts::Game Game>
class Manager {
 public:
  using NNEvaluationService = mcts::NNEvaluationService<Game>;
  using Node = mcts::Node<Game>;
  using SearchThread = mcts::SearchThread<Game>;
  using SharedData = mcts::SharedData<Game>;
  using node_pool_index_t = Node::node_pool_index_t;
  using edge_t = Node::edge_t;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using FullState = Game::FullState;
  using SearchResults = Game::Types::SearchResults;

  Manager(const ManagerParams& params);
  ~Manager();

  const ManagerParams& params() const { return params_; }
  int num_search_threads() const { return params().num_search_threads; }

  void start();
  void clear();
  void receive_state_change(core::seat_index_t, const FullState&, core::action_t);
  const SearchResults* search(const FullState& state, const SearchParams& params);

  void start_search_threads(const SearchParams& search_params);
  void wait_for_search_threads();
  void stop_search_threads();

  void end_session() {
    if (nn_eval_service_) nn_eval_service_->end_session();
  }

  void set_player_data(void* player_data) { player_data_ = player_data; }
  void* get_player_data() const { return player_data_; }

 private:
  using search_thread_vec_t = std::vector<SearchThread*>;

  void announce_shutdown();
  void load_action_symmetries(Node* root, core::action_t* actions);
  void prune_policy_target(const SearchParams&);
  static void init_profiling_dir(const std::string& profiling_dir);

  static int next_instance_id_;  // for naming debug/profiling output files

  const ManagerParams params_;
  const SearchParams pondering_search_params_;
  SharedData shared_data_;
  search_thread_vec_t search_threads_;
  NNEvaluationService* nn_eval_service_ = nullptr;

  SearchResults results_;

  void* player_data_ = nullptr;
  bool connected_ = false;
};

}  // namespace mcts

#include <inline/mcts/Manager.inl>
