#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/NNEvaluationServiceBase.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/SearchThread.hpp>
#include <mcts/SharedData.hpp>
#include <mcts/TypeDefs.hpp>

namespace mcts {

/*
 * The Manager class is the main entry point for doing MCTS searches.
 *
 * It maintains the search-tree and manages the threads and services that perform the search.
 */
template <core::concepts::Game Game>
class Manager {
 public:
  using ManagerParams = mcts::ManagerParams<Game>;
  using NNEvaluationService = mcts::NNEvaluationService<Game>;
  using NNEvaluationServiceBase = mcts::NNEvaluationServiceBase<Game>;
  using Node = mcts::Node<Game>;
  using LocalPolicyArray = Node::LocalPolicyArray;
  using SearchThread = mcts::SearchThread<Game>;
  using SharedData = mcts::SharedData<Game>;
  using node_pool_index_t = Node::node_pool_index_t;
  using Edge = Node::Edge;
  using ActionSymmetryTable = Game::Types::ActionSymmetryTable;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using SearchResults = Game::Types::SearchResults;

  /*
   * Construct a Manager object.
   *
   * Can optionally pass in an NNEvaluationService object. This is useful to pass in a mock service
   * for testing.
   *
   * Can optionally pass a mutex_cv_pool to be used by the nodes. If not provided, the Manager will
   * create a separate single-element mutex-pool.
   */
  Manager(const ManagerParams& params, NNEvaluationServiceBase* service=nullptr);
  Manager(mutex_cv_vec_sptr_t& mutex_cv_pool, const ManagerParams& params,
          NNEvaluationServiceBase* service = nullptr);

  ~Manager();

  const ManagerParams& params() const { return params_; }
  int num_search_threads() const { return params().num_search_threads; }

  void start_threads();
  void start();
  void clear();
  void receive_state_change(core::seat_index_t, const State&, core::action_t);
  const SearchResults* search(const SearchParams& params);
  void load_root_action_values(ActionValueTensor& action_values);

  void start_search_threads(const SearchParams& search_params);
  void wait_for_search_threads();
  void stop_search_threads();

  void end_session() {
    nn_eval_service_->end_session();
  }

  SharedData* shared_data() { return &shared_data_; }
  void set_post_visit_func(std::function<void()> func);

 private:
  using search_thread_vec_t = std::vector<SearchThread*>;

  Manager(bool dummy, mutex_cv_vec_sptr_t mutex_cv_pool,
          const ManagerParams& params, NNEvaluationServiceBase* service);

  void announce_shutdown();
  void load_action_symmetries(Node* root, core::action_t* actions);
  void prune_policy_target(const SearchParams&, group::element_t inv_sym);
  static void init_profiling_dir(const std::string& profiling_dir);

  static int next_instance_id_;  // for naming debug/profiling output files

  const ManagerParams params_;
  const SearchParams pondering_search_params_;
  SharedData shared_data_;
  search_thread_vec_t search_threads_;
  NNEvaluationServiceBase* nn_eval_service_ = nullptr;

  SearchResults results_;

  bool connected_ = false;
};

}  // namespace mcts

#include <inline/mcts/Manager.inl>
