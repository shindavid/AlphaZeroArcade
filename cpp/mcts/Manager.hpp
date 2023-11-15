#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <vector>

#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/Node.hpp>
#include <mcts/PUCTStats.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/SearchResults.hpp>
#include <mcts/TreeTraversalThread.hpp>
#include <mcts/SharedData.hpp>

namespace mcts {

/*
 * The Manager class is the main entry point for doing MCTS searches.
 *
 * It maintains the search-tree and manages the threads and services that perform the search.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class Manager {
 public:
  using dtype = torch_util::dtype;
  using NNEvaluationService = mcts::NNEvaluationService<GameState, Tensorizor>;
  using Node = mcts::Node<GameState, Tensorizor>;
  using PUCTStats = mcts::PUCTStats<GameState, Tensorizor>;
  using SearchResults = mcts::SearchResults<GameState>;
  using TreeTraversalThread = mcts::TreeTraversalThread<GameState, Tensorizor>;
  using TreeTraversalThreadManager = mcts::TreeTraversalThreadManager<GameState, Tensorizor>;
  using SharedData = mcts::SharedData<GameState, Tensorizor>;

  using TensorizorTypes = core::TensorizorTypes<Tensorizor>;
  using GameStateTypes = core::GameStateTypes<GameState>;

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using Action = typename GameStateTypes::Action;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using InputTensor = typename TensorizorTypes::InputTensor;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;

  Manager(const ManagerParams& params);
  ~Manager();

  const ManagerParams& params() const { return params_; }
  int num_search_threads() const { return params().num_search_threads; }

  void start();
  void clear();
  void receive_state_change(core::seat_index_t, const GameState&, const Action&);
  const SearchResults* search(const Tensorizor& tensorizor, const GameState& game_state,
                              const SearchParams& params);

  void get_cache_stats(int& hits, int& misses, int& size, float& hash_balance_factor) const;

  static void end_session() { NNEvaluationService::end_session(); }

 private:
  using search_thread_vec_t = std::vector<TreeTraversalThread*>;
  void prune_counts(const SearchParams&);
  static void init_profiling_dir(const std::string& profiling_dir);

  static int next_instance_id_;  // for naming debug/profiling output files

  const ManagerParams params_;
  SharedData shared_data_;
  const SearchParams pondering_search_params_;
  TreeTraversalThreadManager* search_thread_manager_;
  NNEvaluationService* nn_eval_service_ = nullptr;

  SearchResults results_;
  bool connected_ = false;
};

}  // namespace mcts

#include <mcts/inl/Manager.inl>
