#pragma once

#include <bitset>
#include <mutex>
#include <thread>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/Node.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/SharedData.hpp>
#include <mcts/TypeDefs.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class SearchThread {
public:
  using GameStateTypes = core::GameStateTypes<GameState>;
  using NNEvaluation = mcts::NNEvaluation<GameState>;
  using NNEvaluationService = mcts::NNEvaluationService<GameState, Tensorizor>;
  using Node = mcts::Node<GameState, Tensorizor>;
  using PUCTStats = mcts::PUCTStats<GameState, Tensorizor>;

  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using NNEvaluation_sptr = typename NNEvaluation::sptr;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  static constexpr int kNumPlayers = GameState::kNumPlayers;

  using dtype = torch_util::dtype;
  using player_bitset_t = std::bitset<kNumPlayers>;
  using profiler_t = search_thread_profiler_t;

  SearchThread(SharedData* shared_data, NNEvaluationService* nn_eval_service, const ManagerParams* manager_params,
               int thread_id);
  ~SearchThread();

  int thread_id() const { return thread_id_; }

  void join();
  void kill();
  void launch(const SearchParams* search_params, std::function<void()> f);
  bool needs_more_visits(Node* root, int tree_size_limit);
  void visit(Node* tree, int depth);
  bool is_pondering() const { return search_params_->ponder; }

  void dump_profiling_stats() { profiler_.dump(64); }

private:
  struct evaluate_and_expand_result_t {
    NNEvaluation_sptr evaluation;
    bool backpropagated_virtual_loss;
  };

  void add_dirichlet_noise(LocalPolicyArray& P);
  void backprop_outcome(Node* tree, const ValueArray& outcome);
  evaluate_and_expand_result_t evaluate_and_expand(Node* tree);
  void evaluate_and_expand_unset(
      Node* tree, std::unique_lock<std::mutex>* lock, evaluate_and_expand_result_t* data);

  /*
   * Used in visit().
   *
   * Applies PUCT criterion to select the best child-index to visit from the given Node.
   *
   * TODO: as we experiment with things like auxiliary NN output heads, dynamic cPUCT values, etc., this method will
   * evolve. It probably makes sense to have the behavior as part of the Tensorizor, since there is coupling with NN
   * architecture (in the form of output heads).
   */
  mcts::child_index_t get_best_child_index(Node* tree, NNEvaluation* evaluation);

  bool search_active() const { return shared_data_->search_active; }
  auto& dirichlet_gen() { return shared_data_->dirichlet_gen; }
  auto& rng() { return shared_data_->rng; }
  float root_softmax_temperature() const { return shared_data_->root_softmax_temperature.value(); }

  SharedData* const shared_data_;
  NNEvaluationService* const nn_eval_service_;
  const ManagerParams* manager_params_;
  const SearchParams* search_params_ = nullptr;
  std::thread* thread_ = nullptr;
  profiler_t profiler_;
  const int thread_id_;
};

}  // namespace mcts

#include <mcts/inl/SearchThread.inl>
