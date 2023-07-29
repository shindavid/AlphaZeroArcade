#pragma once

#include <bitset>
#include <mutex>
#include <thread>
#include <vector>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/Node.hpp>
#include <mcts/NodeCache.hpp>
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
  using NodeCache = mcts::NodeCache<GameState, Tensorizor>;
  using PUCTStats = mcts::PUCTStats<GameState, Tensorizor>;
  using SharedData = mcts::SharedData<GameState, Tensorizor>;

  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using NNEvaluation_sptr = typename NNEvaluation::sptr;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = GameStateTypes::kNumGlobalActions;

  using dtype = torch_util::dtype;
  using profiler_t = search_thread_profiler_t;

  SearchThread(SharedData* shared_data, NNEvaluationService* nn_eval_service, const ManagerParams* manager_params,
               int thread_id);
  ~SearchThread();

  int thread_id() const { return thread_id_; }

  void join();
  void kill();
  void launch(const SearchParams* search_params, std::function<void()> f);
  bool needs_more_visits(Node* root, int tree_size_limit);
  void run();
  bool is_pondering() const { return search_params_->ponder; }

  void dump_profiling_stats() { profiler_.dump(64); }

private:
  struct evaluation_result_t {
    NNEvaluation_sptr evaluation;
    bool backpropagated_virtual_loss;
  };

  struct traverse_request_t {
    NodeCache* node_cache;
    child_index_t child_index;
    move_number_t move_number;
    float value_delta_threshold;
  };

  struct visitation_t {
    visitation_t(Node* n, child_index_t c) : node(n), child_index(c) {}
    Node* node;
    child_index_t child_index;
  };
  using search_path_t = std::vector<visitation_t>;

  void visit(Node* tree, child_index_t child_index, move_number_t move_number);
  void add_dirichlet_noise(LocalPolicyArray& P);
  void virtual_backprop();
  void backprop(const ValueArray& value);
  void backprop_with_virtual_undo(const ValueArray& value);
  evaluation_result_t evaluate(Node* tree);
  void evaluate_unset(Node* tree, std::unique_lock<std::mutex>* lock, evaluation_result_t* data);
  std::string search_path_str() const;  // slow, for debugging

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
  search_path_t search_path_;
  profiler_t profiler_;
  const int thread_id_;
};

}  // namespace mcts

#include <mcts/inl/SearchThread.inl>
