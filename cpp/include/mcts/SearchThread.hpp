#pragma once

#include <bitset>
#include <mutex>
#include <thread>
#include <vector>

#include <core/concepts/Game.hpp>
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

template <core::concepts::Game Game>
class SearchThread {
 public:
  using NNEvaluation = mcts::NNEvaluation<Game>;
  using NNEvaluationService = mcts::NNEvaluationService<Game>;
  using Node = mcts::Node<Game>;
  using NodeCache = mcts::NodeCache<Game>;
  using PUCTStats = mcts::PUCTStats<Game>;
  using SharedData = mcts::SharedData<Game>;
  using LocalPolicyArray = typename Node::LocalPolicyArray;
  using edge_t = typename Node::edge_t;
  using base_state_vec_t = typename SharedData::base_state_vec_t;

  using IO = Game::IO;
  using Rules = typename Game::Rules;
  using FullState = typename Game::FullState;
  using ActionMask = typename Game::ActionMask;
  using NNEvaluation_sptr = typename NNEvaluation::sptr;
  using PolicyShape = typename Game::PolicyShape;
  using PolicyTensor = typename Game::PolicyTensor;
  using ValueArray = typename Game::ValueArray;
  using ValueTensor = typename NNEvaluation::ValueTensor;

  static constexpr int kNumPlayers = Game::kNumPlayers;
  static constexpr int kNumActions = Game::kNumActions;

  using profiler_t = search_thread_profiler_t;

  SearchThread(SharedData* shared_data, NNEvaluationService* nn_eval_service,
               const ManagerParams* manager_params, int thread_id);
  ~SearchThread();

  int thread_id() const { return thread_id_; }
  std::string thread_id_whitespace() const {
    return util::make_whitespace(kThreadWhitespaceLength * thread_id_);
  }
  std::string break_plus_thread_id_whitespace() const {
    int n = util::logging::kTimestampPrefixLength + kThreadWhitespaceLength * thread_id_;
    return "\n" + util::make_whitespace(n);
  }

  void set_profiling_dir(const boost::filesystem::path& profiling_dir);

  void dump_profiling_stats() { profiler_.dump(64); }

 private:
  struct VirtualIncrement {
    void operator()(Node* node) const { node->stats().virtual_increment(); }
  };

  struct RealIncrement {
    void operator()(Node* node) const { node->stats().real_increment(); }
  };

  struct IncrementTransfer {
    void operator()(Node* node) const { node->stats().increment_transfer(); }
  };

  struct SetEvalExact {
    SetEvalExact(const ValueArray& value) : value(value) {}
    void operator()(Node* node) const { node->stats().set_eval_exact(value); }
    const ValueArray& value;
  };

  struct SetEvalWithVirtualUndo {
    SetEvalWithVirtualUndo(const ValueArray& value) : value(value) {}
    void operator()(Node* node) const { node->stats().set_eval_with_virtual_undo(value); }
    const ValueArray& value;
  };

  struct evaluation_result_t {
    NNEvaluation_sptr evaluation;
    bool backpropagated_virtual_loss;
  };

  struct visitation_t {
    visitation_t(Node* n, edge_t* e) : node(n), edge(e) {}
    Node* node;
    edge_t* edge;
  };
  using search_path_t = std::vector<visitation_t>;

  void wait_for_activation() const;
  void perform_visits();
  void deactivate() const;
  void loop();
  void visit(Node* tree, edge_t* edge, move_number_t move_number);
  void add_dirichlet_noise(LocalPolicyArray& P);
  void virtual_backprop();
  void pure_backprop(const ValueArray& value);
  void backprop_with_virtual_undo(const ValueArray& value);
  void short_circuit_backprop(edge_t* last_edge);
  evaluation_result_t evaluate(Node* tree);
  void evaluate_unset(Node* tree, std::unique_lock<std::mutex>* lock, evaluation_result_t* data);
  std::string search_path_str() const;  // slow, for debugging

  /*
   * Used in visit().
   *
   * Applies PUCT criterion to select the best child-index to visit from the given Node.
   *
   * TODO: as we experiment with things like auxiliary NN output heads, dynamic cPUCT values,
   * etc., this method will evolve. It probably makes sense to have the behavior as part of the
   * Tensorizor, since there is coupling with NN architecture (in the form of output heads).
   */
  core::action_index_t get_best_action_index(Node* tree, NNEvaluation* evaluation);

  auto& dirichlet_gen() { return shared_data_->dirichlet_gen; }
  auto& rng() { return shared_data_->rng; }
  float root_softmax_temperature() const { return shared_data_->root_softmax_temperature.value(); }

  FullState state_;
  base_state_vec_t state_history_;
  SharedData* const shared_data_;
  NNEvaluationService* const nn_eval_service_;
  const ManagerParams* manager_params_;
  std::thread* thread_ = nullptr;
  search_path_t search_path_;
  profiler_t profiler_;
  const int thread_id_;
};

}  // namespace mcts

#include <inline/mcts/SearchThread.inl>
