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
#include <mcts/PUCTStats.hpp>
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
  using PUCTStats = mcts::PUCTStats<Game>;
  using SharedData = mcts::SharedData<Game>;
  using LocalPolicyArray = Node::LocalPolicyArray;
  using edge_t = Node::edge_t;
  using node_pool_index_t = Node::node_pool_index_t;
  using base_state_vec_t = SharedData::base_state_vec_t;

  using FullState = Game::FullState;
  using BaseState = Game::BaseState;
  using ActionOutcome = Game::Types::ActionOutcome;
  using ActionMask = Game::Types::ActionMask;
  using NNEvaluation_sptr = NNEvaluation::sptr;
  using PolicyShape = Game::Types::PolicyShape;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using ValueTensor = NNEvaluation::ValueTensor;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kNumActions = Game::Constants::kNumActions;

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

  struct InitQAndRealIncrement {
    InitQAndRealIncrement(const ValueArray& value) : value(value) {}
    void operator()(Node* node) const { node->stats().init_q_and_real_increment(value); }
    const ValueArray& value;
  };

  struct InitQAndIncrementTransfer {
    InitQAndIncrementTransfer(const ValueArray& value) : value(value) {}
    void operator()(Node* node) const { node->stats().init_q_and_increment_transfer(value); }
    const ValueArray& value;
  };

  struct visitation_t {
    edge_t* edge;
    Node* child;
  };

  struct state_data_t {
    void load(const FullState& s, const base_state_vec_t& h);
    void add_state_to_history();
    void canonical_validate();

    FullState state;
    base_state_vec_t state_history;
  };

  using search_path_t = std::vector<visitation_t>;

  void wait_for_activation() const;
  Node* init_root_node();
  void init_node(state_data_t*, node_pool_index_t, Node* node);
  void transform_policy(node_pool_index_t, LocalPolicyArray&) const;
  void perform_visits();
  void deactivate() const;
  void loop();
  void print_visit_info(Node* node, edge_t* parent_edge);
  void visit(Node* node, edge_t* edge);
  void add_dirichlet_noise(LocalPolicyArray& P) const;
  void virtual_backprop();
  void pure_backprop(const ValueArray& value);
  void backprop_with_virtual_undo();
  void short_circuit_backprop();
  bool expand(state_data_t*, Node*, edge_t*);  // returns true if a new node was expanded
  std::string search_path_str() const;  // slow, for debugging
  void calc_canonical_state_data();

  /*
   * Used in visit().
   *
   * Applies PUCT criterion to select the best child-index to visit from the given Node.
   *
   * TODO: as we experiment with things like auxiliary NN output heads, dynamic cPUCT values,
   * etc., this method will evolve. It probably makes sense to have the behavior as part of the
   * Tensorizor, since there is coupling with NN architecture (in the form of output heads).
   */
  int get_best_child_index(Node* node);
  void print_puct_details(Node* node, const PUCTStats& stats, int argmax_index) const;

  auto& dirichlet_gen() const { return shared_data_->dirichlet_gen; }
  auto& rng() const { return shared_data_->rng; }
  float root_softmax_temperature() const { return shared_data_->root_softmax_temperature.value(); }

  ActionOutcome outcome_;
  SharedData* const shared_data_;
  NNEvaluationService* const nn_eval_service_;
  const ManagerParams* manager_params_;
  std::thread* thread_ = nullptr;

  group::element_t canonical_sym_;
  state_data_t raw_state_data_;
  state_data_t canonical_state_data_;  // pseudo-local-var, here to avoid repeated vector allocation

  search_path_t search_path_;
  profiler_t profiler_;
  const int thread_id_;
};

}  // namespace mcts

#include <inline/mcts/SearchThread.inl>
