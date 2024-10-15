#pragma once

#include <bitset>
#include <mutex>
#include <thread>
#include <vector>

#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationRequest.hpp>
#include <mcts/NNEvaluationService.hpp>
#include <mcts/Node.hpp>
#include <mcts/ActionSelector.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/SharedData.hpp>
#include <mcts/TypeDefs.hpp>

class SearchThreadTest;  // forward-declaration

namespace mcts {

template <core::concepts::Game Game>
class SearchThread {
 public:
  using ManagerParams = mcts::ManagerParams<Game>;
  using NNEvaluation = mcts::NNEvaluation<Game>;
  using NNEvaluationRequest = mcts::NNEvaluationRequest<Game>;
  using NNEvaluationService = mcts::NNEvaluationService<Game>;
  using Node = mcts::Node<Game>;
  using ActionSelector = mcts::ActionSelector<Game>;
  using SharedData = mcts::SharedData<Game>;
  using LocalPolicyArray = Node::LocalPolicyArray;
  using edge_t = Node::edge_t;
  using node_pool_index_t = Node::node_pool_index_t;
  using StateHistoryArray = SharedData::StateHistoryArray;
  using LookupTable = Node::LookupTable;

  using StateHistory = Game::StateHistory;
  using State = Game::State;
  using ActionOutcome = Game::Types::ActionOutcome;
  using ActionMask = Game::Types::ActionMask;
  using NNEvaluation_sptr = NNEvaluation::sptr;
  using PolicyShape = Game::Types::PolicyShape;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueArray = Game::Types::ValueArray;
  using SymmetryMask = Game::Types::SymmetryMask;

  using item_vec_t = NNEvaluationRequest::item_vec_t;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kNumActions = Game::Constants::kNumActions;

  using profiler_t = search_thread_profiler_t;

  SearchThread(SharedData* shared_data, NNEvaluationService* nn_eval_service,
               const ManagerParams* manager_params, int thread_id);
  ~SearchThread();

  void start();
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
  struct visitation_t {
    Node* node;
    edge_t* edge;  // emanates from node, possibly nullptr
  };

  using search_path_t = std::vector<visitation_t>;

  void wait_for_activation() const;
  Node* init_root_node();
  void init_node(StateHistory*, node_pool_index_t, Node* node);
  void expand_all_children(Node*, NNEvaluationRequest* request=nullptr);
  void transform_policy(node_pool_index_t, LocalPolicyArray&) const;
  void perform_visits();
  void deactivate() const;
  void loop();
  void print_visit_info(Node* node);
  void visit(Node* node);
  void add_dirichlet_noise(LocalPolicyArray& P) const;
  void virtual_backprop();
  void pure_backprop(const ValueArray& value);
  void standard_backprop();
  void short_circuit_backprop();
  bool expand(StateHistory*, Node*, edge_t*);  // returns true if a new node was expanded
  std::string search_path_str() const;  // slow, for debugging
  void calc_canonical_state_data();

  // In debug builds, calls node->validate_state() for each node in search path.
  // In release builds, NO-OP.
  void validate_search_path() const;

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
  void print_action_selection_details(Node* node, const ActionSelector&, int argmax_index) const;

  auto& dirichlet_gen() const { return shared_data_->dirichlet_gen; }
  auto& rng() const { return shared_data_->rng; }
  float root_softmax_temperature() const { return shared_data_->root_softmax_temperature.value(); }

  ActionOutcome outcome_;
  SharedData* const shared_data_;
  NNEvaluationService* const nn_eval_service_;
  const ManagerParams* manager_params_;
  std::thread* thread_ = nullptr;

  group::element_t canonical_sym_;
  StateHistory raw_history_;

  /*
   * These variables would more naturally be declared as local variables in the contexts in which
   * they are used, but they are declared here to avoid repeated allocation/deallocation.
   */
  struct pseudo_local_vars_t {
    StateHistory canonical_history;
    item_vec_t request_items;
    StateHistoryArray state_history_array;
  };

  pseudo_local_vars_t pseudo_local_vars_;

  search_path_t search_path_;
  profiler_t profiler_;
  const int thread_id_;

  friend ::SearchThreadTest;
};

}  // namespace mcts

#include <inline/mcts/SearchThread.inl>
