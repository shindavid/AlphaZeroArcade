#pragma once

#include "beta0/APhiEvaluator.hpp"
#include "beta0/Edge.hpp"
#include "beta0/GraphTraits.hpp"
#include "beta0/ManagerParams.hpp"
#include "beta0/Node.hpp"
#include "beta0/NodeStableData.hpp"
#include "beta0/PuctCalculator.hpp"
#include "beta0/SearchContext.hpp"
#include "beta0/SearchResults.hpp"
#include "beta0/TrainingInfo.hpp"
#include "core/ActionPrinter.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionSymmetryTable.hpp"
#include "core/BasicTypes.hpp"
#include "core/ChanceEventHandleRequest.hpp"
#include "core/GameServerBase.hpp"
#include "core/InfoSetIterator.hpp"
#include "search/LookupTable.hpp"
#include "search/NNEvalTraits.hpp"
#include "search/NNEvaluationServiceBase.hpp"
#include "search/NNEvaluationServiceFactory.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchRequest.hpp"
#include "search/SearchResponse.hpp"
#include "util/EigenUtil.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <EigenRand/EigenRand>

#include <memory>
#include <queue>
#include <vector>

namespace beta0 {

/*
 * The Manager class is the main entry point for doing BetaZero MCTS searches.
 *
 * Forked from alpha0::Manager. Key differences:
 * - NodeStableData includes uncertainty_ (U from the "uncertainty" network head)
 * - NodeStats includes W (LoTV uncertainty estimate)
 * - Edge includes child_AU (per-action uncertainty from AU head)
 * - load_evaluations reads 5 heads: P, V, U, AV, AU
 * - update_stats computes W alongside Q via LoTV formula
 * - SearchResults includes W and AU
 */
template <beta0::concepts::Spec Spec>
class Manager {
 public:
  using Game = Spec::Game;
  using Edge = beta0::Edge<Spec>;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using PolicyTensor = PolicyEncoding::Tensor;
  using Params = beta0::ManagerParams<Spec>;
  using SearchResults = beta0::SearchResults<Spec>;
  using TrainingInfo = beta0::TrainingInfo<Spec>;
  using GraphTraits = beta0::GraphTraits<Spec>;
  using NetworkHeads = Spec::NetworkHeads;
  using InputFrame = Spec::InputFrame;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using NNEvalTraits = search::NNEvalTraits<GraphTraits, TensorEncodings, NNEvaluation>;
  using EvalServiceBase = search::NNEvaluationServiceBase<NNEvalTraits>;
  using EvalServiceFactory = search::NNEvaluationServiceFactory<NNEvalTraits>;
  using EvalServiceBase_sptr = std::shared_ptr<EvalServiceBase>;

  using Visitation = beta0::SearchContext<Spec>::Visitation;
  using Node = beta0::Node<Spec>;
  using NodeStats = beta0::NodeStats<Spec>;
  using NodeStableData = beta0::NodeStableData<Spec>;

  using LookupTable = search::LookupTable<GraphTraits>;

  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ActionValueEncoding = TensorEncodings::ActionValueEncoding;
  using ChanceEventHandleRequest = core::ChanceEventHandleRequest<Game>;

  using ActionRequest = core::ActionRequest<Game>;
  using ActionPrinter = core::ActionPrinter<Game>;
  using ActionSymmetryTable = core::ActionSymmetryTable<Spec>;
  using ActionSymmetryTableBuilder = ActionSymmetryTable::Builder;
  using Rules = Game::Rules;
  using Symmetries = Spec::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;
  using IO = Game::IO;
  using Constants = Game::Constants;
  using State = Game::State;
  using InputEncoder = TensorEncodings::InputEncoder;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using Transposer = Spec::Transposer;
  using TransposeKey = Transposer::Key;
  using PuctCalculator = beta0::PuctCalculator<Spec>;

  using GameResultTensor = GameResultEncoding::Tensor;
  using ValueArray = Game::Types::ValueArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using SymmetryMask = Game::Types::SymmetryMask;
  using InfoSetIterator = core::InfoSetIterator<Game>;
  using player_bitset_t = Game::Types::player_bitset_t;

  struct RootInfo {
    void clear();

    State state;
    InputEncoder input_encoder;
    int state_step = 0;  // incremented every time state changes
    core::node_pool_index_t node_index = -1;
    core::seat_index_t active_seat = -1;
    bool add_noise = false;
  };

  using SearchContext = beta0::SearchContext<Spec>;
  using SearchResponse = search::SearchResponse<SearchResults>;
  using SearchRequest = search::SearchRequest;
  using SearchParams = search::SearchParams;

  using post_visit_func_t = std::function<void()>;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  enum execution_state_t : int8_t { kIdle, kInitializingRoot, kInVisitLoop };

  struct StateMachine {
    mutable mit::mutex mutex;
    int16_t in_visit_loop_count = 0;
    execution_state_t state = kIdle;
  };

  Manager(const Params&, core::GameServerBase* server = nullptr,
          EvalServiceBase_sptr service = nullptr);

  Manager(core::mutex_vec_sptr_t& node_mutex_pool, core::mutex_vec_sptr_t& context_mutex_pool,
          const Params& params, core::GameServerBase* server = nullptr,
          EvalServiceBase_sptr service = nullptr);

  ~Manager();

  const Params& params() const { return params_; }
  int num_search_threads() const { return params().num_search_threads; }
  bool multithreaded() const { return num_search_threads() > 1; }

  void start();
  void clear();
  void receive_state_change(core::seat_index_t, const State&, const Move&);
  void update(const Move&);
  void backtrack(InfoSetIterator it, core::step_t step);

  void set_search_params(const search::SearchParams& search_params);
  SearchResponse search(const search::SearchRequest& request);
  core::yield_instruction_t load_root_action_values(const ChanceEventHandleRequest&,
                                                    core::seat_index_t seat, TrainingInfo&);

  const LookupTable* lookup_table() const { return &lookup_table_; }
  const RootInfo* root_info() const { return &root_info_; }
  LookupTable* lookup_table() { return &lookup_table_; }
  RootInfo* root_info() { return &root_info_; }

  void end_session() { nn_eval_service_->end_session(); }

  void set_post_visit_func(post_visit_func_t func) { post_visit_func_ = func; }

  // Loads A_phi weights from a flat float array (W_AD, W_out, b_out layout).
  // Safe to call between searches (not during active search).
  void set_aphi_weights(const float* weights, size_t n_floats);

 private:
  using context_vec_t = std::vector<SearchContext>;
  using context_id_queue_t = std::queue<core::context_id_t>;

  static void print_visit_info(const SearchContext&);

  template <typename MutexProtectedFunc>
  void backprop(SearchContext& context, Node* node, Edge* edge, MutexProtectedFunc&& func);

  static void init_node_stats_from_terminal(Node* node);
  static void update_node_stats(Node* node, bool undo_virtual);
  static void update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual);
  static void virtually_update_node_stats(Node* node);
  static void virtually_update_node_stats_and_edge(Node* node, Edge* edge);
  static void undo_virtual_update(Node* node, Edge* edge);

  void validate_search_path(const SearchContext& context);
  static bool should_short_circuit(const Edge* edge, const Node* child);
  bool more_search_iterations_needed(const Node* root) const;
  void init_root_info(search::RootInitPurpose);
  void init_root_edges() {}
  int get_best_child_index(const SearchContext& context);
  void load_evaluations(SearchContext& context);

  void update_stats(NodeStats& stats, const Node* node);
  void write_results(const Node* root);
  void validate_state(Node* node);
  void transform_policy(SearchContext&, LocalPolicyArray& P);
  void add_dirichlet_noise(LocalPolicyArray& P);
  void prune_policy_target();
  void print_action_selection_details(const SearchContext& context, const PuctCalculator& selector,
                                      int argmax_index);
  void load_action_symmetries(const Node* root);

  Manager(bool dummy, core::mutex_vec_sptr_t node_mutex_pool,
          core::mutex_vec_sptr_t context_mutex_pool, const Params& params, core::GameServerBase*,
          EvalServiceBase_sptr service);

  SearchResponse search_helper(const search::SearchRequest& request);

  int update_state_machine_to_in_visit_loop(SearchContext& context);
  core::yield_instruction_t mark_as_done_with_visit_loop(SearchContext& context,
                                                         int extra_enqueue_count);
  void init_context(core::context_id_t i);

  core::yield_instruction_t begin_root_initialization(SearchContext& context);
  core::yield_instruction_t resume_root_initialization(SearchContext& context);
  core::yield_instruction_t begin_node_initialization(SearchContext& context);
  core::yield_instruction_t resume_node_initialization(SearchContext& context);

  core::yield_instruction_t begin_search_iteration(SearchContext& context);
  core::yield_instruction_t resume_search_iteration(SearchContext& context);
  core::yield_instruction_t begin_visit(SearchContext& context);
  core::yield_instruction_t resume_visit(SearchContext& context);
  core::yield_instruction_t begin_expansion(SearchContext& context);
  core::yield_instruction_t resume_expansion(SearchContext& context);

  void virtual_backprop(SearchContext& context);
  void undo_virtual_backprop(SearchContext& context);
  void standard_backprop(SearchContext& context, bool undo_virtual = false);
  void short_circuit_backprop(SearchContext& context);

  core::node_pool_index_t lookup_child_by_move(const Node* node, const Move& move) const;
  void initialize_edges(Node* node, const MoveSet& valid_moves);
  bool all_children_edges_initialized(const Node* root) const;
  void add_pending_notification(SearchContext& context, Edge* edge);
  void set_edge_state(SearchContext& context, Edge* edge, Edge::expansion_state_t state);
  void pre_expand_children(SearchContext& context, Node* node);
  int sample_chance_child_index(const SearchContext& context);
  group::element_t get_random_symmetry(const InputEncoder& input_encoder) const;
  group::element_t get_random_symmetry(const InputEncoder& input_encoder,
                                       const State& next_state) const;
  void apply_move(State& state, InputEncoder& input_encoder, const Move& move);

  static int next_instance_id_;
  int manager_id_;

  const Params params_;
  LookupTable lookup_table_;
  RootInfo root_info_;
  math::ExponentialDecay root_softmax_temperature_;
  context_vec_t contexts_;
  StateMachine state_machine_;
  SearchResults results_;
  ActionSymmetryTableBuilder action_symmetry_table_builder_;
  search::SearchParams search_params_;
  EvalServiceBase_sptr nn_eval_service_;
  bool connected_ = false;
  bool mid_load_root_action_values_ = false;

  core::mutex_vec_sptr_t context_mutex_pool_;
  post_visit_func_t post_visit_func_;

  mutable eigen_util::UniformDirichletGen<float> dirichlet_gen_;
  mutable Eigen::Rand::P8_mt19937_64 rng_;

  APhiEvaluator<Spec> aphi_evaluator_;
};

template <beta0::concepts::Spec Spec>
int Manager<Spec>::next_instance_id_ = 0;

}  // namespace beta0

#include "inline/beta0/Manager.inl"
