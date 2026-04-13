#pragma once

#include "alpha0/GeneralContext.hpp"
#include "alpha0/PuctCalculator.hpp"
#include "alpha0/SearchContext.hpp"
#include "alpha0/Spec.hpp"
#include "core/ActionPrinter.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionSymmetryTable.hpp"
#include "core/BasicTypes.hpp"
#include "core/ChanceEventHandleRequest.hpp"
#include "core/GameServerBase.hpp"
#include "core/StateIterator.hpp"
#include "search/NNEvaluationServiceBase.hpp"
#include "search/NNEvaluationServiceFactory.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchRequest.hpp"
#include "search/SearchResponse.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <memory>
#include <queue>
#include <vector>

namespace alpha0 {

/*
 * The Manager class is the main entry point for doing MCTS searches.
 *
 * It maintains the search-tree and manages the threads and services that perform the search.
 */
template <alpha0::concepts::EvalSpec EvalSpec>
class Manager {
 public:
  using Spec = alpha0::Spec<typename EvalSpec::Game, EvalSpec>;
  using Game = EvalSpec::Game;
  using Edge = Spec::Edge;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using PolicyTensor = PolicyEncoding::Tensor;
  using AuxState = Spec::AuxState;
  using SearchResults = alpha0::SearchResults<EvalSpec>;
  using ManagerParams = alpha0::ManagerParams<EvalSpec>;
  using TrainingInfo = alpha0::TrainingInfo<EvalSpec>;
  using EvalServiceBase = search::NNEvaluationServiceBase<Spec>;
  using EvalServiceFactory = search::NNEvaluationServiceFactory<Spec>;
  using EvalServiceBase_sptr = std::shared_ptr<EvalServiceBase>;

  using Visitation = alpha0::SearchContext<EvalSpec>::Visitation;
  using Node = Spec::Node;
  using NodeStats = Spec::NodeStats;

  using LookupTable = search::LookupTable<Spec>;

  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ActionValueEncoding = TensorEncodings::ActionValueEncoding;
  using ChanceEventHandleRequest = core::ChanceEventHandleRequest<Game>;

  using GeneralContext = alpha0::GeneralContext<EvalSpec>;
  using RootInfo = GeneralContext::RootInfo;
  using SearchContext = alpha0::SearchContext<EvalSpec>;
  using SearchResponse = search::SearchResponse<SearchResults>;
  using SearchRequest = search::SearchRequest;
  using SearchParams = search::SearchParams;

  using ActionRequest = core::ActionRequest<Game>;
  using ActionPrinter = core::ActionPrinter<Game>;
  using ActionSymmetryTable = core::ActionSymmetryTable<EvalSpec>;
  using Rules = Game::Rules;
  using Symmetries = EvalSpec::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;
  using IO = Game::IO;
  using Constants = Game::Constants;
  using State = Game::State;
  using InputEncoder = TensorEncodings::InputEncoder;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using InputFrame = EvalSpec::InputFrame;
  using Transposer = EvalSpec::Transposer;
  using TransposeKey = Transposer::Key;
  using PuctCalculator = alpha0::PuctCalculator<Spec>;

  using GameResultTensor = GameResultEncoding::Tensor;
  using ValueArray = Game::Types::ValueArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using SymmetryMask = Game::Types::SymmetryMask;
  using StateIterator = core::StateIterator<Game>;
  using player_bitset_t = Game::Types::player_bitset_t;

  using post_visit_func_t = std::function<void()>;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  enum execution_state_t : int8_t { kIdle, kInitializingRoot, kInVisitLoop };

  struct StateMachine {
    mutable mit::mutex mutex;
    int16_t in_visit_loop_count = 0;
    execution_state_t state = kIdle;
  };

  Manager(const ManagerParams&, core::GameServerBase* server = nullptr,
          EvalServiceBase_sptr service = nullptr);

  Manager(core::mutex_vec_sptr_t& node_mutex_pool, core::mutex_vec_sptr_t& context_mutex_pool,
          const ManagerParams& params, core::GameServerBase* server = nullptr,
          EvalServiceBase_sptr service = nullptr);

  ~Manager();

  const ManagerParams& params() const { return general_context_.manager_params; }
  int num_search_threads() const { return params().num_search_threads; }
  bool multithreaded() const { return num_search_threads() > 1; }

  void start();
  void clear();
  void receive_state_change(core::seat_index_t, const State&, const Move&);
  void update(const Move&);
  void backtrack(StateIterator it, core::step_t step);

  void set_search_params(const search::SearchParams& search_params);
  SearchResponse search(const search::SearchRequest& request);
  core::yield_instruction_t load_root_action_values(const ChanceEventHandleRequest&,
                                                    core::seat_index_t seat, TrainingInfo&);

  const LookupTable* lookup_table() const { return &general_context_.lookup_table; }
  const RootInfo* root_info() const { return &general_context_.root_info; }
  LookupTable* lookup_table() { return &general_context_.lookup_table; }
  RootInfo* root_info() { return &general_context_.root_info; }

  void end_session() { nn_eval_service_->end_session(); }

  void set_post_visit_func(post_visit_func_t func) { post_visit_func_ = func; }

 private:
  using context_vec_t = std::vector<SearchContext>;
  using context_id_queue_t = std::queue<core::context_id_t>;

  // --- Methods moved from alpha0::Algorithms ---
  static void algo_print_visit_info(const SearchContext&);

  template <typename MutexProtectedFunc>
  static void algo_backprop(SearchContext& context, Node* node, Edge* edge,
                            MutexProtectedFunc&& func);

  static void algo_init_node_stats_from_terminal(Node* node);
  static void algo_update_node_stats(Node* node, bool undo_virtual);
  static void algo_update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual);
  static void algo_virtually_update_node_stats(Node* node);
  static void algo_virtually_update_node_stats_and_edge(Node* node, Edge* edge);
  static void algo_undo_virtual_update(Node* node, Edge* edge);

  static void algo_validate_search_path(const SearchContext& context);
  static bool algo_should_short_circuit(const Edge* edge, const Node* child);
  static bool algo_more_search_iterations_needed(const GeneralContext&, const Node* root);
  static void algo_init_root_info(GeneralContext&, search::RootInitPurpose);
  static void algo_init_root_edges(GeneralContext&) {}
  static int algo_get_best_child_index(const SearchContext& context);
  static void algo_load_evaluations(SearchContext& context);
  static void algo_to_results(const GeneralContext&, SearchResults&);

  // --- Helpers moved from alpha0::Algorithms ---
  static void algo_update_stats(NodeStats& stats, const Node* node, LookupTable& lookup_table);
  static void algo_write_results(const GeneralContext&, const Node* root, SearchResults& results);
  static void algo_validate_state(LookupTable& lookup_table, Node* node);
  static void algo_transform_policy(SearchContext&, LocalPolicyArray& P);
  static void algo_add_dirichlet_noise(GeneralContext&, LocalPolicyArray& P);
  static void algo_prune_policy_target(const GeneralContext&, SearchResults&);
  static void algo_print_action_selection_details(const SearchContext& context,
                                                  const PuctCalculator& selector, int argmax_index);
  static void algo_load_action_symmetries(const GeneralContext&, const Node* root,
                                          SearchResults&);

  Manager(bool dummy, core::mutex_vec_sptr_t node_mutex_pool,
          core::mutex_vec_sptr_t context_mutex_pool, const ManagerParams& params,
          core::GameServerBase*, EvalServiceBase_sptr service);

  SearchResponse search_helper(const search::SearchRequest& request);

  int update_state_machine_to_in_visit_loop(SearchContext& context);

  core::yield_instruction_t mark_as_done_with_visit_loop(SearchContext& context,
                                                         int extra_enqueue_count);

  void init_context(core::context_id_t);
  core::yield_instruction_t begin_root_initialization(SearchContext&);
  core::yield_instruction_t resume_root_initialization(SearchContext&);
  core::yield_instruction_t begin_node_initialization(SearchContext&);
  core::yield_instruction_t resume_node_initialization(SearchContext& context);
  core::yield_instruction_t begin_search_iteration(SearchContext& context);
  core::yield_instruction_t resume_search_iteration(SearchContext& context);
  core::yield_instruction_t begin_visit(SearchContext& context);
  core::yield_instruction_t resume_visit(SearchContext& context);
  core::yield_instruction_t begin_expansion(SearchContext& context);
  core::yield_instruction_t resume_expansion(SearchContext& context);

  static void virtual_backprop(SearchContext& context);
  static void undo_virtual_backprop(SearchContext& context);
  static void standard_backprop(SearchContext& context, bool undo_virtual = false);
  static void short_circuit_backprop(SearchContext& context);

  core::node_pool_index_t lookup_child_by_move(const Node* node, const Move& move) const;
  void initialize_edges(Node* node, const MoveSet& valid_moves);
  bool all_children_edges_initialized(const Node* root) const;
  void add_pending_notification(SearchContext&, Edge*);
  void set_edge_state(SearchContext&, Edge*, Edge::expansion_state_t);
  void pre_expand_children(SearchContext& context, Node* node);
  int sample_chance_child_index(const SearchContext& context);
  void apply_move(State& state, InputEncoder& input_encoder, const Move& move);

  group::element_t get_random_symmetry(const InputEncoder&) const;
  group::element_t get_random_symmetry(const InputEncoder&, const State& next_state) const;

  static inline int next_instance_id_ = 0;

  const int manager_id_ = -1;

  GeneralContext general_context_;

  post_visit_func_t post_visit_func_;

  context_vec_t contexts_;
  StateMachine state_machine_;
  core::mutex_vec_sptr_t context_mutex_pool_;
  EvalServiceBase_sptr nn_eval_service_;

  SearchResults results_;

  bool mid_load_root_action_values_ = false;
  bool connected_ = false;
};

}  // namespace alpha0

#include "inline/alpha0/Manager.inl"
