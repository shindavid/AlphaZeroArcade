#pragma once

#include "alpha0/Edge.hpp"
#include "alpha0/GeneralContext.hpp"
#include "alpha0/GraphTraits.hpp"
#include "alpha0/Node.hpp"
#include "alpha0/PuctCalculator.hpp"
#include "alpha0/SearchContext.hpp"
#include "alpha0/SearchResults.hpp"
#include "alpha0/TrainingInfo.hpp"
#include "core/ActionPrinter.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionSymmetryTable.hpp"
#include "core/BasicTypes.hpp"
#include "core/ChanceEventHandleRequest.hpp"
#include "core/GameServerBase.hpp"
#include "core/StateIterator.hpp"
#include "search/NNEvalTraits.hpp"
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
template <alpha0::concepts::Spec Spec>
class Manager {
 public:
  using Game = Spec::Game;
  using Edge = alpha0::Edge<Spec>;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using TensorEncodings = Spec::TensorEncodings;
  using PolicyEncoding = TensorEncodings::PolicyEncoding;
  using PolicyTensor = PolicyEncoding::Tensor;
  using AuxState = alpha0::AuxState<alpha0::ManagerParams<Spec>>;
  using SearchResults = alpha0::SearchResults<Spec>;
  using ManagerParams = alpha0::ManagerParams<Spec>;
  using TrainingInfo = alpha0::TrainingInfo<Spec>;
  using GraphTraits = alpha0::GraphTraits<Spec>;
  using NetworkHeads = Spec::NetworkHeads;
  using InputFrame = Spec::InputFrame;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using NNEvalTraits = search::NNEvalTraits<GraphTraits, TensorEncodings, NNEvaluation>;
  using EvalServiceBase = search::NNEvaluationServiceBase<NNEvalTraits>;
  using EvalServiceFactory = search::NNEvaluationServiceFactory<NNEvalTraits>;
  using EvalServiceBase_sptr = std::shared_ptr<EvalServiceBase>;

  using Visitation = alpha0::SearchContext<Spec>::Visitation;
  using Node = alpha0::Node<Spec>;
  using NodeStats = alpha0::NodeStats<Spec>;

  using LookupTable = search::LookupTable<GraphTraits>;

  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ActionValueEncoding = TensorEncodings::ActionValueEncoding;
  using ChanceEventHandleRequest = core::ChanceEventHandleRequest<Game>;

  using GeneralContext = alpha0::GeneralContext<Spec>;
  using RootInfo = GeneralContext::RootInfo;
  using SearchContext = alpha0::SearchContext<Spec>;
  using SearchResponse = search::SearchResponse<SearchResults>;
  using SearchRequest = search::SearchRequest;
  using SearchParams = search::SearchParams;

  using ActionRequest = core::ActionRequest<Game>;
  using ActionPrinter = core::ActionPrinter<Game>;
  using ActionSymmetryTable = core::ActionSymmetryTable<Spec>;
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
  static int get_best_child_index(const SearchContext& context);
  void load_evaluations(SearchContext& context);
  void to_results(SearchResults&);

  void update_stats(NodeStats& stats, const Node* node);
  void write_results(const Node* root, SearchResults& results);
  void validate_state(Node* node);
  void transform_policy(SearchContext&, LocalPolicyArray& P);
  void add_dirichlet_noise(LocalPolicyArray& P);
  void prune_policy_target(SearchResults&);
  static void print_action_selection_details(const SearchContext& context,
                                             const PuctCalculator& selector, int argmax_index);
  void load_action_symmetries(const Node* root, SearchResults&);

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

  void virtual_backprop(SearchContext& context);
  void undo_virtual_backprop(SearchContext& context);
  void standard_backprop(SearchContext& context, bool undo_virtual = false);
  void short_circuit_backprop(SearchContext& context);

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
