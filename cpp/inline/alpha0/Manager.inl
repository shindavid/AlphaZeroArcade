#include "alpha0/Manager.hpp"

#include "core/BasicTypes.hpp"
#include "search/Constants.hpp"
#include "search/SearchParams.hpp"
#include "util/Asserts.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/FiniteGroups.hpp"
#include "util/LoggingUtil.hpp"
#include "util/MetaProgramming.hpp"

#include <boost/filesystem.hpp>
#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>
#include <spdlog/spdlog.h>

#include <sstream>
#include <unordered_map>

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
Manager<Spec>::Manager(bool dummy, core::mutex_vec_sptr_t node_mutex_pool,
                       core::mutex_vec_sptr_t context_mutex_pool, const ManagerParams& params,
                       core::GameServerBase* server, EvalServiceBase_sptr service)
    : manager_id_(next_instance_id_++),
      general_context_(params, node_mutex_pool),
      context_mutex_pool_(context_mutex_pool) {
  if (params.enable_pondering) {
    throw util::CleanException("Pondering mode temporarily unsupported");
  }

  if (service) {
    nn_eval_service_ = service;
  } else {
    nn_eval_service_ = EvalServiceFactory::create(params, server);
  }

  contexts_.resize(num_search_threads());
  for (int i = 0; i < num_search_threads(); ++i) {
    init_context(i);
  }
}

template <alpha0::concepts::Spec Spec>
Manager<Spec>::Manager(const ManagerParams& params, core::GameServerBase* server,
                       EvalServiceBase_sptr service)
    : Manager(true, std::make_shared<core::mutex_vec_t>(1), std::make_shared<core::mutex_vec_t>(1),
              params, server, service) {}

template <alpha0::concepts::Spec Spec>
Manager<Spec>::Manager(core::mutex_vec_sptr_t& node_mutex_pool,
                       core::mutex_vec_sptr_t& context_mutex_pool, const ManagerParams& params,
                       core::GameServerBase* server, EvalServiceBase_sptr service)
    : Manager(true, node_mutex_pool, context_mutex_pool, params, server, service) {}

template <alpha0::concepts::Spec Spec>
inline Manager<Spec>::~Manager() {
  clear();
  nn_eval_service_->disconnect();
}

template <alpha0::concepts::Spec Spec>
inline void Manager<Spec>::start() {
  clear();

  if (!connected_) {
    nn_eval_service_->connect();
    connected_ = true;
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::clear() {
  general_context_.clear();
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::receive_state_change(core::seat_index_t, const State&, const Move& move) {
  update(move);
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::update(const Move& move) {
  apply_move(root_info()->state, root_info()->input_encoder, move);
  root_info()->state_step++;
  general_context_.step();

  core::node_pool_index_t root_index = root_info()->node_index;
  if (root_index < 0) return;

  Node* root = lookup_table()->get_node(root_index);
  root_info()->node_index = lookup_child_by_move(root, move);  // tree reuse
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::backtrack(StateIterator it, core::step_t step) {
  general_context_.jump_to(it, step);
  const State& state = root_info()->state;
  TransposeKey key = Transposer::key(state);
  core::node_pool_index_t node_index = lookup_table()->lookup_node(key);
  root_info()->node_index = node_index;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::set_search_params(const SearchParams& params) {
  general_context_.search_params = params;
}

template <alpha0::concepts::Spec Spec>
typename Manager<Spec>::SearchResponse Manager<Spec>::search(const SearchRequest& request) {
  auto context_id = request.context_id();

  DEBUG_ASSERT(context_id < num_search_threads(), "Invalid context_id: {} (max: {})", context_id,
               num_search_threads());

  LOG_TRACE("{:>{}}search(): manager={} state={} c={}", "", contexts_[context_id].log_prefix_n(),
            manager_id_, state_machine_.state, context_id);

  SearchResponse response = search_helper(request);

  LOG_TRACE("{:>{}}{}() exit: manager={} state={} instr={}", "",
            contexts_[context_id].log_prefix_n(), __func__, manager_id_, state_machine_.state,
            response.yield_instruction);

  return response;
}

/*
 * Here, we do a skimmed-down version of Manager::search()
 */
template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::load_root_action_values(
  const ChanceEventHandleRequest& chance_request, core::seat_index_t seat,
  TrainingInfo& training_info) {
  if (!mid_load_root_action_values_) {
    algo_init_root_info(general_context_, search::kToLoadRootActionValues);

    // We do a dummy search with 0 iterations, just to get SearchThread to call init_root_node(),
    // which will expand all the root's children.
    constexpr int tree_size_limit = 0;
    constexpr bool full_search = true;
    constexpr bool ponder = false;
    SearchParams params{tree_size_limit, full_search, ponder};
    general_context_.search_params = params;
    mid_load_root_action_values_ = true;
  }

  SearchRequest request(chance_request.notification_unit);
  SearchResponse response = search(request);
  if (response.yield_instruction == core::kYield) return core::kYield;
  RELEASE_ASSERT(response.yield_instruction == core::kContinue);

  Node* root = lookup_table()->get_node(root_info()->node_index);
  const auto& stable_data = root->stable_data();
  const auto& frame = root_info()->input_encoder.current_frame();

  ActionValueTensor& action_values = training_info.action_values_target;
  action_values.setZero();

  for (int i = 0; i < stable_data.num_valid_moves; i++) {
    const Edge* edge = lookup_table()->get_edge(root, i);
    const Node* child = lookup_table()->get_node(edge->child_index);
    Move move = edge->move;
    auto index = PolicyEncoding::to_index(frame, move);

    ValueArray V;
    if (!child) {
      V = edge->child_AV;
    } else {
      V = child->stable_data().V();
    }
    eigen_util::chip_assign(action_values, eigen_util::reinterpret_as_tensor(V), index);
  }

  training_info.frame = frame;
  training_info.move = chance_request.chance_move;
  training_info.use_for_training = true;
  training_info.active_seat = seat;
  training_info.action_values_target_valid = true;

  mid_load_root_action_values_ = false;
  return core::kContinue;
}

template <alpha0::concepts::Spec Spec>
typename Manager<Spec>::SearchResponse Manager<Spec>::search_helper(const SearchRequest& request) {
  mit::unique_lock lock(state_machine_.mutex);
  auto context_id = request.context_id();
  SearchContext& context = contexts_[context_id];
  int extra_enqueue_count = 0;
  context.search_request = &request;

  if (state_machine_.state == kIdle) {
    RELEASE_ASSERT(context_id == 0);
    state_machine_.state = kInitializingRoot;
    lock.unlock();
    if (begin_root_initialization(context) == core::kContinue) {
      lock.lock();
      extra_enqueue_count = update_state_machine_to_in_visit_loop(context);
    } else {
      return SearchResponse::make_yield();
    }
  }

  if (state_machine_.state == kInitializingRoot) {
    RELEASE_ASSERT(context_id == 0);
    if (resume_root_initialization(context) == core::kYield) {
      return SearchResponse::make_yield();
    }
    extra_enqueue_count = update_state_machine_to_in_visit_loop(context);
  }

  RELEASE_ASSERT(state_machine_.state == kInVisitLoop);
  lock.unlock();
  if (context.mid_search_iteration) {
    if (resume_search_iteration(context) == core::kYield) {
      return SearchResponse::make_yield();
    }
  }

  Node* root = lookup_table()->get_node(root_info()->node_index);
  while (algo_more_search_iterations_needed(general_context_, root)) {
    if (begin_search_iteration(context) == core::kYield) {
      return SearchResponse::make_yield(extra_enqueue_count);
    }
  }

  lock.lock();
  core::yield_instruction_t yield_instruction =
    mark_as_done_with_visit_loop(context, extra_enqueue_count);
  if (yield_instruction == core::kDrop) {
    return SearchResponse::make_drop();
  }
  RELEASE_ASSERT(yield_instruction == core::kContinue);

  RootInfo& root_info = general_context_.root_info;
  LookupTable& lookup_table = general_context_.lookup_table;
  lookup_table.defragment(root_info.node_index);

  algo_to_results(general_context_, results_);
  return SearchResponse(&results_);
}

template <alpha0::concepts::Spec Spec>
int Manager<Spec>::update_state_machine_to_in_visit_loop(SearchContext& context) {
  // Assumes state_machine_.mutex is held
  if (state_machine_.state == kInVisitLoop) return 0;

  LOG_TRACE("{:>{}}{}(): manager={}", "", context.log_prefix_n(), __func__, manager_id_);

  state_machine_.state = kInVisitLoop;
  state_machine_.in_visit_loop_count = num_search_threads();

  for (auto& context2 : contexts_) {
    context2.in_visit_loop = true;
  }

  return state_machine_.in_visit_loop_count - 1;
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::mark_as_done_with_visit_loop(SearchContext& context,
                                                                      int extra_enqueue_count) {
  // Assumes state_machine_.mutex is held
  RELEASE_ASSERT(context.in_visit_loop);
  context.in_visit_loop = false;
  state_machine_.in_visit_loop_count--;
  if (state_machine_.in_visit_loop_count == extra_enqueue_count) {
    state_machine_.state = kIdle;

    if (extra_enqueue_count > 0) {
      // This means that we did update_state_machine_to_in_visit_loop() in the current search()
      // call. The significance of this is that the other context never got a chance to actually
      // do any work. We can't count the other contexts to get to this function and set their
      // own in_visit_loop to false, so we need to do it here.
      for (auto& context2 : contexts_) {
        context2.in_visit_loop = false;
      }
    }
    return core::kContinue;
  }
  return core::kDrop;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::init_context(core::context_id_t i) {
  SearchContext& context = contexts_[i];
  context.id = i;
  context.general_context = &general_context_;

  int n = context_mutex_pool_->size();
  if (n > 1) {
    context.pending_notifications_mutex_id = util::Random::uniform_sample(0, n);
  }
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_root_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  RootInfo& root_info = general_context_.root_info;
  LookupTable& lookup_table = general_context_.lookup_table;

  algo_init_root_info(general_context_, search::kForStandardSearch);

  core::node_pool_index_t root_index = root_info.node_index;
  Node* root = lookup_table.get_node(root_index);
  RELEASE_ASSERT(!root->is_terminal(), "unexpected terminal root node");

  if (!root->edges_initialized()) {
    initialize_edges(root, Game::Rules::analyze(root_info.state).valid_moves());
  }

  algo_init_root_edges(general_context_);

  if (all_children_edges_initialized(root)) {
    const ManagerParams& manager_params = general_context_.manager_params;
    const SearchParams& search_params = general_context_.search_params;
    bool pre_expand = manager_params.force_evaluate_all_root_children && search_params.full_search;
    if (pre_expand) {
      int n_moves = root->stable_data().num_valid_moves;
      for (int e = 0; e < n_moves; e++) {
        Edge* edge = lookup_table.get_edge(root, e);
        edge->was_pre_expanded = true;
      }
    }
    return core::kContinue;
  }

  context.current_state = root_info.state;
  context.input_encoder = root_info.input_encoder;
  context.active_seat = root_info.active_seat;
  context.initialization_index = root_index;
  return begin_node_initialization(context);
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_root_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  return resume_node_initialization(context);
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_node_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  const SearchParams& search_params = general_context_.search_params;
  const RootInfo& root_info = general_context_.root_info;
  LookupTable& lookup_table = general_context_.lookup_table;
  const ManagerParams& manager_params = general_context_.manager_params;

  InputEncoder& input_encoder = context.input_encoder;

  core::node_pool_index_t node_index = context.initialization_index;
  Node* node = lookup_table.get_node(node_index);
  const auto& frame = input_encoder.current_frame();

  context.mid_node_initialization = true;
  RELEASE_ASSERT(context.eval_request.num_fresh_items() == 0);

  bool is_root = (node_index == root_info.node_index);
  if (!node->is_terminal()) {
    bool pre_expand =
      manager_params.force_evaluate_all_root_children && is_root && search_params.full_search;

    if (!node->stable_data().R_valid) {
      group::element_t sym = get_random_symmetry(input_encoder);
      bool incorporate = manager_params.incorporate_sym_into_cache_key;
      auto eval_key = input_encoder.eval_key();
      context.eval_request.emplace_back(frame, node, &lookup_table, eval_key, input_encoder, sym,
                                        incorporate);
    }
    if (pre_expand) {
      pre_expand_children(context, node);
    }

    const SearchRequest& search_request = *context.search_request;
    context.eval_request.set_notification_task_info(search_request.notification_unit);

    if (nn_eval_service_->evaluate(context.eval_request) == core::kYield) return core::kYield;
  }

  return resume_node_initialization(context);
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_node_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  const RootInfo& root_info = general_context_.root_info;
  LookupTable& lookup_table = general_context_.lookup_table;

  const State& state = context.current_state;
  core::node_pool_index_t node_index = context.initialization_index;
  Node* node = lookup_table.get_node(node_index);
  bool is_root = (node_index == root_info.node_index);

  algo_load_evaluations(context);
  context.eval_request.mark_all_as_stale();

  if (!node->is_terminal() && node->stable_data().is_chance_node) {
    auto chance_dist = Rules::get_chance_distribution(state);
    for (int i = 0; i < node->stable_data().num_valid_moves; i++) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->chance_prob = chance_dist.get(edge->move);
    }
  }

  auto transpose_key = Transposer::key(state);
  bool overwrite = is_root;
  context.inserted_node_index = lookup_table.insert_node(transpose_key, node_index, overwrite);
  context.mid_node_initialization = false;
  return core::kContinue;
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_search_iteration(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  const RootInfo& root_info = general_context_.root_info;
  LookupTable& lookup_table = general_context_.lookup_table;

  Node* root = lookup_table.get_node(root_info.node_index);

  if (context.state_step != root_info.state_step) {
    context.state_step = root_info.state_step;
    context.current_state = root_info.state;
  } else {
    Rules::backtrack_state(context.current_state, root_info.state);
  }

  context.input_encoder = root_info.input_encoder;
  context.active_seat = root_info.active_seat;
  context.search_path.clear();
  context.search_path.emplace_back(root, nullptr);
  context.visit_node = root;
  context.mid_search_iteration = true;

  return resume_search_iteration(context);
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_search_iteration(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  const RootInfo& root_info = general_context_.root_info;

  if (context.mid_visit) {
    if (resume_visit(context) == core::kYield) return core::kYield;
  }

  while (context.visit_node) {
    if (begin_visit(context) == core::kYield) return core::kYield;
  }

  Rules::backtrack_state(context.current_state, root_info.state);
  context.input_encoder = root_info.input_encoder;
  context.active_seat = root_info.active_seat;
  if (post_visit_func_) post_visit_func_();
  context.mid_search_iteration = false;
  return core::kContinue;
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_visit(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  LookupTable& lookup_table = general_context_.lookup_table;

  Node* node = context.visit_node;
  algo_print_visit_info(context);
  context.mid_visit = true;
  context.expanded_new_node = false;

  const auto& stable_data = node->stable_data();
  if (stable_data.terminal) {
    standard_backprop(context);
    context.visit_node = nullptr;
    context.mid_visit = false;
    return core::kContinue;
  }

  int child_index;
  if (stable_data.is_chance_node) {
    child_index = sample_chance_child_index(context);
  } else {
    child_index = algo_get_best_child_index(context);
  }

  Edge* edge = lookup_table.get_edge(node, child_index);
  context.visit_edge = edge;
  context.search_path.back().edge = edge;
  context.applied_move = false;

  if (edge->state != Edge::kExpanded) {
    // reread state under mutex in case of race-condition
    mit::unique_lock lock(node->mutex());

    if (edge->state == Edge::kNotExpanded) {
      set_edge_state(context, edge, Edge::kMidExpansion);
      lock.unlock();

      apply_move(context.current_state, context.input_encoder, edge->move);
      const State& leaf_state = context.current_state;

      if (!Rules::is_chance_state(leaf_state)) {
        context.active_seat = Rules::get_current_player(leaf_state);
      }
      context.applied_move = true;

      if (begin_expansion(context) == core::kYield) return core::kYield;
    } else if (edge->state == Edge::kMidExpansion) {
      add_pending_notification(context, edge);
      return core::kYield;
    } else if (edge->state == Edge::kPreExpanded) {
      set_edge_state(context, edge, Edge::kMidExpansion);
      lock.unlock();

      DEBUG_ASSERT(edge->child_index >= 0);
      Node* child = lookup_table.get_node(edge->child_index);
      context.search_path.emplace_back(child, nullptr);

      if (algo_should_short_circuit(edge, child)) {
        short_circuit_backprop(context);
      } else {
        standard_backprop(context);
      }

      lock.lock();
      set_edge_state(context, edge, Edge::kExpanded);
      context.visit_node = nullptr;
      context.mid_visit = false;
      return core::kContinue;
    }
  }

  return resume_visit(context);
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_visit(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  Edge* edge = context.visit_edge;

  if (context.mid_expansion) {
    if (resume_expansion(context) == core::kYield) return core::kYield;
  }

  if (context.expanded_new_node) {
    context.visit_node = nullptr;
    context.mid_visit = false;
    LOG_TRACE("{:>{}}{}() continuing @{}", "", context.log_prefix_n(), __func__, __LINE__);
    return core::kContinue;
  }

  // we could have hit the yield in the kMidExpansion case, as the non-primary context
  RELEASE_ASSERT(edge->state == Edge::kExpanded, "Expected edge state to be kExpanded, but got {}",
                 edge->state);

  Node* child = lookup_table()->get_node(edge->child_index);
  if (child) {
    context.search_path.emplace_back(child, nullptr);

    if (algo_should_short_circuit(edge, child)) {
      short_circuit_backprop(context);
      context.visit_node = nullptr;
      context.mid_visit = false;
      LOG_TRACE("{:>{}}{}() continuing @{}", "", context.log_prefix_n(), __func__, __LINE__);
      return core::kContinue;
    }
  }
  if (!context.applied_move) {
    apply_move(context.current_state, context.input_encoder, edge->move);
    const State& state = context.current_state;

    if (!Rules::is_chance_state(state)) {
      context.active_seat = Rules::get_current_player(state);
    }
  }
  context.visit_node = child;
  context.mid_visit = false;
  LOG_TRACE("{:>{}}{}() continuing @{}", "", context.log_prefix_n(), __func__, __LINE__);
  return core::kContinue;
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_expansion(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  LookupTable& lookup_table = general_context_.lookup_table;

  context.mid_expansion = true;

  const State& state = context.current_state;
  TransposeKey transpose_key = Transposer::key(state);

  // NOTE: we do a lookup_node() call here, and then later, inside resume_node_initialization(), we
  // do a corresponding insert_node() call. This is analagous to:
  //
  // if key not in dict:
  //   ...
  //   dict[key] = value
  //
  // If there are multiple search threads, this represents a potential race-condition. The
  // straightforward solution is to hold a mutex during that entire sequence of operations. However,
  // this would hold the mutex for far too long.
  //
  // Instead, the below code carefully detects whether the race-condition has occurred, and if so,
  // keeps the first resume_node_initialization() and "unwinds" the second one.
  context.initialization_index = lookup_table.lookup_node(transpose_key);

  context.expanded_new_node = context.initialization_index < 0;
  if (context.expanded_new_node) {
    context.initialization_index = lookup_table.alloc_node();
    Node* child = lookup_table.get_node(context.initialization_index);

    auto result = Rules::analyze(state);
    bool terminal = result.is_terminal();

    // NOTE: for chance events, this should really be entering a different code-path. Right now,
    // we're lucky that for stochastic-nim, Rules::analyze() happens to return a Rules::Result whose
    // valid_moves exactly correspond to the chance outcomes. In general, this might not be the
    // case.
    if (terminal) {
      new (child) Node(lookup_table.get_random_mutex(), state, result.outcome());
      algo_init_node_stats_from_terminal(child);
    } else {
      new (child) Node(lookup_table.get_random_mutex(), state, result.valid_moves().size(),
                       context.active_seat);
      initialize_edges(child, result.valid_moves());
    }

    context.search_path.emplace_back(child, nullptr);
    bool do_virtual = !terminal && multithreaded();
    if (do_virtual) {
      virtual_backprop(context);
    }

    if (begin_node_initialization(context) == core::kYield) return core::kYield;
  }
  return resume_expansion(context);
}

template <alpha0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_expansion(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  LookupTable& lookup_table = general_context_.lookup_table;
  Edge* edge = context.visit_edge;
  Node* parent = context.visit_node;

  if (context.mid_node_initialization) {
    if (resume_node_initialization(context) == core::kYield) return core::kYield;
  }

  if (context.expanded_new_node) {
    Node* child = lookup_table.get_node(context.initialization_index);
    bool terminal = child->is_terminal();
    bool do_virtual = !terminal && multithreaded();

    edge->child_index = context.inserted_node_index;
    if (context.initialization_index != context.inserted_node_index) {
      // This means that we hit the race-condition described in begin_expansion(). We need to
      // "unwind" the second resume_node_initialization() call, and instead use the first one.
      //
      // Note that all the work done in constructing child is effectively discarded. We don't
      // need to explicit undo the alloc_node() call, as the memory will naturally be reclaimed
      // when the lookup_table is defragmented.
      context.search_path.pop_back();
      if (do_virtual) {
        undo_virtual_backprop(context);
      }
      context.expanded_new_node = false;
    } else {
      standard_backprop(context, do_virtual);
    }
  } else {
    edge->child_index = context.initialization_index;
  }

  // TODO: in the !expanded_new_node case, we should check to see if there are sister edges from the
  // same parent that point to the same child. In this case, we can "slide" the visits and
  // policy-mass from one edge to the other, effectively pretending that we had merged the two edges
  // from the beginning. This should result in a more efficient search.

  mit::unique_lock lock(parent->mutex());
  set_edge_state(context, edge, Edge::kExpanded);
  lock.unlock();

  context.mid_expansion = false;
  return core::kContinue;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::virtual_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  algo_backprop(context, last_node, nullptr, [&] { algo_virtually_update_node_stats(last_node); });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    algo_backprop(context, node, edge,
                  [&] { algo_virtually_update_node_stats_and_edge(node, edge); });
  }
  algo_validate_search_path(context);
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::undo_virtual_backprop(SearchContext& context) {
  // NOTE: this is not an exact undo of virtual_backprop(), since the context.search_path is
  // modified in between the two calls.

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  RELEASE_ASSERT(!context.search_path.empty());

  for (int i = context.search_path.size() - 1; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    algo_backprop(context, node, edge, [&] { algo_undo_virtual_update(node, edge); });
  }
  algo_validate_search_path(context);
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::standard_backprop(SearchContext& context, bool undo_virtual) {
  Node* last_node = context.search_path.back().node;
  auto value = last_node->stable_data().V();

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, context.search_path_str(),
             fmt::streamed(value.transpose()));
  }

  algo_backprop(context, last_node, nullptr,
                [&] { algo_update_node_stats(last_node, undo_virtual); });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    algo_backprop(context, node, edge,
                  [&] { algo_update_node_stats_and_edge(node, edge, undo_virtual); });
  }
  algo_validate_search_path(context);
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::short_circuit_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    algo_backprop(context, node, edge, [&] { algo_update_node_stats_and_edge(node, edge, false); });
  }
  algo_validate_search_path(context);
}

template <alpha0::concepts::Spec Spec>
core::node_pool_index_t Manager<Spec>::lookup_child_by_move(const Node* node,
                                                            const Move& move) const {
  // Returns the child node index if found, and -1 otherwise.
  const LookupTable& lookup_table = general_context_.lookup_table;
  int n = node->stable_data().num_valid_moves;

  if constexpr (MoveSet::kSortedByMove) {
    // binary search
    int left = 0;
    int right = n - 1;
    while (left <= right) {
      int mid = left + (right - left) / 2;
      const Edge* edge = lookup_table.get_edge(node, mid);
      if (edge->move == move) {
        return edge->child_index;
      } else if (edge->move < move) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }
    return -1;
  } else {
    // linear search
    for (int i = 0; i < n; i++) {
      const Edge* edge = lookup_table.get_edge(node, i);
      if (edge->move == move) {
        return edge->child_index;
      }
    }
    return -1;
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::initialize_edges(Node* node, const MoveSet& valid_moves) {
  int n_edges = node->stable_data().num_valid_moves;
  RELEASE_ASSERT(n_edges == (int)valid_moves.size());
  if (n_edges == 0) return;

  LookupTable& lookup_table = general_context_.lookup_table;
  node->set_first_edge_index(lookup_table.alloc_edges(n_edges));

  int i = 0;
  for (Move move : valid_moves) {
    Edge* edge = lookup_table.get_edge(node, i);
    new (edge) Edge();
    edge->move = move;
    i++;
  }
}

template <alpha0::concepts::Spec Spec>
bool Manager<Spec>::all_children_edges_initialized(const Node* root) const {
  int n = root->stable_data().num_valid_moves;
  if (n == 0) return true;
  if (root->get_first_edge_index() == -1) return false;

  const LookupTable& lookup_table = general_context_.lookup_table;
  for (int i = 0; i < n; i++) {
    Edge* edge = lookup_table.get_edge(root, i);
    if (edge->state != Edge::kExpanded) {
      return false;
    }
  }
  return true;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::add_pending_notification(SearchContext& context, Edge* edge) {
  // Assumes edge's parent node's mutex is held
  DEBUG_ASSERT(multithreaded());
  DEBUG_ASSERT(edge->expanding_context_id >= 0);
  DEBUG_ASSERT(edge->expanding_context_id != context.id);

  core::SlotContext slot_context(context.search_request->game_slot_index(), context.id);

  SearchContext& notifying_context = contexts_[edge->expanding_context_id];
  mit::mutex& mutex = (*context_mutex_pool_)[notifying_context.pending_notifications_mutex_id];
  mit::unique_lock lock(mutex);
  notifying_context.pending_notifications.push_back(slot_context);
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::set_edge_state(SearchContext& context, Edge* edge,
                                   Edge::expansion_state_t state) {
  LOG_TRACE("{:>{}}{}() state={}", "", context.log_prefix_n(), __func__, state);
  if (state == Edge::kPreExpanded) {
    // Makes no assumptions about mutexes
    edge->state = state;
  } else if (state == Edge::kMidExpansion) {
    // Assumes edge's parent node's mutex is held
    edge->state = state;
    edge->expanding_context_id = context.id;
  } else if (state == Edge::kExpanded) {
    // Assumes edge's parent node's mutex is held
    mit::mutex& mutex = (*context_mutex_pool_)[context.pending_notifications_mutex_id];
    mit::unique_lock lock(mutex);
    edge->state = state;
    edge->expanding_context_id = -1;
    context.search_request->yield_manager()->notify(context.pending_notifications);
    context.pending_notifications.clear();
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::pre_expand_children(SearchContext& context, Node* node) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  LookupTable& lookup_table = general_context_.lookup_table;
  const ManagerParams& manager_params = general_context_.manager_params;

  InputFrame parent_frame = context.input_encoder.current_frame();
  const State& parent_state = root_info()->state;
  RELEASE_ASSERT(parent_state == context.current_state);

  // Evaluate every child of the root node
  //
  // TODO: for games with a large branching factor, we may want to only evaluate a subset of the
  // children.
  int n_moves = node->stable_data().num_valid_moves;
  for (int e = 0; e < n_moves; e++) {
    Edge* edge = lookup_table.get_edge(node, e);
    edge->was_pre_expanded = true;
    if (edge->child_index >= 0) continue;

    Rules::apply(context.current_state, edge->move);
    const State& child_state = context.current_state;

    // compute active-seat as local-variable, so we don't need an undo later
    core::seat_index_t child_active_seat = context.active_seat;
    if (!Rules::is_chance_state(child_state)) {
      child_active_seat = Rules::get_current_player(child_state);
    }

    set_edge_state(context, edge, Edge::kPreExpanded);

    TransposeKey transpose_key = Transposer::key(child_state);
    core::node_pool_index_t child_index = lookup_table.lookup_node(transpose_key);
    if (child_index >= 0) {
      edge->child_index = child_index;
      Rules::backtrack_state(context.current_state, parent_state);
      continue;
    }

    edge->child_index = lookup_table.alloc_node();
    Node* child = lookup_table.get_node(edge->child_index);

    auto result = Rules::analyze(child_state);
    bool terminal = result.is_terminal();

    // NOTE: for chance events, this should really be entering a different code-path. Right now,
    // we're lucky that for stochastic-nim, Rules::analyze() happens to return a Rules::Result whose
    // valid_moves exactly correspond to the chance outcomes. In general, this might not be the
    // case.

    if (terminal) {
      new (child) Node(lookup_table.get_random_mutex(), child_state, result.outcome());
      algo_init_node_stats_from_terminal(child);
    } else {
      new (child) Node(lookup_table.get_random_mutex(), child_state, result.valid_moves().size(),
                       child_active_seat);
      initialize_edges(child, result.valid_moves());
    }
    bool overwrite = false;
    lookup_table.insert_node(transpose_key, edge->child_index, overwrite);

    if (child->is_terminal()) {
      Rules::backtrack_state(context.current_state, parent_state);
      continue;
    }

    group::element_t sym = get_random_symmetry(context.input_encoder, child_state);
    bool incorporate = manager_params.incorporate_sym_into_cache_key;

    context.input_encoder.update(child_state);
    InputFrame child_frame = context.input_encoder.current_frame();
    auto eval_key = context.input_encoder.eval_key();
    context.input_encoder.undo();

    context.eval_request.emplace_back(parent_frame, child, &lookup_table, eval_key,
                                      context.input_encoder, child_frame, sym, incorporate);
    Rules::backtrack_state(context.current_state, parent_state);
  }
  RELEASE_ASSERT(context.current_state == root_info()->state);
}

template <alpha0::concepts::Spec Spec>
int Manager<Spec>::sample_chance_child_index(const SearchContext& context) {
  const LookupTable& lookup_table = general_context_.lookup_table;
  Node* node = context.visit_node;
  int n = node->stable_data().num_valid_moves;
  float chance_dist[n];
  for (int i = 0; i < n; i++) {
    chance_dist[i] = lookup_table.get_edge(node, i)->chance_prob;
  }
  return util::Random::weighted_sample(chance_dist, chance_dist + n);
}

template <alpha0::concepts::Spec Spec>
group::element_t Manager<Spec>::get_random_symmetry(const InputEncoder& input_encoder) const {
  group::element_t sym = group::kIdentity;
  if (general_context_.manager_params.apply_random_symmetries) {
    sym = input_encoder.get_random_symmetry();
  }
  return sym;
}

template <alpha0::concepts::Spec Spec>
group::element_t Manager<Spec>::get_random_symmetry(const InputEncoder& input_encoder,
                                                    const State& next_state) const {
  group::element_t sym = group::kIdentity;
  if (general_context_.manager_params.apply_random_symmetries) {
    sym = input_encoder.get_random_symmetry(next_state);
  }
  return sym;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::apply_move(State& state, InputEncoder& input_encoder, const Move& move) {
  Rules::apply(state, move);
  input_encoder.update(state);
}

// ============================================================================
// Methods moved from alpha0::Algorithms
// ============================================================================

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_print_visit_info(const SearchContext& context) {
  if (search::kEnableSearchDebug) {
    const Node* node = context.visit_node;
    LOG_INFO("{:>{}}visit {} seat={}", "", context.log_prefix_n(), context.search_path_str(),
             node->stable_data().active_seat);
  }
}

template <alpha0::concepts::Spec Spec>
template <typename MutexProtectedFunc>
void Manager<Spec>::algo_backprop(SearchContext& context, Node* node, Edge* edge,
                                  MutexProtectedFunc&& func) {
  mit::unique_lock lock(node->mutex());
  func();
  if (!edge) return;
  NodeStats stats = node->stats();  // copy
  lock.unlock();

  algo_update_stats(stats, node, context.general_context->lookup_table);

  lock.lock();

  // Carefully copy back fields of stats back to node->stats()
  // We don't copy counts, which may have been updated by other threads.
  int RN = node->stats().RN;
  int VN = node->stats().VN;
  node->stats() = stats;
  node->stats().RN = RN;
  node->stats().VN = VN;
  lock.unlock();
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_init_node_stats_from_terminal(Node* node) {
  NodeStats& stats = node->stats();
  RELEASE_ASSERT(stats.RN == 0);
  const ValueArray q = node->stable_data().V();

  stats.Q = q;
  stats.Q_sq = q * q;

  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    stats.provably_winning[p] = q(p) >= GameResultEncoding::kMaxValue;
    stats.provably_losing[p] = q(p) <= GameResultEncoding::kMinValue;
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_update_node_stats(Node* node, bool undo_virtual) {
  auto& stats = node->stats();

  stats.RN++;
  stats.VN -= undo_virtual;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual) {
  auto& stats = node->stats();

  edge->E += !undo_virtual;
  stats.RN++;
  stats.VN -= undo_virtual;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_virtually_update_node_stats(Node* node) {
  node->stats().VN++;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_virtually_update_node_stats_and_edge(Node* node, Edge* edge) {
  edge->E++;
  node->stats().VN++;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_undo_virtual_update(Node* node, Edge* edge) {
  edge->E--;
  node->stats().VN--;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_validate_search_path(const SearchContext& context) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;

  LookupTable& lookup_table = context.general_context->lookup_table;
  int N = context.search_path.size();
  for (int i = N - 1; i >= 0; --i) {
    algo_validate_state(lookup_table, context.search_path[i].node);
  }
}

template <alpha0::concepts::Spec Spec>
bool Manager<Spec>::algo_should_short_circuit(const Edge* edge, const Node* child) {
  int edge_count = edge->E;
  int child_count = child->stats().RN;  // not thread-safe but race-condition is benign
  return edge_count < child_count;
}

template <alpha0::concepts::Spec Spec>
bool Manager<Spec>::algo_more_search_iterations_needed(const GeneralContext& general_context,
                                                       const Node* root) {
  // root->stats() usage here is not thread-safe but this race-condition is benign
  const search::SearchParams& search_params = general_context.search_params;
  if (!search_params.ponder && root->stable_data().num_valid_moves == 1) return false;
  return root->stats().total_count() <= search_params.tree_size_limit;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_init_root_info(GeneralContext& general_context,
                                        search::RootInitPurpose purpose) {
  const ManagerParams& manager_params = general_context.manager_params;
  const search::SearchParams& search_params = general_context.search_params;

  bool add_noise = false;
  switch (purpose) {
    case search::kForStandardSearch: {
      add_noise = search_params.full_search && manager_params.mode == search::kTraining;
      break;
    }
    case search::kToLoadRootActionValues: {
      add_noise = false;
      break;
    }
    default: {
      throw util::Exception("Unknown purpose {}", purpose);
    }
  }

  RootInfo& root_info = general_context.root_info;
  LookupTable& lookup_table = general_context.lookup_table;

  root_info.add_noise = add_noise;
  if (root_info.node_index < 0 || add_noise) {
    root_info.node_index = lookup_table.alloc_node();
    Node* root = lookup_table.get_node(root_info.node_index);

    const State& cur_state = root_info.state;
    core::seat_index_t active_seat = Game::Rules::get_current_player(cur_state);
    RELEASE_ASSERT(active_seat >= 0 && active_seat < Game::Constants::kNumPlayers);
    root_info.active_seat = active_seat;
    auto legal_moves = Game::Rules::analyze(cur_state).valid_moves();
    new (root) Node(lookup_table.get_random_mutex(), cur_state, legal_moves.size(), active_seat);
  }

  if (search::kEnableSearchDebug && purpose == search::kForStandardSearch) {
    IO::print_state(std::cout, root_info.state);
  }
}

template <alpha0::concepts::Spec Spec>
int Manager<Spec>::algo_get_best_child_index(const SearchContext& context) {
  const GeneralContext& general_context = *context.general_context;
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  Node* node = context.visit_node;
  bool is_root = (node == lookup_table.get_node(root_info.node_index));
  PuctCalculator action_selector(lookup_table, manager_params, search_params, node, is_root);

  using PVec = LocalPolicyArray;

  const PVec& P = action_selector.P;
  const PVec& mE = action_selector.mE;
  PVec& PUCT = action_selector.PUCT;

  int argmax_index;

  if (search_params.tree_size_limit == 1) {
    // net-only, use P
    P.maxCoeff(&argmax_index);
  } else {
    bool force_playouts = manager_params.forced_playouts && is_root && search_params.full_search &&
                          manager_params.dirichlet_mult > 0;

    if (force_playouts) {
      PVec n_forced = (P * manager_params.k_forced * mE.sum()).sqrt();
      auto F1 = (mE < n_forced).template cast<float>();
      auto F2 = (mE > 0).template cast<float>();
      auto F = F1 * F2;
      PUCT = PUCT * (1 - F) + F * 1e+6;
    }

    PUCT.maxCoeff(&argmax_index);
  }

  algo_print_action_selection_details(context, action_selector, argmax_index);
  return argmax_index;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_load_evaluations(SearchContext& context) {
  const LookupTable& lookup_table = context.general_context->lookup_table;
  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());

    auto& stable_data = node->stable_data();
    auto& stats = node->stats();

    int n = stable_data.num_valid_moves;
    GameResultTensor R;

    LocalPolicyArray P_raw(n);
    LocalActionValueArray AV(n, Game::Constants::kNumPlayers);

    auto eval = item.eval();

    using NetworkHeadsList = Spec::NetworkHeads::List;
    using Head0 = mp::TypeAt_t<NetworkHeadsList, 0>;
    using Head1 = mp::TypeAt_t<NetworkHeadsList, 1>;
    using Head2 = mp::TypeAt_t<NetworkHeadsList, 2>;

    static_assert(util::str_equal<Head0::kName, "policy">());
    static_assert(util::str_equal<Head1::kName, "value">());
    static_assert(util::str_equal<Head2::kName, "action_value">());

    std::copy_n(eval->data(0), P_raw.size(), P_raw.data());
    std::copy_n(eval->data(1), R.size(), R.data());
    std::copy_n(eval->data(2), AV.size(), AV.data());

    RELEASE_ASSERT(eigen_util::isfinite(P_raw), "Non-finite values in policy head");
    RELEASE_ASSERT(eigen_util::isfinite(R), "Non-finite values in value head");
    RELEASE_ASSERT(eigen_util::isfinite(AV), "Non-finite values in action value head");

    LocalPolicyArray P_adjusted = P_raw;
    algo_transform_policy(context, P_adjusted);

    stable_data.R = R;
    stable_data.R_valid = true;

    // No need to worry about thread-safety when modifying edges or stats below, since no other
    // threads can access this node until after load_eval() returns
    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->policy_prior_prob = P_raw[i];
      edge->adjusted_base_prob = P_adjusted[i];
      edge->child_AV = AV.row(i);
    }

    ValueArray V = GameResultEncoding::to_value_array(R);
    stats.Q = V;
    stats.Q_sq = V * V;
  }

  const RootInfo& root_info = context.general_context->root_info;
  Node* root = lookup_table.get_node(root_info.node_index);
  if (root) {
    root->stats().RN = std::max(root->stats().RN, 1);
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_to_results(const GeneralContext& general_context, SearchResults& results) {
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  const Node* root = lookup_table.get_node(root_info.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();  // thread-safe since single-threaded here
  const State& state = root_info.state;

  results.valid_moves = Game::Rules::analyze(state).valid_moves();
  results.frame = root_info.input_encoder.current_frame();
  results.P.setZero();
  results.pre_expanded_moves.setZero();

  RELEASE_ASSERT((int)results.valid_moves.size() == stable_data.num_valid_moves, "{} != {}",
                 results.valid_moves.size(), stable_data.num_valid_moves);

  int i = 0;
  for (Move move : results.valid_moves) {
    auto* edge = lookup_table.get_edge(root, i);
    auto index = PolicyEncoding::to_index(results.frame, move);
    results.P.coeffRef(index) = edge->policy_prior_prob;
    results.pre_expanded_moves.coeffRef(index) = edge->was_pre_expanded;

    i++;
  }

  algo_load_action_symmetries(general_context, root, results);
  algo_write_results(general_context, root, results);
  results.policy_target = results.counts;
  results.provably_lost = stats.provably_losing[stable_data.active_seat];
  if (manager_params.forced_playouts && root_info.add_noise) {
    algo_prune_policy_target(general_context, results);
  }

  results.Q = stats.Q;
  results.R = stable_data.R;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_update_stats(NodeStats& stats, const Node* node,
                                      LookupTable& lookup_table) {
  ValueArray Q_sum;
  ValueArray Q_sq_sum;
  Q_sum.setZero();
  Q_sq_sum.setZero();

  player_bitset_t all_provably_winning;
  player_bitset_t all_provably_losing;
  all_provably_winning.set();
  all_provably_losing.set();

  const auto& stable_data = node->stable_data();

  int num_valid_moves = stable_data.num_valid_moves;
  core::seat_index_t seat = stable_data.active_seat;

  if (stable_data.is_chance_node) {
    int num_expanded_edges = 0;
    for (int i = 0; i < num_valid_moves; i++) {
      const Edge* edge = lookup_table.get_edge(node, i);
      const Node* child = lookup_table.get_node(edge->child_index);

      if (!child) {
        break;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      Q_sum += child_stats.Q * edge->chance_prob;
      Q_sq_sum += child_stats.Q_sq * edge->chance_prob;
      num_expanded_edges++;

      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
    }
    if (num_expanded_edges == num_valid_moves) {
      stats.Q = Q_sum;
      stats.Q_sq = Q_sq_sum;
      stats.provably_winning = all_provably_winning;
      stats.provably_losing = all_provably_losing;
    }
    return;
  } else {
    // provably winning/losing calculation
    bool cp_has_winning_move = false;
    int num_expanded_edges = 0;
    int N = 0;

    DEBUG_ASSERT(num_valid_moves > 0);
    for (int i = 0; i < num_valid_moves; i++) {
      const Edge* edge = lookup_table.get_edge(node, i);
      const Node* child = lookup_table.get_node(edge->child_index);
      if (!child) {
        continue;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      if (child_stats.RN > 0) {
        int e = edge->E;
        N += e;
        Q_sum += child_stats.Q * e;
        Q_sq_sum += child_stats.Q_sq * e;
        eigen_util::debug_assert_is_valid_prob_distr(child_stats.Q);
      }

      cp_has_winning_move |= child_stats.provably_winning[seat];
      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
      num_expanded_edges++;
    }

    bool all_edges_expanded = (num_expanded_edges == num_valid_moves);
    if (!all_edges_expanded) {
      all_provably_winning.reset();
      all_provably_losing.reset();
    }

    DEBUG_ASSERT(stable_data.R_valid);
    ValueArray V = stable_data.V();
    Q_sum += V;
    Q_sq_sum += V * V;
    N++;
    eigen_util::debug_assert_is_valid_prob_distr(V);

    auto Q = Q_sum / N;
    auto Q_sq = Q_sq_sum / N;

    stats.Q = Q;
    stats.Q_sq = Q_sq;
    eigen_util::debug_assert_is_valid_prob_distr(stats.Q);
    if (cp_has_winning_move) {
      stats.provably_winning[seat] = true;
      stats.provably_losing.set();
      stats.provably_losing[seat] = false;
    } else if (all_edges_expanded) {
      stats.provably_winning = all_provably_winning;
      stats.provably_losing = all_provably_losing;
    }
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_write_results(const GeneralContext& general_context, const Node* root,
                                       SearchResults& results) {
  // This should only be called in contexts where the search-threads are inactive, so we do not need
  // to worry about thread-safety

  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& params = general_context.manager_params;

  core::seat_index_t seat = root->stable_data().active_seat;
  DEBUG_ASSERT(seat >= 0 && seat < kNumPlayers);

  const auto& frame = results.frame;
  auto& counts = results.counts;
  auto& AV = results.AV;
  auto& AQs = results.AQs;
  auto& AQs_sq = results.AQs_sq;

  counts.setZero();
  AV.setZero();
  AQs.setZero();
  AQs_sq.setZero();

  const auto& parent_stats = root->stats();  // thread-safe because single-threaded here

  bool provably_winning = parent_stats.provably_winning[seat];
  bool provably_losing = parent_stats.provably_losing[seat];

  for (int i = 0; i < root->stable_data().num_valid_moves; i++) {
    const Edge* edge = lookup_table.get_edge(root, i);
    Move move = edge->move;
    auto index = PolicyEncoding::to_index(frame, move);

    int count = edge->E;
    int modified_count = count;

    const Node* child = lookup_table.get_node(edge->child_index);
    if (!child) continue;

    const auto& child_stats = child->stats();  // thread-safe because single-threaded here
    if (params.avoid_proven_losers && !provably_losing && child_stats.provably_losing[seat]) {
      modified_count = 0;
    } else if (params.exploit_proven_winners && provably_winning &&
               !child_stats.provably_winning[seat]) {
      modified_count = 0;
    }

    if (modified_count) {
      counts.coeffRef(index) = modified_count;
      AQs.coeffRef(index) = child_stats.Q(seat);
      AQs_sq.coeffRef(index) = child_stats.Q_sq(seat);
    }

    const auto& child_stable_data = child->stable_data();
    RELEASE_ASSERT(child_stable_data.R_valid);
    ValueArray V = child_stable_data.V();
    eigen_util::chip_assign(AV, eigen_util::reinterpret_as_tensor(V), index);
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_validate_state(LookupTable& lookup_table, Node* node) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  if (node->is_terminal()) return;

  mit::unique_lock lock(node->mutex());

  int N = 1;
  for (int i = 0; i < node->stable_data().num_valid_moves; ++i) {
    auto edge = lookup_table.get_edge(node, i);
    N += edge->E;
    DEBUG_ASSERT(edge->E >= 0);
  }

  const auto stats_copy = node->stats();  // thread-safe because we hold the mutex
  lock.unlock();

  DEBUG_ASSERT(N == stats_copy.RN + stats_copy.VN, "[{}] {} != {} + {}", (void*)node, N,
               stats_copy.RN, stats_copy.VN);
  DEBUG_ASSERT(stats_copy.RN >= 0);
  DEBUG_ASSERT(stats_copy.VN >= 0);
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_transform_policy(SearchContext& context, LocalPolicyArray& P) {
  core::node_pool_index_t index = context.initialization_index;
  GeneralContext& general_context = *context.general_context;
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const ManagerParams& manager_params = general_context.manager_params;

  if (index == root_info.node_index) {
    if (search_params.full_search) {
      if (manager_params.dirichlet_mult) {
        algo_add_dirichlet_noise(general_context, P);
      }
      float temp = general_context.aux_state.root_softmax_temperature.value();
      if (temp > 0.0f) {
        P = P.pow(1.0f / temp);
      }
      P /= P.sum();
    }
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_add_dirichlet_noise(GeneralContext& general_context, LocalPolicyArray& P) {
  const ManagerParams& manager_params = general_context.manager_params;
  auto& dirichlet_gen = general_context.aux_state.dirichlet_gen;
  auto& rng = general_context.aux_state.rng;

  int n = P.rows();
  double alpha = manager_params.dirichlet_alpha_factor / sqrt(n);
  LocalPolicyArray noise = dirichlet_gen.template generate<LocalPolicyArray>(rng, alpha, n);
  P = (1.0 - manager_params.dirichlet_mult) * P + manager_params.dirichlet_mult * noise;
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_prune_policy_target(const GeneralContext& general_context,
                                             SearchResults& results) {
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  if (manager_params.no_model) return;

  const auto& frame = results.frame;
  const Node* root = lookup_table.get_node(root_info.node_index);
  PuctCalculator action_selector(lookup_table, manager_params, search_params, root, true);

  const auto& P = action_selector.P;
  const auto& E = action_selector.E;
  const auto& PW = action_selector.PW;
  const auto& PL = action_selector.PL;
  const auto& mE = action_selector.mE;
  const auto& Q = action_selector.Q;
  const auto& PUCT = action_selector.PUCT;

  auto mE_sum = mE.sum();
  auto n_forced = (P * manager_params.k_forced * mE_sum).sqrt();

  int mE_max_index;
  auto mE_max = mE.maxCoeff(&mE_max_index);

  auto PUCT_max = PUCT(mE_max_index);
  auto sqrt_mE = sqrt(mE_sum + PuctCalculator::eps);
  auto denom = PUCT_max - 2 * Q;

  LocalPolicyArray mE_floor = manager_params.cPUCT * P * sqrt_mE / denom - 1;

  int n_moves = root->stable_data().num_valid_moves;
  for (int i = 0; i < n_moves; ++i) {
    const Edge* edge = lookup_table.get_edge(root, i);
    const Move& move = edge->move;
    auto index = PolicyEncoding::to_index(frame, move);
    if (mE(i) == 0) {
      results.policy_target.coeffRef(index) = 0;
      continue;
    }
    if (mE(i) == mE_max) continue;
    if (denom(i) == 0) continue;
    if (mE_floor(i) >= mE(i)) continue;
    auto n = std::max(mE_floor(i), mE(i) - n_forced(i));
    if (n <= 1.0) {
      n = 0;
    }
    results.policy_target.coeffRef(index) = n;
  }

  if (eigen_util::sum(results.policy_target) <= 0) {
    // can happen in certain edge cases
    results.policy_target = results.counts;
  }

  if (search::kEnableSearchDebug) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    LocalPolicyArray pruned(n_moves);

    ActionPrinter printer(lookup_table.get_moves(root));
    for (int i = 0; i < n_moves; ++i) {
      const Edge* edge = lookup_table.get_edge(root, i);
      Move move = edge->move;
      auto index = PolicyEncoding::to_index(frame, move);

      pruned(i) = results.policy_target.coeff(index);
    }

    LocalPolicyArray actions = printer.flat_array();
    LocalPolicyArray target = pruned / pruned.sum();

    static std::vector<std::string> columns = {"action", "P",  "Q",  "PUCT",   "E",
                                               "PW",     "PL", "mE", "pruned", "target"};
    auto data = eigen_util::sort_rows(
      eigen_util::concatenate_columns(actions, P, Q, PUCT, E, PW, PL, mE, pruned, target));

    eigen_util::PrintArrayFormatMap fmt_map;
    printer.update_format_map(fmt_map);

    std::cout << std::endl << "Policy target pruning:" << std::endl;
    eigen_util::print_array(std::cout, data, columns, &fmt_map);
#pragma GCC diagnostic pop
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_print_action_selection_details(const SearchContext& context,
                                                        const PuctCalculator& selector,
                                                        int argmax_index) {
  LookupTable& lookup_table = context.general_context->lookup_table;
  Node* node = context.visit_node;
  if (search::kEnableSearchDebug) {
    std::ostringstream ss;

    core::seat_index_t seat = node->stable_data().active_seat;

    int n_moves = node->stable_data().num_valid_moves;

    ValueArray players;
    ValueArray nQ = node->stats().Q;
    ValueArray CP;
    for (int p = 0; p < kNumPlayers; ++p) {
      players(p) = p;
      CP(p) = p == seat;
    }

    static std::vector<std::string> player_columns = {"Seat", "Q", "CurP"};
    auto player_data = eigen_util::concatenate_columns(players, nQ, CP);

    eigen_util::PrintArrayFormatMap fmt_map1{
      {"Seat", [&](float x, int) { return std::to_string(int(x)); }},
      {"CurP", [&](float x, int) { return std::string(x ? "*" : ""); }},
    };

    eigen_util::print_array(ss, player_data, player_columns, &fmt_map1);

    const LocalPolicyArray& P = selector.P;
    const LocalPolicyArray& Q = selector.Q;
    const LocalPolicyArray& FPU = selector.FPU;
    const LocalPolicyArray& PW = selector.PW;
    const LocalPolicyArray& PL = selector.PL;
    const LocalPolicyArray& E = selector.E;
    const LocalPolicyArray& mE = selector.mE;
    const LocalPolicyArray& RN = selector.RN;
    const LocalPolicyArray& VN = selector.VN;
    const LocalPolicyArray& PUCT = selector.PUCT;

    LocalPolicyArray child_addr(n_moves);
    LocalPolicyArray argmax(n_moves);
    child_addr.setConstant(-1);
    argmax.setZero();
    argmax(argmax_index) = 1;

    ActionPrinter printer(lookup_table.get_moves(node));
    for (int i = 0; i < n_moves; ++i) {
      const Edge* edge = lookup_table.get_edge(node, i);
      child_addr(i) = edge->child_index;
    }

    LocalPolicyArray actions = printer.flat_array();

    static std::vector<std::string> action_columns = {
      "action", "P", "Q", "FPU", "PW", "PL", "E", "mE", "RN", "VN", "&ch", "PUCT", "argmax"};
    auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
      actions, P, Q, FPU, PW, PL, E, mE, RN, VN, child_addr, PUCT, argmax));

    eigen_util::PrintArrayFormatMap fmt_map2{
      {"&ch", [](float x, int) { return x < 0 ? std::string() : std::to_string((int)x); }},
      {"argmax", [](float x, int) { return std::string(x == 0 ? "" : "*"); }},
    };
    printer.update_format_map(fmt_map2);

    eigen_util::print_array(ss, action_data, action_columns, &fmt_map2);
    util::Logging::multi_line_log_info(ss.str(), context.log_prefix_n());
  }
}

template <alpha0::concepts::Spec Spec>
void Manager<Spec>::algo_load_action_symmetries(const GeneralContext& general_context,
                                                const Node* root, SearchResults& results) {
  const auto& stable_data = root->stable_data();
  const LookupTable& lookup_table = general_context.lookup_table;
  const State& root_state = general_context.root_info.state;

  using Item = ActionSymmetryTable::Item;
  std::vector<Item> items;
  items.reserve(stable_data.num_valid_moves);

  using equivalence_class_t = int;
  using map_t = std::unordered_map<InputFrame, equivalence_class_t>;
  map_t map;

  State state = root_state;  // copy
  for (int e = 0; e < stable_data.num_valid_moves; ++e) {
    Edge* edge = lookup_table.get_edge(root, e);
    Game::Rules::apply(state, edge->move);
    InputFrame frame(state);
    group::element_t sym = Symmetries::get_canonical_symmetry(frame);
    Symmetries::apply(frame, sym);

    auto [it, inserted] = map.try_emplace(frame, map.size());
    items.emplace_back(it->second, edge->move);
    Game::Rules::backtrack_state(state, root_state);
  }

  results.action_symmetry_table.load(items);
  results.trivial = (map.size() <= 1);
}

}  // namespace alpha0
