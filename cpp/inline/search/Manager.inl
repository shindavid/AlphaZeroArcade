#include "search/Manager.hpp"

#include "core/BasicTypes.hpp"
#include "search/Constants.hpp"
#include "search/SearchParams.hpp"
#include "util/Asserts.hpp"
#include "util/Exceptions.hpp"
#include "util/FiniteGroups.hpp"
#include "util/LoggingUtil.hpp"

#include <boost/filesystem.hpp>
#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

namespace search {

template <search::concepts::Traits Traits>
Manager<Traits>::Manager(bool dummy, core::mutex_vec_sptr_t node_mutex_pool,
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

template <search::concepts::Traits Traits>
Manager<Traits>::Manager(const ManagerParams& params, core::GameServerBase* server,
                         EvalServiceBase_sptr service)
    : Manager(true, std::make_shared<core::mutex_vec_t>(1), std::make_shared<core::mutex_vec_t>(1),
              params, server, service) {}

template <search::concepts::Traits Traits>
Manager<Traits>::Manager(core::mutex_vec_sptr_t& node_mutex_pool,
                         core::mutex_vec_sptr_t& context_mutex_pool, const ManagerParams& params,
                         core::GameServerBase* server, EvalServiceBase_sptr service)
    : Manager(true, node_mutex_pool, context_mutex_pool, params, server, service) {}

template <search::concepts::Traits Traits>
inline Manager<Traits>::~Manager() {
  clear();
  nn_eval_service_->disconnect();
}

template <search::concepts::Traits Traits>
inline void Manager<Traits>::start() {
  clear();

  if (!connected_) {
    nn_eval_service_->connect();
    connected_ = true;
  }
}

template <search::concepts::Traits Traits>
void Manager<Traits>::clear() {
  general_context_.clear();
}

template <search::concepts::Traits Traits>
void Manager<Traits>::receive_state_change(core::seat_index_t, const State&,
                                           core::action_t action) {
  update(action);
}

template <search::concepts::Traits Traits>
void Manager<Traits>::backtrack(StateIterator it, core::step_t step) {
  general_context_.jump_to(it, step);
  const State& state = root_info()->input_tensorizor.current_state();
  TransposeKey key = Keys::transpose_key(state);
  core::node_pool_index_t node_index = lookup_table()->lookup_node(key);
  root_info()->node_index = node_index;
}

template <search::concepts::Traits Traits>
void Manager<Traits>::update(core::action_t action) {
  root_info()->input_tensorizor.apply_action(action);
  general_context_.step();

  core::node_pool_index_t root_index = root_info()->node_index;
  if (root_index < 0) return;

  Node* root = lookup_table()->get_node(root_index);
  root_info()->node_index = lookup_child_by_action(root, action);  // tree reuse
}

template <search::concepts::Traits Traits>
void Manager<Traits>::set_search_params(const SearchParams& params) {
  general_context_.search_params = params;
}

template <search::concepts::Traits Traits>
typename Manager<Traits>::SearchResponse Manager<Traits>::search(const SearchRequest& request) {
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
 *
 * TODO: dispatch to Algorithms:: here, since different paradigms want to fill in training_info
 * differently.
 */
template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::load_root_action_values(
  const ChanceEventHandleRequest& chance_request, core::seat_index_t seat,
  TrainingInfo& training_info) {
  if (!mid_load_root_action_values_) {
    Algorithms::init_root_info(general_context_, kToLoadRootActionValues);

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

  core::action_mode_t mode = root->action_mode();

  RELEASE_ASSERT(Rules::is_chance_mode(mode));

  ActionValueTensor& action_values = training_info.action_values_target;
  action_values.setZero();

  int i = 0;
  for (core::action_t action : stable_data.valid_action_mask.on_indices()) {
    auto* edge = lookup_table()->get_edge(root, i);
    core::node_pool_index_t child_node_index = lookup_child_by_action(root, action);
    ValueArray V;
    if (child_node_index < 0) {
      V = edge->child_AV;
    } else {
      Node* child = lookup_table()->get_node(child_node_index);
      V = child->stable_data().V();
    }
    action_values.chip(action, 0) = eigen_util::reinterpret_as_tensor(V);
    i++;
  }

  training_info.state = chance_request.state;
  training_info.action = chance_request.chance_action;
  training_info.use_for_training = true;
  training_info.active_seat = seat;
  training_info.action_values_target_valid = true;

  mid_load_root_action_values_ = false;
  return core::kContinue;
}

template <search::concepts::Traits Traits>
typename Manager<Traits>::SearchResponse Manager<Traits>::search_helper(
  const SearchRequest& request) {
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
  while (Algorithms::more_search_iterations_needed(general_context_, root)) {
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

  Algorithms::to_results(general_context_, results_);
  return SearchResponse(&results_);
}

template <search::concepts::Traits Traits>
int Manager<Traits>::update_state_machine_to_in_visit_loop(SearchContext& context) {
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

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::mark_as_done_with_visit_loop(SearchContext& context,
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

template <search::concepts::Traits Traits>
void Manager<Traits>::init_context(core::context_id_t i) {
  SearchContext& context = contexts_[i];
  context.id = i;
  context.general_context = &general_context_;

  int n = context_mutex_pool_->size();
  if (n > 1) {
    context.pending_notifications_mutex_id = util::Random::uniform_sample(0, n);
  }
}

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::begin_root_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  RootInfo& root_info = general_context_.root_info;
  LookupTable& lookup_table = general_context_.lookup_table;

  Algorithms::init_root_info(general_context_, kForStandardSearch);

  core::node_pool_index_t root_index = root_info.node_index;
  Node* root = lookup_table.get_node(root_index);
  if (root->is_terminal()) return core::kContinue;

  if (!root->edges_initialized()) {
    initialize_edges(root);
  }

  Algorithms::init_root_edges(general_context_);

  if (all_children_edges_initialized(root)) {
    return core::kContinue;
  }

  context.input_tensorizor = root_info.input_tensorizor;
  context.active_seat = root_info.active_seat;
  context.initialization_index = root_index;
  return begin_node_initialization(context);
}

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::resume_root_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  return resume_node_initialization(context);
}

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::begin_node_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  const SearchParams& search_params = general_context_.search_params;
  const RootInfo& root_info = general_context_.root_info;
  LookupTable& lookup_table = general_context_.lookup_table;
  const ManagerParams& manager_params = general_context_.manager_params;

  InputTensorizor& input_tensorizor = context.input_tensorizor;

  core::node_pool_index_t node_index = context.initialization_index;
  Node* node = lookup_table.get_node(node_index);

  context.mid_node_initialization = true;
  RELEASE_ASSERT(context.eval_request.num_fresh_items() == 0);

  bool is_root = (node_index == root_info.node_index);
  if (!node->is_terminal()) {
    bool eval_all_children =
      manager_params.force_evaluate_all_root_children && is_root && search_params.full_search;

    if (!node->stable_data().R_valid) {
      group::element_t sym = get_random_symmetry(input_tensorizor);
      bool incorporate = manager_params.incorporate_sym_into_cache_key;
      context.eval_request.emplace_back(node, input_tensorizor, sym, incorporate);
    }
    if (eval_all_children) {
      expand_all_children(context, node);
    }

    const SearchRequest& search_request = *context.search_request;
    context.eval_request.set_notification_task_info(search_request.notification_unit);

    if (nn_eval_service_->evaluate(context.eval_request) == core::kYield) return core::kYield;
  }

  return resume_node_initialization(context);
}

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::resume_node_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  const RootInfo& root_info = general_context_.root_info;
  LookupTable& lookup_table = general_context_.lookup_table;

  const State& state = context.input_tensorizor.current_state();
  core::node_pool_index_t node_index = context.initialization_index;
  Node* node = lookup_table.get_node(node_index);
  bool is_root = (node_index == root_info.node_index);

  Algorithms::load_evaluations(context);
  context.eval_request.mark_all_as_stale();

  if (!node->is_terminal() && node->stable_data().is_chance_node) {
    ChanceDistribution chance_dist = Rules::get_chance_distribution(state);
    for (int i = 0; i < node->stable_data().num_valid_actions; i++) {
      Edge* edge = lookup_table.get_edge(node, i);
      core::action_t action = edge->action;
      edge->chance_prob = chance_dist(action);
    }
  }

  auto transpose_key = Keys::transpose_key(state);
  bool overwrite = is_root;
  context.inserted_node_index = lookup_table.insert_node(transpose_key, node_index, overwrite);
  context.mid_node_initialization = false;
  return core::kContinue;
}

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::begin_search_iteration(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  const RootInfo& root_info = general_context_.root_info;
  LookupTable& lookup_table = general_context_.lookup_table;

  Node* root = lookup_table.get_node(root_info.node_index);

  context.input_tensorizor = root_info.input_tensorizor;
  context.active_seat = root_info.active_seat;
  context.search_path.clear();
  context.search_path.emplace_back(root, nullptr);
  context.visit_node = root;
  context.mid_search_iteration = true;

  return resume_search_iteration(context);
}

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::resume_search_iteration(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  const RootInfo& root_info = general_context_.root_info;

  if (context.mid_visit) {
    if (resume_visit(context) == core::kYield) return core::kYield;
  }

  while (context.visit_node) {
    if (begin_visit(context) == core::kYield) return core::kYield;
  }

  context.input_tensorizor = root_info.input_tensorizor;
  context.active_seat = root_info.active_seat;
  if (post_visit_func_) post_visit_func_();
  context.mid_search_iteration = false;
  return core::kContinue;
}

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::begin_visit(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  LookupTable& lookup_table = general_context_.lookup_table;

  Node* node = context.visit_node;
  Algorithms::print_visit_info(context);
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
    child_index = Algorithms::get_best_child_index(context);
  }

  Edge* edge = lookup_table.get_edge(node, child_index);
  context.visit_edge = edge;
  context.search_path.back().edge = edge;
  context.applied_action = false;

  if (edge->state != Edge::kExpanded) {
    // reread state under mutex in case of race-condition
    mit::unique_lock lock(node->mutex());

    if (edge->state == Edge::kNotExpanded) {
      set_edge_state(context, edge, Edge::kMidExpansion);
      lock.unlock();

      context.input_tensorizor.apply_action(edge->action);
      const State& leaf_state = context.input_tensorizor.current_state();

      core::action_mode_t child_mode = Rules::get_action_mode(leaf_state);
      if (!Rules::is_chance_mode(child_mode)) {
        context.active_seat = Rules::get_current_player(leaf_state);
      }
      context.applied_action = true;

      if (begin_expansion(context) == core::kYield) return core::kYield;
    } else if (edge->state == Edge::kMidExpansion) {
      add_pending_notification(context, edge);
      return core::kYield;
    } else if (edge->state == Edge::kPreExpanded) {
      set_edge_state(context, edge, Edge::kMidExpansion);
      lock.unlock();

      DEBUG_ASSERT(node == lookup_table.get_node(general_context_.root_info.node_index));
      DEBUG_ASSERT(edge->child_index >= 0);
      Node* child = lookup_table.get_node(edge->child_index);
      context.search_path.emplace_back(child, nullptr);

      if (Algorithms::should_short_circuit(edge, child)) {
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

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::resume_visit(SearchContext& context) {
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

    if (Algorithms::should_short_circuit(edge, child)) {
      short_circuit_backprop(context);
      context.visit_node = nullptr;
      context.mid_visit = false;
      LOG_TRACE("{:>{}}{}() continuing @{}", "", context.log_prefix_n(), __func__, __LINE__);
      return core::kContinue;
    }
  }
  if (!context.applied_action) {
    context.input_tensorizor.apply_action(edge->action);
    const State& state = context.input_tensorizor.current_state();

    core::action_mode_t child_mode = Rules::get_action_mode(state);
    if (!Rules::is_chance_mode(child_mode)) {
      context.active_seat = Rules::get_current_player(state);
    }
  }
  context.visit_node = child;
  context.mid_visit = false;
  LOG_TRACE("{:>{}}{}() continuing @{}", "", context.log_prefix_n(), __func__, __LINE__);
  return core::kContinue;
}

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::begin_expansion(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  LookupTable& lookup_table = general_context_.lookup_table;

  context.mid_expansion = true;

  Node* parent = context.visit_node;
  Edge* edge = context.visit_edge;

  const State& state = context.input_tensorizor.current_state();
  TransposeKey transpose_key = Keys::transpose_key(state);

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

    GameResultTensor game_outcome;
    core::action_t last_action = edge->action;

    bool terminal =
      Rules::is_terminal(state, parent->stable_data().active_seat, last_action, game_outcome);

    if (terminal) {
      new (child) Node(lookup_table.get_random_mutex(), state, game_outcome);
      Algorithms::init_node_stats_from_terminal(child);
    } else {
      new (child) Node(lookup_table.get_random_mutex(), state, context.active_seat);
    }

    context.search_path.emplace_back(child, nullptr);
    initialize_edges(child);
    bool do_virtual = !terminal && multithreaded();
    if (do_virtual) {
      virtual_backprop(context);
    }

    if (begin_node_initialization(context) == core::kYield) return core::kYield;
  }
  return resume_expansion(context);
}

template <search::concepts::Traits Traits>
core::yield_instruction_t Manager<Traits>::resume_expansion(SearchContext& context) {
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

template <search::concepts::Traits Traits>
void Manager<Traits>::virtual_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  Algorithms::backprop(context, last_node, nullptr,
                       [&] { Algorithms::virtually_update_node_stats(last_node); });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    Algorithms::backprop(context, node, edge,
                         [&] { Algorithms::virtually_update_node_stats_and_edge(node, edge); });
  }
  Algorithms::validate_search_path(context);
}

template <search::concepts::Traits Traits>
void Manager<Traits>::undo_virtual_backprop(SearchContext& context) {
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

    Algorithms::backprop(context, node, edge, [&] { Algorithms::undo_virtual_update(node, edge); });
  }
  Algorithms::validate_search_path(context);
}

template <search::concepts::Traits Traits>
void Manager<Traits>::standard_backprop(SearchContext& context, bool undo_virtual) {
  Node* last_node = context.search_path.back().node;
  auto value = last_node->stable_data().V();

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, context.search_path_str(),
             fmt::streamed(value.transpose()));
  }

  Algorithms::backprop(context, last_node, nullptr,
                       [&] { Algorithms::update_node_stats(last_node, undo_virtual); });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    Algorithms::backprop(context, node, edge,
                         [&] { Algorithms::update_node_stats_and_edge(node, edge, undo_virtual); });
  }
  Algorithms::validate_search_path(context);
}

template <search::concepts::Traits Traits>
void Manager<Traits>::short_circuit_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    Algorithms::backprop(context, node, edge,
                         [&] { Algorithms::update_node_stats_and_edge(node, edge, false); });
  }
  Algorithms::validate_search_path(context);
}

template <search::concepts::Traits Traits>
core::node_pool_index_t Manager<Traits>::lookup_child_by_action(const Node* node,
                                                                core::action_t action) const {
  // NOTE: this can be switched to use binary search if we'd like
  const LookupTable& lookup_table = general_context_.lookup_table;
  int i = 0;
  for (core::action_t a : node->stable_data().valid_action_mask.on_indices()) {
    if (a == action) {
      return lookup_table.get_edge(node, i)->child_index;
    }
    ++i;
  }
  return -1;
}

template <search::concepts::Traits Traits>
void Manager<Traits>::initialize_edges(Node* node) {
  int n_edges = node->stable_data().num_valid_actions;
  if (n_edges == 0) return;

  LookupTable& lookup_table = general_context_.lookup_table;
  node->set_first_edge_index(lookup_table.alloc_edges(n_edges));

  int i = 0;
  for (core::action_t action : node->stable_data().valid_action_mask.on_indices()) {
    Edge* edge = lookup_table.get_edge(node, i);
    new (edge) Edge();
    edge->action = action;
    i++;
  }
}

template <search::concepts::Traits Traits>
bool Manager<Traits>::all_children_edges_initialized(const Node* root) const {
  int n = root->stable_data().num_valid_actions;
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

template <search::concepts::Traits Traits>
void Manager<Traits>::add_pending_notification(SearchContext& context, Edge* edge) {
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

template <search::concepts::Traits Traits>
void Manager<Traits>::set_edge_state(SearchContext& context, Edge* edge,
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

template <search::concepts::Traits Traits>
void Manager<Traits>::expand_all_children(SearchContext& context, Node* node) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  LookupTable& lookup_table = general_context_.lookup_table;
  const ManagerParams& manager_params = general_context_.manager_params;

  State parent_state = context.input_tensorizor.current_state(); // make a copy
  // Evaluate every child of the root node
  int n_actions = node->stable_data().num_valid_actions;
  for (int e = 0; e < n_actions; e++) {
    Edge* edge = lookup_table.get_edge(node, e);
    if (edge->child_index >= 0) continue;

    context.input_tensorizor.apply_action(edge->action);
    const State& child_state = context.input_tensorizor.current_state();

    // compute active-seat as local-variable, so we don't need an undo later
    core::action_mode_t child_mode = Rules::get_action_mode(child_state);
    core::seat_index_t child_active_seat = context.active_seat;
    if (!Rules::is_chance_mode(child_mode)) {
      child_active_seat = Rules::get_current_player(child_state);
    }

    set_edge_state(context, edge, Edge::kPreExpanded);

    TransposeKey transpose_key = Keys::transpose_key(child_state);
    core::node_pool_index_t child_index = lookup_table.lookup_node(transpose_key);
    if (child_index >= 0) {
      edge->child_index = child_index;
      context.input_tensorizor.undo(parent_state);
      continue;
    }

    edge->child_index = lookup_table.alloc_node();
    Node* child = lookup_table.get_node(edge->child_index);

    core::seat_index_t parent_active_seat = node->stable_data().active_seat;

    GameResultTensor game_outcome;
    if (Rules::is_terminal(child_state, parent_active_seat, edge->action, game_outcome)) {
      new (child) Node(lookup_table.get_random_mutex(), child_state, game_outcome);
      Algorithms::init_node_stats_from_terminal(child);
    } else {
      new (child) Node(lookup_table.get_random_mutex(), child_state, child_active_seat);
    }
    initialize_edges(child);
    bool overwrite = false;
    lookup_table.insert_node(transpose_key, edge->child_index, overwrite);

    context.input_tensorizor.undo(parent_state);

    if (child->is_terminal()) continue;

    group::element_t sym = get_random_symmetry(context.input_tensorizor);
    bool incorporate = manager_params.incorporate_sym_into_cache_key;
    context.eval_request.emplace_back(child, context.input_tensorizor, child_state, sym, incorporate);
  }
}

template <search::concepts::Traits Traits>
int Manager<Traits>::sample_chance_child_index(const SearchContext& context) {
  const LookupTable& lookup_table = general_context_.lookup_table;
  Node* node = context.visit_node;
  int n = node->stable_data().num_valid_actions;
  float chance_dist[n];
  for (int i = 0; i < n; i++) {
    chance_dist[i] = lookup_table.get_edge(node, i)->chance_prob;
  }
  return util::Random::weighted_sample(chance_dist, chance_dist + n);
}

template <search::concepts::Traits Traits>
group::element_t Manager<Traits>::get_random_symmetry(const InputTensorizor& input_tensorizor) const {
  group::element_t sym = group::kIdentity;
  if (general_context_.manager_params.apply_random_symmetries) {
    sym = input_tensorizor.get_random_symmetry();
  }
  return sym;
}

}  // namespace search
