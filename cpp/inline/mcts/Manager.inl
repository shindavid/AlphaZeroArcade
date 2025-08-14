#include "mcts/Manager.hpp"

#include "core/BasicTypes.hpp"
#include "mcts/TypeDefs.hpp"
#include "mcts/UniformNNEvaluationService.hpp"
#include "util/Asserts.hpp"
#include "util/Exceptions.hpp"
#include "util/LoggingUtil.hpp"

#include <boost/filesystem.hpp>
#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

namespace mcts {

template <core::concepts::Game Game>
int Manager<Game>::next_instance_id_ = 0;

template <core::concepts::Game Game>
Manager<Game>::Manager(bool dummy, mutex_vec_sptr_t node_mutex_pool,
                       mutex_vec_sptr_t context_mutex_pool, const ManagerParams& params,
                       core::GameServerBase* server, NNEvaluationServiceBase_sptr service)
    : params_(params),
      pondering_search_params_(
        SearchParams::make_pondering_params(params.pondering_tree_size_limit)),
      manager_id_(next_instance_id_++),
      lookup_table_(node_mutex_pool),
      root_softmax_temperature_(params.starting_root_softmax_temperature,
                                params.ending_root_softmax_temperature,
                                params.root_softmax_temperature_half_life),
      context_mutex_pool_(context_mutex_pool) {
  if (params_.enable_pondering) {
    throw util::CleanException("Pondering mode temporarily unsupported");
  }

  if (service) {
    nn_eval_service_ = service;
  } else if (!params.no_model) {
    nn_eval_service_ = NNEvaluationService::create(params, server);
  } else if (params.model_filename.empty()) {
    nn_eval_service_ = std::make_shared<UniformNNEvaluationService<Game>>();
  } else {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }

  if (params.enable_pondering) {
    throw util::CleanException("pondering mode temporarily unsupported");
  }
  contexts_.resize(num_search_threads());
  for (int i = 0; i < num_search_threads(); ++i) {
    init_context(i);
  }
}

template <core::concepts::Game Game>
Manager<Game>::Manager(const ManagerParams& params, core::GameServerBase* server,
                       NNEvaluationServiceBase_sptr service)
    : Manager(true, std::make_shared<mutex_vec_t>(1), std::make_shared<mutex_vec_t>(1), params,
              server, service) {}

template <core::concepts::Game Game>
Manager<Game>::Manager(mutex_vec_sptr_t& node_mutex_pool, mutex_vec_sptr_t& context_mutex_pool,
                       const ManagerParams& params, core::GameServerBase* server,
                       NNEvaluationServiceBase_sptr service)
    : Manager(true, node_mutex_pool, context_mutex_pool, params, server, service) {}

template <core::concepts::Game Game>
inline Manager<Game>::~Manager() {
  clear();
  nn_eval_service_->disconnect();
}

template <core::concepts::Game Game>
inline void Manager<Game>::start() {
  clear();

  if (!connected_) {
    nn_eval_service_->connect();
    connected_ = true;
  }
}

template <core::concepts::Game Game>
void Manager<Game>::clear() {
  root_softmax_temperature_.reset();
  lookup_table_.clear();
  root_info_.node_index = -1;

  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    root_info_.history_array[sym].initialize(Rules{});
    State& state = root_info_.history_array[sym].current();
    Symmetries::apply(state, sym);
  }

  const State& raw_state = root_info_.history_array[group::kIdentity].current();
  root_info_.canonical_sym = Symmetries::get_canonical_symmetry(raw_state);
}

template <core::concepts::Game Game>
void Manager<Game>::receive_state_change(core::seat_index_t, const State&, core::action_t action) {
  update(action);
}

template <core::concepts::Game Game>
void Manager<Game>::update(core::action_t action) {
  group::element_t root_sym = root_info_.canonical_sym;

  core::action_mode_t mode =
    Rules::get_action_mode(root_info_.history_array[group::kIdentity].current());

  // TODO: this logic is currently not quite right. It assumes that the symmetry-set is constant,
  // which is not true for games like chess where the symmetries of the game is dependent on the
  // board state.
  //
  // To fix this, I think we need to track the intersection of the symmetry-set across the entire
  // history.
  //
  // I also think that this loop needs to change to symmetrize the states, rather than the action.
  // It makes no sense to symmetrize an action like a king-side-castle into 8 different versions.
  // But we can symmetrize the state that results from the king-side-castle. By respecting the valid
  // symmetry set that comes from the intersection, we won't actually use any of those symmetries
  // that correspond to nonsensical states.
  //
  // As part of this, I think the get_canonical_symmetry() function needs to accept a history,
  // rather than a state. The history object should probably be extended to easily compute the
  // symmetry-set-intersection.
  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    core::action_t transformed_action = action;
    Symmetries::apply(transformed_action, sym, mode);
    Rules::apply(root_info_.history_array[sym], transformed_action);
  }

  const State& raw_state = root_info_.history_array[group::kIdentity].current();
  root_info_.canonical_sym = Symmetries::get_canonical_symmetry(raw_state);

  root_softmax_temperature_.step();
  node_pool_index_t root_index = root_info_.node_index;
  if (root_index < 0) return;

  core::action_t transformed_action = action;
  Symmetries::apply(transformed_action, root_sym, mode);

  Node* root = lookup_table_.get_node(root_index);
  root_info_.node_index = root->lookup_child_by_action(transformed_action);  // tree reuse
}

template <core::concepts::Game Game>
void Manager<Game>::set_search_params(const SearchParams& params) {
  search_params_ = params;
}

template <core::concepts::Game Game>
typename Manager<Game>::SearchResponse Manager<Game>::search(const SearchRequest& request) {
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
template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::load_root_action_values(
  const core::YieldNotificationUnit& notification_unit, ActionValueTensor& action_values) {
  if (!mid_load_root_action_values_) {
    action_values.setZero();
    init_root_info(false);

    // We do a dummy search with 0 iterations, just to get SearchThread to call init_root_node(),
    // which will expand all the root's children.
    constexpr int tree_size_limit = 0;
    constexpr bool full_search = true;
    constexpr bool ponder = false;
    SearchParams params{tree_size_limit, full_search, ponder};
    search_params_ = params;
    mid_load_root_action_values_ = true;
  }

  SearchRequest request(notification_unit);
  SearchResponse response = search(request);
  if (response.yield_instruction == core::kYield) return core::kYield;
  RELEASE_ASSERT(response.yield_instruction == core::kContinue);

  Node* root = lookup_table_.get_node(root_info_.node_index);
  const auto& stable_data = root->stable_data();

  core::action_mode_t mode = root->action_mode();
  group::element_t sym = root_info_.canonical_sym;

  RELEASE_ASSERT(Rules::is_chance_mode(mode));

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(stable_data.valid_action_mask)) {
    auto* edge = root->get_edge(i);
    core::action_t transformed_action = action;
    Symmetries::apply(transformed_action, sym, mode);
    node_pool_index_t child_node_index = root->lookup_child_by_action(transformed_action);
    if (child_node_index < 0) {
      action_values(action) = edge->child_V_estimate;
    } else {
      Node* child = lookup_table_.get_node(child_node_index);
      action_values(action) = child->stable_data().VT(stable_data.active_seat);
    }
    i++;
  }

  mid_load_root_action_values_ = false;
  return core::kContinue;
}

template <core::concepts::Game Game>
typename Manager<Game>::SearchResponse Manager<Game>::search_helper(const SearchRequest& request) {
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

  Node* root = lookup_table_.get_node(root_info_.node_index);
  while (more_search_iterations_needed(root)) {
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
  prepare_results();
  return SearchResponse(&results_);
}

template <core::concepts::Game Game>
int Manager<Game>::update_state_machine_to_in_visit_loop(SearchContext& context) {
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

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::mark_as_done_with_visit_loop(SearchContext& context,
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

template <core::concepts::Game Game>
void Manager<Game>::init_context(core::context_id_t i) {
  SearchContext& context = contexts_[i];
  context.id = i;

  int n = context_mutex_pool_->size();
  if (n > 1) {
    context.pending_notifications_mutex_id = util::Random::uniform_sample(0, n);
  }
}

template <core::concepts::Game Game>
inline void Manager<Game>::init_root_info(bool add_noise) {
  root_info_.add_noise = add_noise;
  if (root_info_.node_index < 0 || add_noise) {
    const StateHistory& canonical_history = root_info_.history_array[root_info_.canonical_sym];
    root_info_.node_index = lookup_table_.alloc_node();
    Node* root = lookup_table_.get_node(root_info_.node_index);
    core::seat_index_t active_seat = Rules::get_current_player(canonical_history.current());
    RELEASE_ASSERT(active_seat >= 0 && active_seat < Constants::kNumPlayers);
    root_info_.active_seat = active_seat;
    new (root) Node(&lookup_table_, canonical_history, active_seat);
  }

  Node* root2 = lookup_table_.get_node(root_info_.node_index);

  // thread-safe since single-threaded here
  if (root2->stats().RN == 0) {
    root2->stats().RN = 1;
  }
}

template <core::concepts::Game Game>
bool Manager<Game>::more_search_iterations_needed(Node* root) {
  // root->stats() usage here is not thread-safe but this race-condition is benign
  if (!search_params_.ponder && root->trivial()) return false;
  return root->stats().total_count() <= search_params_.tree_size_limit;
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::begin_root_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  bool add_noise = search_params_.full_search && params_.dirichlet_mult > 0;
  init_root_info(add_noise);

  if (mcts::kEnableSearchDebug) {
    const auto& state = root_info_.history_array[group::kIdentity].current();
    IO::print_state(std::cout, state);
  }

  node_pool_index_t root_index = root_info_.node_index;
  Node* root = lookup_table_.get_node(root_index);
  if (root->is_terminal()) return core::kContinue;

  if (!root->edges_initialized()) {
    root->initialize_edges();
  }

  if (root->all_children_edges_initialized()) {
    return core::kContinue;
  }

  StateHistory& history = root_info_.history_array[context.canonical_sym];

  context.canonical_sym = root_info_.canonical_sym;
  context.raw_history = root_info_.history_array[group::kIdentity];
  context.active_seat = root_info_.active_seat;
  context.root_history_array = root_info_.history_array;

  context.canonical_history = history;
  context.initialization_history = &context.canonical_history;
  context.initialization_index = root_index;
  return begin_node_initialization(context);
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::resume_root_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  return resume_node_initialization(context);
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::begin_node_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  StateHistory* history = context.initialization_history;
  node_pool_index_t node_index = context.initialization_index;
  Node* node = lookup_table_.get_node(node_index);

  context.mid_node_initialization = true;
  RELEASE_ASSERT(context.eval_request.num_fresh_items() == 0);

  bool is_root = (node_index == root_info_.node_index);
  if (!node->is_terminal()) {
    bool eval_all_children =
      params_.force_evaluate_all_root_children && is_root && search_params_.full_search;

    if (!node->stable_data().VT_valid) {
      group::element_t sym = group::kIdentity;
      if (params_.apply_random_symmetries) {
        sym = bitset_util::choose_random_on_index(Symmetries::get_mask(history->current()));
      }
      bool incorporate = params_.incorporate_sym_into_cache_key;
      context.eval_request.emplace_back(node, *history, sym, incorporate);
    }
    if (eval_all_children) {
      expand_all_children(context, node);
    }

    const SearchRequest& search_request = *context.search_request;
    context.eval_request.set_notification_task_info(search_request.notification_unit);

    if (mcts::kEnableSearchDebug) {
      LOG_INFO("{:>{}}{}() - size: {}", "", context.log_prefix_n(), __func__,
               context.eval_request.num_fresh_items());
    }
    if (nn_eval_service_->evaluate(context.eval_request) == core::kYield) return core::kYield;
  }

  return resume_node_initialization(context);
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::resume_node_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  StateHistory* history = context.initialization_history;
  node_pool_index_t node_index = context.initialization_index;

  Node* node = lookup_table_.get_node(node_index);
  bool is_root = (node_index == root_info_.node_index);

  for (auto& item : context.eval_request.fresh_items()) {
    item.node()->load_eval(item.eval(),
                           [&](LocalPolicyArray& P) { transform_policy(node_index, P); });
  }
  context.eval_request.mark_all_as_stale();

  if (!node->is_terminal() && node->stable_data().is_chance_node) {
    ChanceDistribution chance_dist = Rules::get_chance_distribution(history->current());
    for (int i = 0; i < node->stable_data().num_valid_actions; i++) {
      Edge* edge = node->get_edge(i);
      core::action_t action = edge->action;
      edge->base_prob = chance_dist(action);
    }
  }

  auto mcts_key = InputTensorizor::mcts_key(*history);
  bool overwrite = is_root;
  context.inserted_node_index = lookup_table_.insert_node(mcts_key, node_index, overwrite);
  context.mid_node_initialization = false;
  return core::kContinue;
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::begin_search_iteration(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  Node* root = lookup_table_.get_node(root_info_.node_index);
  context.canonical_sym = root_info_.canonical_sym;
  context.raw_history = root_info_.history_array[group::kIdentity];
  context.active_seat = root_info_.active_seat;
  context.search_path.clear();
  context.search_path.emplace_back(root, nullptr);
  context.visit_node = root;
  context.mid_search_iteration = true;

  return resume_search_iteration(context);
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::resume_search_iteration(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (context.mid_visit) {
    if (resume_visit(context) == core::kYield) return core::kYield;
  }

  while (context.visit_node) {
    if (begin_visit(context) == core::kYield) return core::kYield;
  }

  Node* root = lookup_table_.get_node(root_info_.node_index);
  root->validate_state();
  context.canonical_sym = root_info_.canonical_sym;
  context.raw_history = root_info_.history_array[group::kIdentity];
  context.active_seat = root_info_.active_seat;
  if (post_visit_func_) post_visit_func_();
  context.mid_search_iteration = false;
  return core::kContinue;
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::begin_visit(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  Node* node = context.visit_node;
  print_visit_info(context);
  context.mid_visit = true;
  context.expanded_new_node = false;

  const auto& stable_data = node->stable_data();
  if (stable_data.terminal) {
    pure_backprop(context, GameResults::to_value_array(stable_data.VT));
    context.visit_node = nullptr;
    context.mid_visit = false;
    return core::kContinue;
  }

  int child_index;
  if (stable_data.is_chance_node) {
    child_index = sample_chance_child_index(context);
  } else {
    child_index = get_best_child_index(context);
  }

  Edge* edge = node->get_edge(child_index);
  context.visit_edge = edge;
  context.search_path.back().edge = edge;
  context.applied_action = false;
  context.inv_canonical_sym = SymmetryGroup::inverse(context.canonical_sym);
  if (edge->state != Node::kExpanded) {
    // reread state under mutex in case of race-condition
    mit::unique_lock lock(node->mutex());

    if (edge->state == Node::kNotExpanded) {
      set_edge_state(context, edge, Node::kMidExpansion);
      lock.unlock();

      // reorient edge->action into raw-orientation
      core::action_t edge_action = edge->action;
      Symmetries::apply(edge_action, context.inv_canonical_sym, node->action_mode());

      // apply raw-orientation action to raw-orientation leaf-state
      Rules::apply(context.raw_history, edge_action);

      // determine canonical orientation of new leaf-state
      group::element_t new_sym = Symmetries::get_canonical_symmetry(context.raw_history.current());
      edge->sym = SymmetryGroup::compose(new_sym, context.inv_canonical_sym);

      context.canonical_sym = new_sym;

      core::action_mode_t child_mode = Rules::get_action_mode(context.raw_history.current());
      if (!Rules::is_chance_mode(child_mode)) {
        context.active_seat = Rules::get_current_player(context.raw_history.current());
      }
      context.applied_action = true;

      context.initialization_history = &context.raw_history;
      if (context.canonical_sym != group::kIdentity) {
        calc_canonical_state_data(context);
        context.initialization_history = &context.canonical_history;
      }

      if (begin_expansion(context) == core::kYield) return core::kYield;
    } else if (edge->state == Node::kMidExpansion) {
      add_pending_notification(context, edge);
      return core::kYield;
    } else if (edge->state == Node::kPreExpanded) {
      set_edge_state(context, edge, Node::kMidExpansion);
      lock.unlock();

      DEBUG_ASSERT(edge->child_index >= 0);
      Node* child = lookup_table_.get_node(edge->child_index);
      context.search_path.emplace_back(child, nullptr);
      int edge_count = edge->E;
      int child_count = child->stats().RN;  // not thread-safe but race-condition is benign
      if (edge_count < child_count) {
        short_circuit_backprop(context);
      } else {
        standard_backprop(context, false);
      }

      lock.lock();
      set_edge_state(context, edge, Node::kExpanded);
      context.visit_node = nullptr;
      context.mid_visit = false;
      return core::kContinue;
    }
  }

  return resume_visit(context);
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::resume_visit(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  Node* node = context.visit_node;
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
  RELEASE_ASSERT(edge->state == Node::kExpanded, "Expected edge state to be kExpanded, but got {}",
                 edge->state);

  Node* child = node->get_child(edge);
  if (child) {
    context.search_path.emplace_back(child, nullptr);
    int edge_count = edge->E;
    int child_count = child->stats().RN;  // not thread-safe but race-condition is benign
    if (edge_count < child_count) {
      short_circuit_backprop(context);
      context.visit_node = nullptr;
      context.mid_visit = false;
      LOG_TRACE("{:>{}}{}() continuing @{}", "", context.log_prefix_n(), __func__, __LINE__);
      return core::kContinue;
    }
  }
  if (!context.applied_action) {
    // reorient edge->action into raw-orientation
    core::action_t edge_action = edge->action;
    Symmetries::apply(edge_action, context.inv_canonical_sym, node->action_mode());

    Rules::apply(context.raw_history, edge_action);
    core::action_mode_t child_mode = Rules::get_action_mode(context.raw_history.current());
    if (!Rules::is_chance_mode(child_mode)) {
      context.active_seat = Rules::get_current_player(context.raw_history.current());
    }
    context.canonical_sym = SymmetryGroup::compose(edge->sym, context.canonical_sym);
  }
  context.visit_node = child;
  context.mid_visit = false;
  LOG_TRACE("{:>{}}{}() continuing @{}", "", context.log_prefix_n(), __func__, __LINE__);
  return core::kContinue;
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::begin_expansion(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  context.mid_expansion = true;

  StateHistory* history = context.initialization_history;
  Node* parent = context.visit_node;
  Edge* edge = context.visit_edge;

  MCTSKey mcts_key = InputTensorizor::mcts_key(*history);

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
  context.initialization_index = lookup_table_.lookup_node(mcts_key);

  context.expanded_new_node = context.initialization_index < 0;
  if (context.expanded_new_node) {
    context.initialization_index = lookup_table_.alloc_node();
    Node* child = lookup_table_.get_node(context.initialization_index);

    ValueTensor game_outcome;
    core::action_t last_action = edge->action;
    Symmetries::apply(last_action, edge->sym, parent->action_mode());

    bool terminal = Rules::is_terminal(history->current(), parent->stable_data().active_seat,
                                       last_action, game_outcome);

    if (terminal) {
      new (child) Node(&lookup_table_, *history, game_outcome);
    } else {
      new (child) Node(&lookup_table_, *history, context.active_seat);
    }

    context.search_path.emplace_back(child, nullptr);
    child->initialize_edges();
    bool do_virtual = !terminal && multithreaded();
    if (do_virtual) {
      virtual_backprop(context);
    }

    context.initialization_history = history;
    if (begin_node_initialization(context) == core::kYield) return core::kYield;
  }
  return resume_expansion(context);
}

template <core::concepts::Game Game>
core::yield_instruction_t Manager<Game>::resume_expansion(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  node_pool_index_t child_index = context.initialization_index;
  Edge* edge = context.visit_edge;
  Node* parent = context.visit_node;

  if (context.mid_node_initialization) {
    if (resume_node_initialization(context) == core::kYield) return core::kYield;
  }

  if (context.expanded_new_node) {
    node_pool_index_t inserted_child_index = context.inserted_node_index;
    Node* child = lookup_table_.get_node(child_index);
    bool terminal = child->is_terminal();
    bool do_virtual = !terminal && multithreaded();

    edge->child_index = inserted_child_index;
    if (child_index != inserted_child_index) {
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
      child_index = edge->child_index;
    } else {
      if (terminal) {
        pure_backprop(context, GameResults::to_value_array(child->stable_data().VT));
      } else {
        standard_backprop(context, do_virtual);
      }
    }
  }

  if (!context.expanded_new_node) {
    // TODO: in this case, we should check to see if there are sister edges that point to the same
    // child. In this case, we can "slide" the visits and policy-mass from one edge to the other,
    // effectively pretending that we had merged the two edges from the beginning. This should
    // result in a more efficient search.
    //
    // We had something like this at some point, and for tic-tac-toe, it led to a significant
    // improvement. But that previous implementation was inefficient for large branching factors,
    // as it did the edge-merging up-front. This proposal only attempts edge-merges on-demand,
    // piggy-backing existing MCGS-key-lookups for minimal additional overhead.
    //
    // Some technical notes on this:
    //
    // - At a minimum we want to slide E and adjusted_base_prob, and then mark the edge as defunct,
    //   so that PUCT will not select it.
    // - We can easily mutex-protect the writes, by doing this under the parent's mutex. For the
    //   reads in ActionSelector, we can probably be unsafe. I think a reasonable order would be:
    //
    //   edge1->merged_edge_index = edge2_index;
    //   edge2->adjusted_base_prob += edge1->adjusted_base_prob;
    //   edge1->adjusted_base_prob = 0;
    //   edge2->E += edge1->E;
    //   edge1->E = 0;
    //
    //   We just have to reason carefully about the order of the reads in ActionSelector. Choosing
    //   which edge merges into which edge can also give us more control over possible races, as
    //   ActionSelector iterates over the edges in a specific order. More careful analysis is
    //   needed here.
    //
    //   Wherever we increment an edge->E, we can check, under the parent-mutex, if
    //   edge->merged_edge_index >= 0, and if so, increment the E of the merged edge instead, in
    //   order to make the writes thread-safe.
    edge->child_index = child_index;
  }

  mit::unique_lock lock(parent->mutex());
  parent->update_child_expand_count();
  set_edge_state(context, edge, Node::kExpanded);
  lock.unlock();

  context.mid_expansion = false;
  return core::kContinue;
}

template <core::concepts::Game Game>
void Manager<Game>::add_pending_notification(SearchContext& context, Edge* edge) {
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

template <core::concepts::Game Game>
void Manager<Game>::set_edge_state(SearchContext& context, Edge* edge, expansion_state_t state) {
  LOG_TRACE("{:>{}}{}() state={}", "", context.log_prefix_n(), __func__, state);
  if (state == Node::kPreExpanded) {
    // Makes no assumptions about mutexes
    edge->state = state;
  } else if (state == Node::kMidExpansion) {
    // Assumes edge's parent node's mutex is held
    edge->state = state;
    edge->expanding_context_id = context.id;
  } else if (state == Node::kExpanded) {
    // Assumes edge's parent node's mutex is held
    mit::mutex& mutex = (*context_mutex_pool_)[context.pending_notifications_mutex_id];
    mit::unique_lock lock(mutex);
    edge->state = state;
    edge->expanding_context_id = -1;
    context.search_request->yield_manager()->notify(context.pending_notifications);
    context.pending_notifications.clear();
  }
}

template <core::concepts::Game Game>
void Manager<Game>::transform_policy(node_pool_index_t index, LocalPolicyArray& P) const {
  if (index == root_info_.node_index) {
    if (search_params_.full_search) {
      if (params_.dirichlet_mult) {
        add_dirichlet_noise(P);
      }
      P = P.pow(1.0 / root_softmax_temperature_.value());
      P /= P.sum();
    }
  }
}

template <core::concepts::Game Game>
inline void Manager<Game>::add_dirichlet_noise(LocalPolicyArray& P) const {
  int n = P.rows();
  double alpha = params_.dirichlet_alpha_factor / sqrt(n);
  LocalPolicyArray noise = dirichlet_gen_.template generate<LocalPolicyArray>(rng_, alpha, n);
  P = (1.0 - params_.dirichlet_mult) * P + params_.dirichlet_mult * noise;
}

template <core::concepts::Game Game>
void Manager<Game>::expand_all_children(SearchContext& context, Node* node) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  group::element_t inv_canonical_sym = SymmetryGroup::inverse(context.canonical_sym);

  // Evaluate every child of the root node
  int n_actions = node->stable_data().num_valid_actions;
  int expand_count = 0;
  for (int e = 0; e < n_actions; e++) {
    Edge* edge = node->get_edge(e);
    if (edge->child_index >= 0) continue;

    // reorient edge->action into raw-orientation
    core::action_t raw_edge_action = edge->action;
    Symmetries::apply(raw_edge_action, inv_canonical_sym, node->action_mode());

    // apply raw-orientation action to raw-orientation child-state
    Rules::apply(context.raw_history, raw_edge_action);

    const State& raw_child_state = context.raw_history.current();

    // compute active-seat as local-variable, so we don't need an undo later
    core::action_mode_t child_mode = Rules::get_action_mode(raw_child_state);
    core::seat_index_t child_active_seat = context.active_seat;
    if (!Rules::is_chance_mode(child_mode)) {
      child_active_seat = Rules::get_current_player(raw_child_state);
    }

    // determine canonical orientation of new leaf-state
    group::element_t canonical_child_sym = Symmetries::get_canonical_symmetry(raw_child_state);
    edge->sym = SymmetryGroup::compose(canonical_child_sym, inv_canonical_sym);

    StateHistory& canonical_history = context.root_history_array[canonical_child_sym];

    core::action_t reoriented_action = raw_edge_action;
    Symmetries::apply(reoriented_action, canonical_child_sym, node->action_mode());
    Rules::apply(canonical_history, reoriented_action);

    expand_count++;
    set_edge_state(context, edge, Node::kPreExpanded);

    MCTSKey mcts_key = InputTensorizor::mcts_key(canonical_history);
    node_pool_index_t child_index = lookup_table_.lookup_node(mcts_key);
    if (child_index >= 0) {
      edge->child_index = child_index;
      canonical_history.undo();
      context.raw_history.undo();
      continue;
    }

    edge->child_index = lookup_table_.alloc_node();
    Node* child = lookup_table_.get_node(edge->child_index);

    core::seat_index_t parent_active_seat = node->stable_data().active_seat;
    DEBUG_ASSERT(parent_active_seat == context.active_seat);

    ValueTensor game_outcome;
    if (Rules::is_terminal(raw_child_state, parent_active_seat, raw_edge_action, game_outcome)) {
      new (child) Node(&lookup_table_, canonical_history, game_outcome);
    } else {
      new (child) Node(&lookup_table_, canonical_history, child_active_seat);
    }
    child->initialize_edges();
    bool overwrite = false;
    lookup_table_.insert_node(mcts_key, edge->child_index, overwrite);

    State canonical_child_state = canonical_history.current();
    canonical_history.undo();
    context.raw_history.undo();

    if (child->is_terminal()) continue;

    group::element_t sym = group::kIdentity;
    if (params_.apply_random_symmetries) {
      sym = bitset_util::choose_random_on_index(Symmetries::get_mask(canonical_child_state));
    }
    bool incorporate = params_.incorporate_sym_into_cache_key;
    context.eval_request.emplace_back(child, canonical_history, canonical_child_state, sym,
                                      incorporate);
  }

  node->update_child_expand_count(expand_count);
}

template <core::concepts::Game Game>
void Manager<Game>::virtual_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, search_path_str(context));
  }

  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  last_node->update_stats([&] {
    last_node->stats().VN++;  // thread-safe because executed under mutex
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().VN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <core::concepts::Game Game>
void Manager<Game>::undo_virtual_backprop(SearchContext& context) {
  // NOTE: this is not an exact undo of virtual_backprop(), since the context.search_path is
  // modified in between the two calls.

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, search_path_str(context));
  }

  RELEASE_ASSERT(!context.search_path.empty());

  for (int i = context.search_path.size() - 1; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E--;
      node->stats().VN--;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <core::concepts::Game Game>
inline void Manager<Game>::pure_backprop(SearchContext& context, const ValueArray& value) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, search_path_str(context),
             fmt::streamed(value.transpose()));
  }

  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  last_node->update_stats([&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    stats.init_q(value, true);
    stats.RN++;
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <core::concepts::Game Game>
void Manager<Game>::standard_backprop(SearchContext& context, bool undo_virtual) {
  Node* last_node = context.search_path.back().node;
  auto value = GameResults::to_value_array(last_node->stable_data().VT);

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, search_path_str(context),
             fmt::streamed(value.transpose()));
  }

  last_node->update_stats([&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    stats.init_q(value, false);
    stats.RN++;
    stats.VN -= undo_virtual;
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E += !undo_virtual;
      auto& stats = node->stats();  // thread-safe because executed under mutex
      stats.RN++;
      stats.VN -= undo_virtual;
    });
  }
  validate_search_path(context);
}

template <core::concepts::Game Game>
void Manager<Game>::short_circuit_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, search_path_str(context));
  }

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <core::concepts::Game Game>
void Manager<Game>::calc_canonical_state_data(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  context.canonical_history = context.raw_history;

  if constexpr (core::concepts::RequiresMctsDoublePass<Game>) {
    using Group = SymmetryGroup;
    context.canonical_history = root_info_.history_array[context.canonical_sym];
    group::element_t cur_canonical_sym = root_info_.canonical_sym;
    group::element_t leaf_canonical_sym = context.canonical_sym;
    for (const Visitation& visitation : context.search_path) {
      Edge* edge = visitation.edge;
      core::action_mode_t mode = visitation.node->action_mode();
      core::action_t action = edge->action;
      group::element_t sym = Group::compose(leaf_canonical_sym, Group::inverse(cur_canonical_sym));
      Symmetries::apply(action, sym, mode);
      Rules::apply(context.canonical_history, action);
      cur_canonical_sym = Group::compose(edge->sym, cur_canonical_sym);
    }

    RELEASE_ASSERT(cur_canonical_sym == leaf_canonical_sym,
                   "cur_canonical_sym={} leaf_canonical_sym={}", cur_canonical_sym,
                   leaf_canonical_sym);
  } else {
    Symmetries::apply(context.canonical_history, context.canonical_sym);
  }

  if (IS_DEFINED(DEBUG_BUILD)) {
    State s = context.canonical_history.current();
    Symmetries::apply(s, Symmetries::get_canonical_symmetry(s));
    if (s != context.canonical_history.current()) {
      std::cout << "ERROR! Bad Canonicalization!" << std::endl;
      std::cout << "canonical_sym_: " << int(context.canonical_sym) << std::endl;
      std::cout << "canonical_history.current():" << std::endl;
      IO::print_state(std::cout, context.canonical_history.current());
      std::cout << "Should be:" << std::endl;
      IO::print_state(std::cout, s);
      RELEASE_ASSERT(false);
    }
  }
}

template <core::concepts::Game Game>
void Manager<Game>::print_visit_info(const SearchContext& context) {
  if (mcts::kEnableSearchDebug) {
    Node* node = context.visit_node;
    LOG_INFO("{:>{}}visit {} seat={}", "", context.log_prefix_n(), search_path_str(context),
             node->stable_data().active_seat);
  }
}

template <core::concepts::Game Game>
void Manager<Game>::validate_search_path(const SearchContext& context) const {
  if (!IS_DEFINED(DEBUG_BUILD)) return;

  int N = context.search_path.size();
  for (int i = N - 1; i >= 0; --i) {
    context.search_path[i].node->validate_state();
  }
}

template <core::concepts::Game Game>
int Manager<Game>::get_best_child_index(const SearchContext& context) {
  Node* node = context.visit_node;
  bool is_root = (node == lookup_table_.get_node(root_info_.node_index));
  ActionSelector action_selector(params_, search_params_, node, is_root);

  using PVec = LocalPolicyArray;

  const PVec& P = action_selector.P;
  const PVec& mE = action_selector.mE;
  PVec& PUCT = action_selector.PUCT;

  int argmax_index;

  if (search_params_.tree_size_limit == 1) {
    // net-only, use P
    P.maxCoeff(&argmax_index);
  } else {
    bool force_playouts = params_.forced_playouts && is_root && search_params_.full_search &&
                          params_.dirichlet_mult > 0;

    if (force_playouts) {
      PVec n_forced = (P * params_.k_forced * mE.sum()).sqrt();
      auto F1 = (mE < n_forced).template cast<float>();
      auto F2 = (mE > 0).template cast<float>();
      auto F = F1 * F2;
      PUCT = PUCT * (1 - F) + F * 1e+6;
    }

    PUCT.maxCoeff(&argmax_index);
  }

  print_action_selection_details(context, action_selector, argmax_index);
  return argmax_index;
}

template <core::concepts::Game Game>
int Manager<Game>::sample_chance_child_index(const SearchContext& context) {
  Node* node = context.visit_node;
  int n = node->stable_data().num_valid_actions;
  float chance_dist[n];
  for (int i = 0; i < n; i++) {
    chance_dist[i] = node->get_edge(i)->base_prob;
  }
  return util::Random::weighted_sample(chance_dist, chance_dist + n);
}

template <core::concepts::Game Game>
std::string Manager<Game>::search_path_str(const SearchContext& context) const {
  group::element_t cur_sym = SymmetryGroup::inverse(root_info_.canonical_sym);
  std::string delim = IO::action_delimiter();
  std::vector<std::string> vec;
  for (const Visitation& visitation : context.search_path) {
    if (!visitation.edge) continue;
    core::action_mode_t mode = visitation.node->action_mode();
    core::action_t action = visitation.edge->action;
    Symmetries::apply(action, cur_sym, mode);
    cur_sym = SymmetryGroup::compose(cur_sym, SymmetryGroup::inverse(visitation.edge->sym));
    vec.push_back(IO::action_to_str(action, mode));
  }
  return std::format("[{}]", boost::algorithm::join(vec, delim));
}

template <core::concepts::Game Game>
void Manager<Game>::print_action_selection_details(const SearchContext& context,
                                                   const ActionSelector& selector,
                                                   int argmax_index) const {
  Node* node = context.visit_node;
  if (mcts::kEnableSearchDebug) {
    std::ostringstream ss;
    ss << std::format("{:>{}}", "", context.log_prefix_n());

    core::seat_index_t seat = node->stable_data().active_seat;

    int n_actions = node->stable_data().num_valid_actions;

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
      {"Seat", [&](float x) { return std::to_string(int(x)); }},
      {"CurP", [&](float x) { return std::string(x == seat ? "*" : ""); }},
    };

    std::stringstream ss1;
    eigen_util::print_array(ss1, player_data, player_columns, &fmt_map1);

    std::string line_break =
      std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength + context.log_prefix_n());

    for (const std::string& line : util::splitlines(ss1.str())) {
      ss << line << line_break;
    }

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

    LocalPolicyArray actions(n_actions);
    LocalPolicyArray child_addr(n_actions);
    LocalPolicyArray argmax(n_actions);
    child_addr.setConstant(-1);
    argmax.setZero();
    argmax(argmax_index) = 1;

    group::element_t inv_sym = SymmetryGroup::inverse(context.canonical_sym);
    for (int e = 0; e < n_actions; ++e) {
      auto edge = node->get_edge(e);
      core::action_t action = edge->action;
      Symmetries::apply(action, inv_sym, node->action_mode());
      actions(e) = action;
      child_addr(e) = edge->child_index;
    }

    static std::vector<std::string> action_columns = {
      "action", "P", "Q", "FPU", "PW", "PL", "E", "mE", "RN", "VN", "&ch", "PUCT", "argmax"};
    auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
      actions, P, Q, FPU, PW, PL, E, mE, RN, VN, child_addr, PUCT, argmax));

    eigen_util::PrintArrayFormatMap fmt_map2{
      {"action", [&](float x) { return IO::action_to_str(x, node->action_mode()); }},
      {"&ch", [](float x) { return x < 0 ? std::string() : std::to_string((int)x); }},
      {"argmax", [](float x) { return std::string(x == 0 ? "" : "*"); }},
    };

    std::stringstream ss2;
    eigen_util::print_array(ss2, action_data, action_columns, &fmt_map2);

    for (const std::string& line : util::splitlines(ss2.str())) {
      ss << line << line_break;
    }

    LOG_INFO(ss.str());
  }
}

template <core::concepts::Game Game>
void Manager<Game>::prepare_results() {
  lookup_table_.defragment(root_info_.node_index);
  Node* root = lookup_table_.get_node(root_info_.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();  // thread-safe since single-threaded here

  core::action_mode_t mode = root->action_mode();
  group::element_t sym = root_info_.canonical_sym;
  group::element_t inv_sym = SymmetryGroup::inverse(sym);

  results_.valid_actions.reset();
  results_.policy_prior.setZero();

  core::action_t actions[stable_data.num_valid_actions];

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(stable_data.valid_action_mask)) {
    Symmetries::apply(action, inv_sym, mode);
    results_.valid_actions.set(action, true);
    actions[i] = action;

    auto* edge = root->get_edge(i);
    results_.policy_prior(action) = edge->base_prob;

    i++;
  }

  load_action_symmetries(root, &actions[0]);
  root->write_results(params_, inv_sym, results_);
  results_.policy_target = results_.counts;
  results_.provably_lost = stats.provably_losing[stable_data.active_seat];
  results_.trivial = root->trivial();
  if (params_.forced_playouts && root_info_.add_noise) {
    prune_policy_target(search_params_, inv_sym);
  }

  Symmetries::apply(results_.counts, inv_sym, mode);
  Symmetries::apply(results_.policy_target, inv_sym, mode);
  Symmetries::apply(results_.Q, inv_sym, mode);
  Symmetries::apply(results_.Q_sq, inv_sym, mode);
  Symmetries::apply(results_.action_values, inv_sym, mode);

  results_.win_rates = stats.Q;
  results_.value_prior = stable_data.VT;
  results_.action_mode = mode;
}

template <core::concepts::Game Game>
inline void Manager<Game>::load_action_symmetries(Node* root, core::action_t* actions) {
  const auto& stable_data = root->stable_data();

  using Item = ActionSymmetryTable::Item;
  std::vector<Item> items;
  items.reserve(stable_data.num_valid_actions);

  for (int e = 0; e < stable_data.num_valid_actions; ++e) {
    Edge* edge = root->get_edge(e);
    if (edge->child_index < 0) continue;
    items.emplace_back(edge->child_index, actions[e]);
  }

  results_.action_symmetry_table.load(items);
}

template <core::concepts::Game Game>
void Manager<Game>::prune_policy_target(const SearchParams& search_params,
                                        group::element_t inv_sym) {
  if (params_.no_model) return;

  Node* root = lookup_table_.get_node(root_info_.node_index);
  ActionSelector action_selector(params_, search_params, root, true);

  const auto& P = action_selector.P;
  const auto& E = action_selector.E;
  const auto& PW = action_selector.PW;
  const auto& PL = action_selector.PL;
  const auto& mE = action_selector.mE;
  const auto& Q = action_selector.Q;
  const auto& PUCT = action_selector.PUCT;

  auto mE_sum = mE.sum();
  auto n_forced = (P * params_.k_forced * mE_sum).sqrt();

  int mE_max_index;
  auto mE_max = mE.maxCoeff(&mE_max_index);

  auto PUCT_max = PUCT(mE_max_index);
  auto sqrt_mE = sqrt(mE_sum + ActionSelector::eps);

  LocalPolicyArray mE_floor = params_.cPUCT * P * sqrt_mE / (PUCT_max - 2 * Q) - 1;

  int n_actions = root->stable_data().num_valid_actions;
  for (int i = 0; i < n_actions; ++i) {
    Edge* edge = root->get_edge(i);
    if (mE(i) == 0) {
      results_.policy_target(edge->action) = 0;
      continue;
    }
    if (mE(i) == mE_max) continue;
    if (!isfinite(mE_floor(i))) continue;
    if (mE_floor(i) >= mE(i)) continue;
    auto n = std::max(mE_floor(i), mE(i) - n_forced(i));
    if (n <= 1.0) {
      n = 0;
    }
    results_.policy_target(edge->action) = n;
  }

  if (eigen_util::sum(results_.policy_target) <= 0) {
    // can happen in certain edge cases
    results_.policy_target = results_.counts;
  }

  if (mcts::kEnableSearchDebug) {
    LocalPolicyArray actions(n_actions);
    LocalPolicyArray pruned(n_actions);

    core::action_mode_t mode = root->action_mode();
    for (int i = 0; i < n_actions; ++i) {
      core::action_t raw_action = root->get_edge(i)->action;
      core::action_t action = raw_action;
      Symmetries::apply(action, inv_sym, mode);
      actions(i) = action;
      pruned(i) = results_.policy_target(raw_action);
    }

    LocalPolicyArray target = pruned / pruned.sum();

    static std::vector<std::string> columns = {"action", "P",  "Q",  "PUCT",   "E",
                                               "PW",     "PL", "mE", "pruned", "target"};
    auto data = eigen_util::sort_rows(
      eigen_util::concatenate_columns(actions, P, Q, PUCT, E, PW, PL, mE, pruned, target));

    eigen_util::PrintArrayFormatMap fmt_map{
      {"action", [&](float x) { return IO::action_to_str(x, mode); }},
    };

    std::cout << std::endl << "Policy target pruning:" << std::endl;
    eigen_util::print_array(std::cout, data, columns, &fmt_map);
  }
}

}  // namespace mcts
