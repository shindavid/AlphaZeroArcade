// Forked from cpp/inline/alpha0/Manager.inl
// Key changes from alpha0::Manager:
// - 5 network heads: P(0), V(1), U(2), AV(3), AU(4)
// - load_evaluations: reads U into stable_data.uncertainty_, AU into edge->child_AU; W = U init
// - init_node_stats_from_terminal: W set to zero
// - update_stats: computes W via LoTV alongside Q via LoTE
// - write_results: populates AU and W in SearchResults

#include "beta0/Manager.hpp"

#include "beta0/BackupNNEvaluator.hpp"
#include "core/AuxEvalService.hpp"
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

#include <memory>
#include <sstream>

namespace beta0 {

template <beta0::concepts::Spec Spec>
Manager<Spec>::Manager(bool dummy, core::mutex_vec_sptr_t node_mutex_pool,
                       core::mutex_vec_sptr_t context_mutex_pool, const Params& params,
                       core::GameServerBase* server, EvalServiceBase_sptr service)
    : manager_id_(next_instance_id_++),
      params_(params),
      lookup_table_(node_mutex_pool),
      root_softmax_temperature_(params.starting_root_softmax_temperature,
                                params.ending_root_softmax_temperature,
                                params.root_softmax_temperature_half_life),
      context_mutex_pool_(context_mutex_pool) {
  if (params.enable_pondering) {
    throw util::CleanException("Pondering mode temporarily unsupported");
  }

  if (service) {
    nn_eval_service_ = service;
  } else {
    auto aux_factory = []() -> std::unique_ptr<core::AuxEvalService> {
      return std::make_unique<BackupNNEvaluator<Spec>>();
    };
    nn_eval_service_ = EvalServiceFactory::create(params, server, std::move(aux_factory));
  }

  // The aux service was constructed by NNEvaluationService (production path) or installed
  // directly on the service by the test fixture. In either case it must be a non-null
  // BackupNNEvaluator<Spec>* — beta0 has no other concrete AuxEvalService type.
  backup_nn_evaluator_ = dynamic_cast<BackupNNEvaluator<Spec>*>(nn_eval_service_->aux_service());
  RELEASE_ASSERT(backup_nn_evaluator_ != nullptr,
                 "beta0::Manager: nn_eval_service_->aux_service() is not a BackupNNEvaluator");

  contexts_.resize(num_search_threads());
  for (int i = 0; i < num_search_threads(); ++i) {
    init_context(i);
  }
}

template <beta0::concepts::Spec Spec>
Manager<Spec>::Manager(const Params& params, core::GameServerBase* server,
                       EvalServiceBase_sptr service)
    : Manager(true, std::make_shared<core::mutex_vec_t>(1), std::make_shared<core::mutex_vec_t>(1),
              params, server, service) {}

template <beta0::concepts::Spec Spec>
Manager<Spec>::Manager(core::mutex_vec_sptr_t& node_mutex_pool,
                       core::mutex_vec_sptr_t& context_mutex_pool, const Params& params,
                       core::GameServerBase* server, EvalServiceBase_sptr service)
    : Manager(true, node_mutex_pool, context_mutex_pool, params, server, service) {}

template <beta0::concepts::Spec Spec>
inline Manager<Spec>::~Manager() {
  clear();
  nn_eval_service_->disconnect();
}

template <beta0::concepts::Spec Spec>
inline void Manager<Spec>::start() {
  clear();

  if (!connected_) {
    nn_eval_service_->connect();
    connected_ = true;
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::RootInfo::clear() {
  state_step = 0;
  node_index = -1;
  active_seat = -1;
  add_noise = false;

  Rules::init_state(state);
  input_encoder.clear();
  input_encoder.update(state);
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::clear() {
  root_softmax_temperature_.reset();
  lookup_table_.clear();
  root_info_.clear();
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::receive_state_change(core::seat_index_t, const State&, const Move& move) {
  update(move);
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::update(const Move& move) {
  apply_move(root_info_.state, root_info_.input_encoder, move);
  root_info_.state_step++;
  root_softmax_temperature_.step();

  core::node_pool_index_t root_index = root_info_.node_index;
  if (root_index < 0) return;

  Node* root = lookup_table_.get_node(root_index);
  root_info_.node_index = lookup_child_by_move(root, move);  // tree reuse
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::backtrack(InfoSetIterator it, core::step_t step) {
  root_info_.state = it->info_set;
  root_info_.state_step++;
  root_info_.input_encoder.jump_to(it);
  root_softmax_temperature_.jump_to(step);
  const State& state = root_info_.state;
  TransposeKey key = Transposer::key(state);
  core::node_pool_index_t node_index = lookup_table_.lookup_node(key);
  root_info_.node_index = node_index;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::set_search_params(const SearchParams& params) {
  search_params_ = params;
}

template <beta0::concepts::Spec Spec>
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
template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::load_root_action_values(
  const ChanceEventHandleRequest& chance_request, core::seat_index_t seat,
  TrainingInfo& training_info) {
  if (!mid_load_root_action_values_) {
    init_root_info(search::kToLoadRootActionValues);

    // We do a dummy search with 0 iterations, just to get SearchThread to call init_root_node(),
    // which will expand all the root's children.
    constexpr int tree_size_limit = 0;
    constexpr bool full_search = true;
    constexpr bool ponder = false;
    SearchParams params{tree_size_limit, full_search, ponder};
    search_params_ = params;
    mid_load_root_action_values_ = true;
  }

  SearchRequest request(chance_request.notification_unit);
  SearchResponse response = search(request);
  if (response.yield_instruction == core::kYield) return core::kYield;
  RELEASE_ASSERT(response.yield_instruction == core::kContinue);

  Node* root = lookup_table_.get_node(root_info_.node_index);
  const auto& stable_data = root->stable_data();
  const auto& frame = root_info_.input_encoder.current_frame();

  ActionValueTensor& action_values = training_info.action_values_target;
  action_values.setZero();

  for (int i = 0; i < stable_data.num_valid_moves; i++) {
    const Edge* edge = lookup_table_.get_edge(root, i);
    const Node* child = lookup_table_.get_node(edge->child_index);
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

template <beta0::concepts::Spec Spec>
typename Manager<Spec>::SearchResponse Manager<Spec>::search_helper(const SearchRequest& request) {
  mit::unique_lock lock(state_machine_.mutex);
  auto context_id = request.context_id();
  SearchContext& context = contexts_[context_id];
  int extra_enqueue_count = 0;
  context.search_request = &request;

  if (state_machine_.state == kIdle) {
    RELEASE_ASSERT(context_id == 0);
    state_machine_.state = kInitializingRoot;
    backup_sample_k_ = request.backup_sample_k;
    results_.backup_sample.valid = false;
    backup_sample_snapshot_taken_.store(false, std::memory_order_relaxed);
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
    if (backup_sample_k_ > 0 && !backup_sample_snapshot_taken_.load(std::memory_order_relaxed)) {
      if (root->stats().total_count() >= backup_sample_k_) {
        bool expected = false;
        if (backup_sample_snapshot_taken_.compare_exchange_strong(expected, true,
                                                                  std::memory_order_acq_rel)) {
          capture_backup_sample(root);
        }
      }
    }
  }

  lock.lock();
  core::yield_instruction_t yield_instruction =
    mark_as_done_with_visit_loop(context, extra_enqueue_count);
  if (yield_instruction == core::kDrop) {
    return SearchResponse::make_drop();
  }
  RELEASE_ASSERT(yield_instruction == core::kContinue);

  lookup_table_.defragment(root_info_.node_index);

  root = lookup_table_.get_node(root_info_.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();  // thread-safe since single-threaded here
  const State& state = root_info_.state;

  results_.valid_moves = Game::Rules::analyze(state).valid_moves();
  results_.frame = root_info_.input_encoder.current_frame();
  results_.P.setZero();
  results_.pre_expanded_moves.setZero();

  RELEASE_ASSERT((int)results_.valid_moves.size() == stable_data.num_valid_moves, "{} != {}",
                 results_.valid_moves.size(), stable_data.num_valid_moves);

  int i = 0;
  for (Move move : results_.valid_moves) {
    auto* edge = lookup_table_.get_edge(root, i);
    auto index = PolicyEncoding::to_index(results_.frame, move);
    results_.P.coeffRef(index) = edge->policy_prior_prob;
    results_.pre_expanded_moves.coeffRef(index) = edge->was_pre_expanded;

    i++;
  }

  load_action_symmetries(root);
  write_results(root);
  results_.policy_target = results_.counts;
  results_.provably_lost = stats.provably_losing[stable_data.active_seat];
  if (params_.forced_playouts && root_info_.add_noise) {
    prune_policy_target();
  }

  results_.Q = stats.Q();
  results_.W = stats.W;
  results_.R = stable_data.R;
  return SearchResponse(&results_);
}

template <beta0::concepts::Spec Spec>
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

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::mark_as_done_with_visit_loop(SearchContext& context,
                                                                      int extra_enqueue_count) {
  // Assumes state_machine_.mutex is held
  RELEASE_ASSERT(context.in_visit_loop);
  context.in_visit_loop = false;
  state_machine_.in_visit_loop_count--;
  if (state_machine_.in_visit_loop_count == extra_enqueue_count) {
    state_machine_.state = kIdle;

    if (extra_enqueue_count > 0) {
      for (auto& context2 : contexts_) {
        context2.in_visit_loop = false;
      }
    }
    return core::kContinue;
  }
  return core::kDrop;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::init_context(core::context_id_t i) {
  SearchContext& context = contexts_[i];
  context.id = i;

  int n = context_mutex_pool_->size();
  if (n > 1) {
    context.pending_notifications_mutex_id = util::Random::uniform_sample(0, n);
  }
}

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_root_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  init_root_info(search::kForStandardSearch);

  core::node_pool_index_t root_index = root_info_.node_index;
  Node* root = lookup_table_.get_node(root_index);
  RELEASE_ASSERT(!root->is_terminal(), "unexpected terminal root node");

  if (!root->edges_initialized()) {
    initialize_edges(root, Game::Rules::analyze(root_info_.state).valid_moves());
  }

  init_root_edges();

  if (all_children_edges_initialized(root)) {
    const Params& manager_params = params_;
    const SearchParams& search_params = search_params_;
    bool pre_expand = manager_params.force_evaluate_all_root_children && search_params.full_search;
    if (pre_expand) {
      int n_moves = root->stable_data().num_valid_moves;
      for (int e = 0; e < n_moves; e++) {
        Edge* edge = lookup_table_.get_edge(root, e);
        edge->was_pre_expanded = true;
      }
    }
    return core::kContinue;
  }

  context.current_state = root_info_.state;
  context.input_encoder = root_info_.input_encoder;
  context.active_seat = root_info_.active_seat;
  context.initialization_index = root_index;
  return begin_node_initialization(context);
}

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_root_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  return resume_node_initialization(context);
}

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_node_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  const SearchParams& search_params = search_params_;
  const Params& manager_params = params_;

  core::node_pool_index_t node_index = context.initialization_index;
  Node* node = lookup_table_.get_node(node_index);

  context.mid_node_initialization = true;
  RELEASE_ASSERT(context.eval_request.num_fresh_items() == 0);

  bool is_root = (node_index == root_info_.node_index);
  if (!node->is_terminal()) {
    bool pre_expand =
      manager_params.force_evaluate_all_root_children && is_root && search_params.full_search;

    if (!node->stable_data().R_valid) {
      enqueue_node_evaluation(context, node);
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

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_node_initialization(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  const State& state = context.current_state;
  core::node_pool_index_t node_index = context.initialization_index;
  Node* node = lookup_table_.get_node(node_index);
  bool is_root = (node_index == root_info_.node_index);

  load_evaluations(context);
  context.eval_request.mark_all_as_stale();

  if (!node->is_terminal() && node->stable_data().is_chance_node) {
    auto chance_dist = Rules::get_chance_distribution(state);
    for (int i = 0; i < node->stable_data().num_valid_moves; i++) {
      Edge* edge = lookup_table_.get_edge(node, i);
      edge->chance_prob = chance_dist.get(edge->move);
    }
  }

  auto transpose_key = Transposer::key(state);
  bool overwrite = is_root;
  context.inserted_node_index = lookup_table_.insert_node(transpose_key, node_index, overwrite);
  context.mid_node_initialization = false;
  return core::kContinue;
}

// Re-runs the main NN for a node whose stable_data.weight_gen is older than the BackupNN
// evaluator's current weight_gen, then refreshes all weight-dependent fields. Triggered from
// begin_visit() (top-of-visit lazy refresh). The fresh inputs land in:
//   - stable_data.R, stable_data.uncertainty_  (rewritten by unpack_evaluation_into_node)
//   - per-edge policy_prior_prob, child_AV, child_AU  (ditto)
//   - stats.backup_accumulator and every edge->e_cached  (rebuilt by seed_backup_accumulator)
//   - stable_data.weight_gen  (set to evaluator->weight_gen())
// Deliberately preserved: search counts (E, RN, VN, backup_counter), child stats,
// edge->adjusted_base_prob (so root Dirichlet noise is not re-rolled), edge->chance_prob,
// stats.S/W/S_baseline/W_baseline (the next backup through this node will refresh
// stats.S/W via apply() under the new weights; the baselines are weight-independent).
//
// TODO(beta0-mt): when beta0 multi-threaded search is enabled, two contexts could race to
// refresh the same stale node. Need a per-node "refresh in progress" flag (analogous to
// Edge::kMidExpansion) so secondary contexts wait via a pending notification rather than
// enqueueing duplicate eval requests.
template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_node_refresh(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  Node* node = context.visit_node;
  RELEASE_ASSERT(!node->is_terminal(), "node refresh requested for terminal node");
  RELEASE_ASSERT(context.eval_request.num_fresh_items() == 0);

  context.mid_node_refresh = true;
  enqueue_node_evaluation(context, node);
  context.eval_request.set_notification_task_info(context.search_request->notification_unit);
  if (nn_eval_service_->evaluate(context.eval_request) == core::kYield) return core::kYield;

  return resume_node_refresh(context);
}

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_node_refresh(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  RELEASE_ASSERT(context.eval_request.num_fresh_items() == 1,
                 "node refresh expected exactly 1 fresh item, got {}",
                 context.eval_request.num_fresh_items());

  auto& item = context.eval_request.fresh_items().front();
  Node* node = static_cast<Node*>(item.node());
  unpack_evaluation_into_node(node, item);

  // TODO(beta0-mt): edge writes inside seed_backup_accumulator are not protected by the
  // parent's mutex; revisit before enabling multi-threaded beta0 search.
  seed_backup_accumulator(node, node->stats(), backup_nn_evaluator_->ready());
  node->stable_data().weight_gen = backup_nn_evaluator_->weight_gen();

  context.eval_request.mark_all_as_stale();
  context.mid_node_refresh = false;
  return core::kContinue;
}

// Append a single main-NN evaluation request for `node` to context.eval_request. Caller is
// responsible for binding the notification task and dispatching via nn_eval_service_.
template <beta0::concepts::Spec Spec>
void Manager<Spec>::enqueue_node_evaluation(SearchContext& context, Node* node) {
  InputEncoder& input_encoder = context.input_encoder;
  const auto& frame = input_encoder.current_frame();
  group::element_t sym = get_random_symmetry(input_encoder);
  bool incorporate = params_.incorporate_sym_into_cache_key;
  auto eval_key = input_encoder.eval_key();
  context.eval_request.emplace_back(frame, node, &lookup_table_, eval_key, input_encoder, sym,
                                    incorporate);
}

// Copy the eval result into `node`'s weight-dependent fields:
//   * stable_data.R / R_valid
//   * stable_data.uncertainty_
//   * stable_data.static_latent
//   * per-edge policy_prior_prob (raw, pre-transform), child_AV, child_AU, action_latent
// Returns the raw policy as a LocalPolicyArray so the caller can apply transform_policy() and
// write adjusted_base_prob separately if desired.
template <beta0::concepts::Spec Spec>
typename Manager<Spec>::LocalPolicyArray Manager<Spec>::unpack_evaluation_into_node(
  Node* node, typename SearchContext::EvalRequest::Item& item) {
  auto& stable_data = node->stable_data();
  int n = stable_data.num_valid_moves;

  GameResultTensor R;
  LocalPolicyArray P_raw(n);
  LocalActionValueArray AV(n, Game::Constants::kNumPlayers);
  LocalActionValueArray AU(n, Game::Constants::kNumPlayers);

  // beta0 head ordering: P(0), V(1), U(2), AV(3), AU(4), static_latent(5), action_latent(6).
  // See load_evaluations() for the static_asserts that pin this layout.
  auto eval = item.eval();
  std::copy_n(eval->data(0), P_raw.size(), P_raw.data());
  std::copy_n(eval->data(1), R.size(), R.data());
  std::copy_n(eval->data(3), AV.size(), AV.data());
  std::copy_n(eval->data(4), AU.size(), AU.data());

  RELEASE_ASSERT(eigen_util::isfinite(P_raw), "Non-finite values in policy head");
  RELEASE_ASSERT(eigen_util::isfinite(R), "Non-finite values in value head");
  RELEASE_ASSERT(eigen_util::isfinite(AV), "Non-finite values in action value head");
  RELEASE_ASSERT(eigen_util::isfinite(AU), "Non-finite values in action value uncertainty head");

  stable_data.R = R;
  stable_data.R_valid = true;
  std::copy_n(eval->data(2), stable_data.uncertainty_.size(), stable_data.uncertainty_.data());
  RELEASE_ASSERT(eigen_util::isfinite(stable_data.uncertainty_),
                 "Non-finite values in uncertainty head");

  std::copy_n(eval->data(5), stable_data.static_latent.size(), stable_data.static_latent.data());
  RELEASE_ASSERT(eigen_util::isfinite(stable_data.static_latent),
                 "Non-finite values in static_latent head");

  // action_latent layout in eval->data(6): per-action row-major, length n * kActionLatentDim.
  const float* za_data = eval->data(6);
  constexpr int kActionLatentDim = SpecTraits<Spec>::kActionLatentDim;
  for (int i = 0; i < n; ++i) {
    Edge* edge = lookup_table_.get_edge(node, i);
    edge->policy_prior_prob = P_raw[i];
    edge->child_AV = AV.row(i);
    edge->child_AU = AU.row(i);
    std::copy_n(za_data + i * kActionLatentDim, kActionLatentDim, edge->action_latent.data());
    RELEASE_ASSERT(eigen_util::isfinite(edge->action_latent),
                   "Non-finite values in action_latent head");
  }
  return P_raw;
}

// Recompute every edge's e_cached and the supplied stats.backup_accumulator from current
// state.
//
// use_backup_nn=false: zero the accumulator and every e_cached; reset last_seen_child_counter
// to the not-yet-seeded sentinel (-1). Used both at expansion when BackupNet has no weights
// loaded, and as a graceful-degradation step in update_stats() when weights are known stale.
//
// use_backup_nn=true: call compute_child_embedding() per visited edge. Unvisited edges
// (edge->E == 0) and edges whose child has not yet been backup'd through (RN == 0) are
// treated as zero-embedding contributors: e_cached is zeroed and last_seen_child_counter is
// reset to the not-yet-seeded sentinel (-1), so the first subsequent update_stats subtract-add
// pass adds in the freshly-computed embedding. This matches the training-time convention in
// write_results, which also skips unvisited edges (their (A, 6) row is all-zero, and the
// Python ChildEmbeddingHead's (P > 0) mask is irrelevant for them since P is also zero).
// Feeding inference rows with P > 0 / nonzero AVs+AUs for unvisited edges -- as we used to --
// summed in embeddings the BackupNet had never been trained on, producing severe distribution
// shift between train and infer.
//
// TODO(beta0-mt): edge writes here are not protected by the parent's mutex; revisit before
// enabling multi-threaded beta0 search.
template <beta0::concepts::Spec Spec>
void Manager<Spec>::seed_backup_accumulator(const Node* node, NodeStats& stats,
                                            bool use_backup_nn) {
  const auto& stable_data = node->stable_data();
  int n = stable_data.num_valid_moves;

  stats.backup_accumulator.setZero();

  if (!use_backup_nn) {
    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table_.get_edge(node, i);
      edge->e_cached.setZero();
      edge->last_seen_child_counter = -1;
    }
    return;
  }

  using BNN = BackupNNEvaluator<Spec>;
  typename BNN::ChildStatArray child_stats_vec;
  core::seat_index_t seat = stable_data.active_seat;

  for (int i = 0; i < n; ++i) {
    Edge* edge = lookup_table_.get_edge(node, i);

    const Node* child =
      edge->child_index >= 0 ? lookup_table_.get_node(edge->child_index) : nullptr;
    if (!child || edge->E == 0) {
      edge->e_cached.setZero();
      edge->last_seen_child_counter = -1;
      continue;
    }
    const auto child_stats = child->stats_safe();
    if (child_stats.RN == 0) {
      edge->e_cached.setZero();
      edge->last_seen_child_counter = -1;
      continue;
    }

    // Child's active-seat win-share scalar; read from the un-rotated S_baseline indexed by
    // the parent's active seat (equivalent to the rotated frame's slot 0).
    // Per-child input layout: [Qs, Ws, N, P, AVs, AUs].
    child_stats_vec(0) = child_stats.S_baseline(seat);
    child_stats_vec(1) = child_stats.W_baseline(seat);
    child_stats_vec(2) = float(edge->E);
    child_stats_vec(3) = edge->policy_prior_prob;
    child_stats_vec(4) = edge->child_AV(seat);
    child_stats_vec(5) = edge->child_AU(seat);
    edge->e_cached =
      backup_nn_evaluator_->compute_child_embedding(child_stats_vec, edge->action_latent);
    edge->last_seen_child_counter = child_stats.backup_counter;
    stats.backup_accumulator += edge->e_cached;
  }
}

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_search_iteration(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  Node* root = lookup_table_.get_node(root_info_.node_index);

  if (context.state_step != root_info_.state_step) {
    context.state_step = root_info_.state_step;
    context.current_state = root_info_.state;
  } else {
    Rules::backtrack_state(context.current_state, root_info_.state);
  }

  context.input_encoder = root_info_.input_encoder;
  context.active_seat = root_info_.active_seat;
  context.search_path.clear();
  context.search_path.emplace_back(root, nullptr);
  context.visit_node = root;
  context.mid_search_iteration = true;

  return resume_search_iteration(context);
}

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_search_iteration(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  if (context.mid_visit) {
    if (resume_visit(context) == core::kYield) return core::kYield;
  }

  while (context.visit_node) {
    if (begin_visit(context) == core::kYield) return core::kYield;
  }

  Rules::backtrack_state(context.current_state, root_info_.state);
  context.input_encoder = root_info_.input_encoder;
  context.active_seat = root_info_.active_seat;
  if (post_visit_func_) post_visit_func_();
  context.mid_search_iteration = false;
  return core::kContinue;
}

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_visit(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  Node* node = context.visit_node;
  print_visit_info(context);
  context.mid_visit = true;
  context.expanded_new_node = false;

  const auto& stable_data = node->stable_data();
  if (stable_data.terminal) {
    standard_backup(context);
    context.visit_node = nullptr;
    context.mid_visit = false;
    return core::kContinue;
  }

  // If this node was last NN-evaluated under an older BackupNN weight_gen than the
  // evaluator's current one, every weight-dependent field on this node (R, U, per-edge
  // P/AV/AU, e_cached) is stale, and stats.S / stats.W (when written by apply()) reflect
  // old-weight inference. Before descending into action selection -- which reads
  // adjusted_base_prob and child Q -- we re-run the main NN for this node, refresh those
  // fields, and reseed the BackupNN accumulator from current child stats. This yields if
  // the NN service yields; on resume, resume_visit() routes back through
  // resume_node_refresh() and then re-enters begin_visit() so action selection runs against
  // fresh values.
  //
  // Skipped when BackupNN is not ready (no weights ever loaded -> no override is computed
  // anyway; weight_gen mismatch is harmless). Terminal nodes are also implicitly skipped
  // (handled above) since they never get NN-evaluated.
  if (backup_nn_evaluator_->ready() &&
      stable_data.weight_gen != backup_nn_evaluator_->weight_gen()) {
    return begin_node_refresh(context);
  }

  int child_index;
  if (stable_data.is_chance_node) {
    child_index = sample_chance_child_index(context);
  } else {
    child_index = get_best_child_index(context);
  }

  Edge* edge = lookup_table_.get_edge(node, child_index);
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
      Node* child = lookup_table_.get_node(edge->child_index);
      context.search_path.emplace_back(child, nullptr);

      if (should_short_circuit(edge, child)) {
        short_circuit_backup(context);
      } else {
        standard_backup(context);
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

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_visit(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  Edge* edge = context.visit_edge;

  if (context.mid_node_refresh) {
    if (resume_node_refresh(context) == core::kYield) return core::kYield;
    // The same node still needs to be visited. Clear mid_visit so the outer loop re-enters
    // begin_visit() against a now-fresh node (whose weight_gen now matches the evaluator's,
    // so the staleness check will fall through to normal action selection).
    context.mid_visit = false;
    return core::kContinue;
  }

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

  Node* child = lookup_table_.get_node(edge->child_index);
  if (child) {
    context.search_path.emplace_back(child, nullptr);

    if (should_short_circuit(edge, child)) {
      short_circuit_backup(context);
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

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::begin_expansion(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  context.mid_expansion = true;

  const State& state = context.current_state;
  TransposeKey transpose_key = Transposer::key(state);

  context.initialization_index = lookup_table_.lookup_node(transpose_key);

  context.expanded_new_node = context.initialization_index < 0;
  if (context.expanded_new_node) {
    context.initialization_index = lookup_table_.alloc_node();
    Node* child = lookup_table_.get_node(context.initialization_index);

    auto result = Rules::analyze(state);
    bool terminal = result.is_terminal();

    if (terminal) {
      new (child) Node(lookup_table_.get_random_mutex(), state, result.outcome());
      init_node_stats_from_terminal(child);
    } else {
      new (child) Node(lookup_table_.get_random_mutex(), state, result.valid_moves().size(),
                       context.active_seat);
      initialize_edges(child, result.valid_moves());
    }

    context.search_path.emplace_back(child, nullptr);
    bool do_virtual = !terminal && multithreaded();
    if (do_virtual) {
      virtual_backup(context);
    }

    if (begin_node_initialization(context) == core::kYield) return core::kYield;
  }
  return resume_expansion(context);
}

template <beta0::concepts::Spec Spec>
core::yield_instruction_t Manager<Spec>::resume_expansion(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  Edge* edge = context.visit_edge;
  Node* parent = context.visit_node;

  if (context.mid_node_initialization) {
    if (resume_node_initialization(context) == core::kYield) return core::kYield;
  }

  if (context.expanded_new_node) {
    Node* child = lookup_table_.get_node(context.initialization_index);
    bool terminal = child->is_terminal();
    bool do_virtual = !terminal && multithreaded();

    edge->child_index = context.inserted_node_index;
    if (context.initialization_index != context.inserted_node_index) {
      context.search_path.pop_back();
      if (do_virtual) {
        undo_virtual_backup(context);
      }
      context.expanded_new_node = false;
    } else {
      standard_backup(context, do_virtual);
    }
  } else {
    edge->child_index = context.initialization_index;
  }

  mit::unique_lock lock(parent->mutex());
  set_edge_state(context, edge, Edge::kExpanded);
  lock.unlock();

  context.mid_expansion = false;
  return core::kContinue;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::virtual_backup(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  backup(context, last_node, nullptr, [&] { virtually_update_node_stats(last_node); });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    backup(context, node, edge, [&] { virtually_update_node_stats_and_edge(node, edge); });
  }
  validate_search_path(context);
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::undo_virtual_backup(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  RELEASE_ASSERT(!context.search_path.empty());

  for (int i = context.search_path.size() - 1; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    backup(context, node, edge, [&] { undo_virtual_update(node, edge); });
  }
  validate_search_path(context);
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::standard_backup(SearchContext& context, bool undo_virtual) {
  Node* last_node = context.search_path.back().node;
  auto value = last_node->stable_data().V();

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, context.search_path_str(),
             fmt::streamed(value.transpose()));
  }

  backup(context, last_node, nullptr, [&] { update_node_stats(last_node, undo_virtual); });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    backup(context, node, edge, [&] { update_node_stats_and_edge(node, edge, undo_virtual); });
  }
  validate_search_path(context);
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::short_circuit_backup(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    backup(context, node, edge, [&] { update_node_stats_and_edge(node, edge, false); });
  }
  validate_search_path(context);
}

template <beta0::concepts::Spec Spec>
core::node_pool_index_t Manager<Spec>::lookup_child_by_move(const Node* node,
                                                            const Move& move) const {
  int n = node->stable_data().num_valid_moves;

  if constexpr (MoveSet::kSortedByMove) {
    int left = 0;
    int right = n - 1;
    while (left <= right) {
      int mid = left + (right - left) / 2;
      const Edge* edge = lookup_table_.get_edge(node, mid);
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
    for (int i = 0; i < n; i++) {
      const Edge* edge = lookup_table_.get_edge(node, i);
      if (edge->move == move) {
        return edge->child_index;
      }
    }
    return -1;
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::initialize_edges(Node* node, const MoveSet& valid_moves) {
  int n_edges = node->stable_data().num_valid_moves;
  RELEASE_ASSERT(n_edges == (int)valid_moves.size());
  if (n_edges == 0) return;

  node->set_first_edge_index(lookup_table_.alloc_edges(n_edges));

  int i = 0;
  for (Move move : valid_moves) {
    Edge* edge = lookup_table_.get_edge(node, i);
    new (edge) Edge();
    edge->move = move;
    i++;
  }
}

template <beta0::concepts::Spec Spec>
bool Manager<Spec>::all_children_edges_initialized(const Node* root) const {
  int n = root->stable_data().num_valid_moves;
  if (n == 0) return true;
  if (root->get_first_edge_index() == -1) return false;

  for (int i = 0; i < n; i++) {
    Edge* edge = lookup_table_.get_edge(root, i);
    if (edge->state != Edge::kExpanded) {
      return false;
    }
  }
  return true;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::add_pending_notification(SearchContext& context, Edge* edge) {
  DEBUG_ASSERT(multithreaded());
  DEBUG_ASSERT(edge->expanding_context_id >= 0);
  DEBUG_ASSERT(edge->expanding_context_id != context.id);

  core::SlotContext slot_context(context.search_request->game_slot_index(), context.id);

  SearchContext& notifying_context = contexts_[edge->expanding_context_id];
  mit::mutex& mutex = (*context_mutex_pool_)[notifying_context.pending_notifications_mutex_id];
  mit::unique_lock lock(mutex);
  notifying_context.pending_notifications.push_back(slot_context);
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::set_edge_state(SearchContext& context, Edge* edge,
                                   Edge::expansion_state_t state) {
  LOG_TRACE("{:>{}}{}() state={}", "", context.log_prefix_n(), __func__, state);
  if (state == Edge::kPreExpanded) {
    edge->state = state;
  } else if (state == Edge::kMidExpansion) {
    edge->state = state;
    edge->expanding_context_id = context.id;
  } else if (state == Edge::kExpanded) {
    mit::mutex& mutex = (*context_mutex_pool_)[context.pending_notifications_mutex_id];
    mit::unique_lock lock(mutex);
    edge->state = state;
    edge->expanding_context_id = -1;
    context.search_request->yield_manager()->notify(context.pending_notifications);
    context.pending_notifications.clear();
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::pre_expand_children(SearchContext& context, Node* node) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);

  const Params& manager_params = params_;

  InputFrame parent_frame = context.input_encoder.current_frame();
  const State& parent_state = root_info_.state;
  RELEASE_ASSERT(parent_state == context.current_state);

  int n_moves = node->stable_data().num_valid_moves;
  for (int e = 0; e < n_moves; e++) {
    Edge* edge = lookup_table_.get_edge(node, e);
    edge->was_pre_expanded = true;
    if (edge->child_index >= 0) continue;

    Rules::apply(context.current_state, edge->move);
    const State& child_state = context.current_state;

    core::seat_index_t child_active_seat = context.active_seat;
    if (!Rules::is_chance_state(child_state)) {
      child_active_seat = Rules::get_current_player(child_state);
    }

    set_edge_state(context, edge, Edge::kPreExpanded);

    TransposeKey transpose_key = Transposer::key(child_state);
    core::node_pool_index_t child_index = lookup_table_.lookup_node(transpose_key);
    if (child_index >= 0) {
      edge->child_index = child_index;
      Rules::backtrack_state(context.current_state, parent_state);
      continue;
    }

    edge->child_index = lookup_table_.alloc_node();
    Node* child = lookup_table_.get_node(edge->child_index);

    auto result = Rules::analyze(child_state);
    bool terminal = result.is_terminal();

    if (terminal) {
      new (child) Node(lookup_table_.get_random_mutex(), child_state, result.outcome());
      init_node_stats_from_terminal(child);
    } else {
      new (child) Node(lookup_table_.get_random_mutex(), child_state, result.valid_moves().size(),
                       child_active_seat);
      initialize_edges(child, result.valid_moves());
    }
    bool overwrite = false;
    lookup_table_.insert_node(transpose_key, edge->child_index, overwrite);

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

    context.eval_request.emplace_back(parent_frame, child, &lookup_table_, eval_key,
                                      context.input_encoder, child_frame, sym, incorporate);
    Rules::backtrack_state(context.current_state, parent_state);
  }
  RELEASE_ASSERT(context.current_state == root_info_.state);
}

template <beta0::concepts::Spec Spec>
int Manager<Spec>::sample_chance_child_index(const SearchContext& context) {
  Node* node = context.visit_node;
  int n = node->stable_data().num_valid_moves;
  float chance_dist[n];
  for (int i = 0; i < n; i++) {
    chance_dist[i] = lookup_table_.get_edge(node, i)->chance_prob;
  }
  return util::Random::weighted_sample(chance_dist, chance_dist + n);
}

template <beta0::concepts::Spec Spec>
group::element_t Manager<Spec>::get_random_symmetry(const InputEncoder& input_encoder) const {
  group::element_t sym = group::kIdentity;
  if (params_.apply_random_symmetries) {
    sym = input_encoder.get_random_symmetry();
  }
  return sym;
}

template <beta0::concepts::Spec Spec>
group::element_t Manager<Spec>::get_random_symmetry(const InputEncoder& input_encoder,
                                                    const State& next_state) const {
  group::element_t sym = group::kIdentity;
  if (params_.apply_random_symmetries) {
    sym = input_encoder.get_random_symmetry(next_state);
  }
  return sym;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::apply_move(State& state, InputEncoder& input_encoder, const Move& move) {
  Rules::apply(state, move);
  input_encoder.update(state);
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::print_visit_info(const SearchContext& context) {
  if (search::kEnableSearchDebug) {
    const Node* node = context.visit_node;
    LOG_INFO("{:>{}}visit {} seat={}", "", context.log_prefix_n(), context.search_path_str(),
             node->stable_data().active_seat);
  }
}

template <beta0::concepts::Spec Spec>
template <typename MutexProtectedFunc>
void Manager<Spec>::backup(SearchContext& context, Node* node, Edge* edge,
                             MutexProtectedFunc&& func) {
  // update_stats() reads child stats (via child->stats_safe()) which acquires child mutexes.
  // The mutex pool can map multiple nodes to the same mutex (notably size-1 pool in tests), so
  // we MUST NOT hold the parent's mutex while calling update_stats. Pattern: lock parent, run
  // func() (mutates stats counters under lock), bump backup_counter, copy stats out, unlock,
  // run update_stats on the copy, relock and merge back (preserving counters).
  //
  // backup_counter is incremented once per call (under lock); parents read this counter via
  // their child's stats_safe() to detect which child contributions changed since their last
  // incremental BackupNN accumulator update. See NodeStats::backup_counter.
  mit::unique_lock lock(node->mutex());
  func();
  if (!edge) return;
  NodeStats stats = node->stats();  // copy
  lock.unlock();

  update_stats(stats, node);

  lock.lock();
  int RN = node->stats().RN;
  int VN = node->stats().VN;
  int backup_counter = node->stats().backup_counter;
  node->stats() = stats;
  node->stats().RN = RN;
  node->stats().VN = VN;
  node->stats().backup_counter = backup_counter + 1;
  lock.unlock();
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::init_node_stats_from_terminal(Node* node) {
  NodeStats& stats = node->stats();
  RELEASE_ASSERT(stats.RN == 0);
  const auto& stable_data = node->stable_data();
  const ValueArray q = stable_data.V();

  stats.S = stable_data.R;
  stats.W.setZero();  // no uncertainty at terminal nodes
  stats.S_baseline = stable_data.R;
  stats.W_baseline.setZero();
  stats.backup_accumulator.setZero();

  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    stats.provably_winning[p] = q(p) >= GameResultEncoding::kMaxValue;
    stats.provably_losing[p] = q(p) <= GameResultEncoding::kMinValue;
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::update_node_stats(Node* node, bool undo_virtual) {
  auto& stats = node->stats();

  stats.RN++;
  stats.VN -= undo_virtual;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual) {
  auto& stats = node->stats();

  edge->E += !undo_virtual;
  stats.RN++;
  stats.VN -= undo_virtual;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::virtually_update_node_stats(Node* node) {
  node->stats().VN++;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::virtually_update_node_stats_and_edge(Node* node, Edge* edge) {
  edge->E++;
  node->stats().VN++;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::undo_virtual_update(Node* node, Edge* edge) {
  edge->E--;
  node->stats().VN--;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::validate_search_path(const SearchContext& context) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;

  int N = context.search_path.size();
  for (int i = N - 1; i >= 0; --i) {
    validate_state(context.search_path[i].node);
  }
}

template <beta0::concepts::Spec Spec>
bool Manager<Spec>::should_short_circuit(const Edge* edge, const Node* child) {
  int edge_count = edge->E;
  int child_count = child->stats().RN;  // not thread-safe but race-condition is benign
  return edge_count < child_count;
}

template <beta0::concepts::Spec Spec>
bool Manager<Spec>::more_search_iterations_needed(const Node* root) const {
  const search::SearchParams& search_params = search_params_;
  if (!search_params.ponder && root->stable_data().num_valid_moves == 1) return false;
  return root->stats().total_count() <= search_params.tree_size_limit;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::init_root_info(search::RootInitPurpose purpose) {
  const Params& manager_params = params_;
  const search::SearchParams& search_params = search_params_;

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

  root_info_.add_noise = add_noise;
  if (root_info_.node_index < 0 || add_noise) {
    root_info_.node_index = lookup_table_.alloc_node();
    Node* root = lookup_table_.get_node(root_info_.node_index);

    const State& cur_state = root_info_.state;
    core::seat_index_t active_seat = Game::Rules::get_current_player(cur_state);
    RELEASE_ASSERT(active_seat >= 0 && active_seat < Game::Constants::kNumPlayers);
    root_info_.active_seat = active_seat;
    auto legal_moves = Game::Rules::analyze(cur_state).valid_moves();
    new (root) Node(lookup_table_.get_random_mutex(), cur_state, legal_moves.size(), active_seat);
  }

  if (search::kEnableSearchDebug && purpose == search::kForStandardSearch) {
    IO::print_state(std::cout, root_info_.state);
  }
}

template <beta0::concepts::Spec Spec>
int Manager<Spec>::get_best_child_index(const SearchContext& context) {
  const search::SearchParams& search_params = search_params_;
  const Params& manager_params = params_;

  Node* node = context.visit_node;
  bool is_root = (node == lookup_table_.get_node(root_info_.node_index));
  PuctCalculator action_selector(lookup_table_, manager_params, search_params, node, is_root);

  using PVec = LocalPolicyArray;

  const PVec& P = action_selector.P;
  const PVec& mE = action_selector.mE;
  PVec& PUCT = action_selector.PUCT;

  int argmax_index;

  if (search_params.tree_size_limit == 1) {
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

  print_action_selection_details(context, action_selector, argmax_index);
  return argmax_index;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::load_evaluations(SearchContext& context) {
  // beta0 head ordering: P(0), V(1), U(2), AV(3), AU(4), static_latent(5), action_latent(6).
  // unpack_evaluation_into_node() relies on these positional indices; pin the layout here.
  using NetworkHeadsList = Spec::NetworkHeads::List;
  static_assert(util::str_equal<mp::TypeAt_t<NetworkHeadsList, 0>::kName, "policy">());
  static_assert(util::str_equal<mp::TypeAt_t<NetworkHeadsList, 1>::kName, "value">());
  static_assert(util::str_equal<mp::TypeAt_t<NetworkHeadsList, 2>::kName, "value_uncertainty">());
  static_assert(util::str_equal<mp::TypeAt_t<NetworkHeadsList, 3>::kName, "action_value">());
  static_assert(
    util::str_equal<mp::TypeAt_t<NetworkHeadsList, 4>::kName, "action_value_uncertainty">());
  static_assert(util::str_equal<mp::TypeAt_t<NetworkHeadsList, 5>::kName, "static_latent">());
  static_assert(util::str_equal<mp::TypeAt_t<NetworkHeadsList, 6>::kName, "action_latent">());

  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());
    auto& stable_data = node->stable_data();
    auto& stats = node->stats();
    int n = stable_data.num_valid_moves;

    // No need to worry about thread-safety when modifying edges or stats below, since no other
    // threads can access this node until after load_evaluations() returns.

    LocalPolicyArray P_raw = unpack_evaluation_into_node(node, item);

    LocalPolicyArray P_adjusted = P_raw;
    transform_policy(context, P_adjusted);
    for (int i = 0; i < n; ++i) {
      lookup_table_.get_edge(node, i)->adjusted_base_prob = P_adjusted[i];
    }

    ValueArray U = stable_data.U();
    stats.S = stable_data.R;
    stats.W = U;  // Initialize W to the prior uncertainty from the neural network
    stats.S_baseline = stable_data.R;
    stats.W_baseline = U;

    // Seed backup_accumulator from the BackupNet's child-embedding head. At this point no
    // children have been expanded, so seed_backup_accumulator() falls into its post-expansion
    // sentinel branch for every edge: e_i computed with (Qs=0, Ws=0, N=0) and the static
    // (P, AVs, AUs) factors from this NN evaluation; last_seen_child_counter set to 0 to
    // match a freshly-constructed child.
    seed_backup_accumulator(node, node->stats(), backup_nn_evaluator_->ready());

    // Snapshot the BackupNN weight generation under which this node's NN evaluation (and the
    // backup_accumulator seed above) were produced. update_stats() and begin_visit() compare
    // this against the evaluator's current weight_gen() to detect staleness after a model
    // reload. Recorded unconditionally (even when !ready()) so that a node first evaluated
    // with no backup-NN weights still gets a well-defined baseline; staleness checks treat
    // weight_gen == evaluator.weight_gen() as up-to-date in either case.
    stable_data.weight_gen = backup_nn_evaluator_->weight_gen();
  }

  Node* root = lookup_table_.get_node(root_info_.node_index);
  if (root) {
    root->stats().RN = std::max(root->stats().RN, 1);
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::update_stats(NodeStats& stats, const Node* node) {
  using ResultArray = Eigen::Array<float, GameResultTensor::Dimensions::total_size, 1>;
  ResultArray S_sum;
  S_sum.setZero();

  player_bitset_t all_provably_winning;
  player_bitset_t all_provably_losing;

  const auto& stable_data = node->stable_data();

  int num_valid_moves = stable_data.num_valid_moves;
  core::seat_index_t seat = stable_data.active_seat;

  if (stable_data.is_chance_node) {
    all_provably_winning.set();
    all_provably_losing.set();

    int num_expanded_edges = 0;
    ValueArray W_sum;
    W_sum.setZero();

    for (int i = 0; i < num_valid_moves; i++) {
      const Edge* edge = lookup_table_.get_edge(node, i);
      const Node* child = lookup_table_.get_node(edge->child_index);

      if (!child) {
        break;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      S_sum += eigen_util::reinterpret_as_array(child_stats.S) * edge->chance_prob;
      W_sum += child_stats.W * edge->chance_prob;
      num_expanded_edges++;

      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
    }
    if (num_expanded_edges == num_valid_moves) {
      eigen_util::reinterpret_as_array(stats.S) = S_sum;
      stats.W = W_sum;
      stats.S_baseline = stats.S;
      stats.W_baseline = W_sum;
      stats.provably_winning = all_provably_winning;
      stats.provably_losing = all_provably_losing;
    }
    return;
  } else {
    // provably winning/losing calculation
    bool cp_has_winning_move = false;
    int N = 0;

    NodeStats child_stats_arr[num_valid_moves];
    Edge* edge_arr[num_valid_moves];
    int child_stats_arr_count = 0;

    DEBUG_ASSERT(num_valid_moves > 0);
    for (int i = 0; i < num_valid_moves; i++) {
      Edge* edge = lookup_table_.get_edge(node, i);
      const Node* child = lookup_table_.get_node(edge->child_index);
      if (!child) {
        continue;
      }
      const NodeStats child_stats = child->stats_safe();  // make a copy
      if (child_stats.RN == 0) {
        continue;
      }
      edge_arr[child_stats_arr_count] = edge;
      child_stats_arr[child_stats_arr_count++] = child_stats;
    }

    bool all_edges_expanded = (child_stats_arr_count == num_valid_moves);
    if (all_edges_expanded) {
      all_provably_winning.set();
      all_provably_losing.set();
    }

    float maxQs = GameResultEncoding::kMinValue;
    int best_q_idx = -1;
    float maxWs = 0.f;
    for (int i = 0; i < child_stats_arr_count; i++) {
      int e = edge_arr[i]->E;
      const auto& child_stats = child_stats_arr[i];
      ValueArray child_Q = child_stats.Q();
      N += e;
      S_sum += eigen_util::reinterpret_as_array(child_stats.S) * float(e);
      if (child_Q(seat) > maxQs) {
        maxQs = child_Q(seat);
        best_q_idx = i;
      }
      maxWs = std::max(maxWs, child_stats.W(seat));
      eigen_util::debug_assert_is_valid_prob_distr(child_Q);
      cp_has_winning_move |= child_stats.provably_winning[seat];
      if (all_edges_expanded) {
        all_provably_winning &= child_stats.provably_winning;
        all_provably_losing &= child_stats.provably_losing;
      }
    }

    if (cp_has_winning_move) {
      stats.provably_winning[seat] = true;
      stats.provably_losing.set();
      stats.provably_losing[seat] = false;
    } else if (all_edges_expanded) {
      stats.provably_winning = all_provably_winning;
      stats.provably_losing = all_provably_losing;
    }

    bool aggregation_needed = !stats.provably_winning[seat] && !stats.provably_losing[seat];
    aggregation_needed |= maxWs == 0.f;
    if (aggregation_needed) {
      DEBUG_ASSERT(stable_data.R_valid);
      const GameResultTensor& R = stable_data.R;
      ValueArray V = stable_data.V();
      S_sum += eigen_util::reinterpret_as_array(R);
      N++;
      eigen_util::debug_assert_is_valid_prob_distr(V);

      eigen_util::reinterpret_as_array(stats.S) = S_sum / float(N);
      ValueArray Q = stats.Q();
      eigen_util::debug_assert_is_valid_prob_distr(Q);

      // LoTV (Law of Total Variance) computation for W:
      // W(s) = (U + sum_i(e_i * [W_i + (Q_i - Q)^2]) + (V - Q)^2) / N
      // where U = prior uncertainty from the uncertainty head
      ValueArray U = stable_data.U();
      ValueArray diff_V = V - Q;
      ValueArray W_sum = U + diff_V * diff_V;

      for (int i = 0; i < child_stats_arr_count; i++) {
        int e = edge_arr[i]->E;
        const auto& child_stats = child_stats_arr[i];
        ValueArray diff = child_stats.Q() - Q;
        W_sum += e * (child_stats.W + diff * diff);
      }

      stats.W = W_sum / float(N);

      // Preserve the prior-augmented children-average baselines (S_baseline, W_baseline) that
      // BackupNet will consume as context inputs. These must be captured before the override
      // below replaces stats.S / stats.W. Stored in the canonical (un-rotated) frame.
      stats.S_baseline = stats.S;
      stats.W_baseline = stats.W;

      // Backup NN override: if evaluator is ready, do an incremental NNUE-style subtract-add
      // pass over every edge whose child's backup_counter has advanced since its e_cached was
      // last refreshed, then apply BackupNet to get the (S, W) override. We must scan all edges
      // (not just the one that triggered this backup) because with multi-threading and/or
      // move-transposition multiple children's stats can change between consecutive backups
      // through this parent.
      //
      // Stale-weights graceful degradation: if this node was last NN-evaluated under an older
      // BackupNN weight_gen than the evaluator's current one, every cached e_i was computed
      // with old weights and the subtract-add chain would produce garbage. Such a node will
      // be fully refreshed on its next visit by begin_visit() -> begin_node_refresh(), which
      // yields for a fresh main-NN evaluation. Until then, we degrade gracefully: zero the
      // accumulator and every e_cached via seed_backup_accumulator(use_backup_nn=false), and
      // skip the override entirely. stats.S / stats.W remain at their LoTE / LoTV values for
      // this one backup -- equivalent to "BackupNN not ready". We do NOT bump
      // stable_data.weight_gen here; that happens once the main-NN refresh completes and the
      // per-edge P/AV/AU/action_latent are also fresh.
      //
      // TODO(beta0-mt): the writes to edge->e_cached / edge->last_seen_child_counter below
      // (both via seed_backup_accumulator and in the subtract-add loop) are not protected by
      // the parent's mutex. This pre-dates the staleness work and is a known thread-safety
      // gap; it must be revisited before beta0 multi-threaded search is enabled.
      if (backup_nn_evaluator_->ready()) {
        bool stale = stable_data.weight_gen != backup_nn_evaluator_->weight_gen();
        if (stale) {
          seed_backup_accumulator(node, stats, /*use_backup_nn=*/false);
        } else {
          using BNN = BackupNNEvaluator<Spec>;
          typename BNN::ChildStatArray child_stats_vec;
          for (int i = 0; i < child_stats_arr_count; i++) {
            Edge* edge = edge_arr[i];
            const auto& child_stats = child_stats_arr[i];
            if (child_stats.backup_counter == edge->last_seen_child_counter) continue;
            // Per-child child_stats_vec layout: [Qs, Ws, N, P, AVs, AUs]. Qs is the parent's
            // active-seat win-share scalar; we read it from the un-rotated S_baseline indexed
            // by the parent's active seat (equivalent to the rotated frame's slot 0).
            child_stats_vec(0) = child_stats.S_baseline(seat);
            child_stats_vec(1) = child_stats.W_baseline(seat);
            child_stats_vec(2) = float(edge->E);
            child_stats_vec(3) = edge->policy_prior_prob;
            child_stats_vec(4) = edge->child_AV(seat);
            child_stats_vec(5) = edge->child_AU(seat);
            auto e_new =
              backup_nn_evaluator_->compute_child_embedding(child_stats_vec, edge->action_latent);
            stats.backup_accumulator += (e_new - edge->e_cached);
            edge->e_cached = e_new;
            edge->last_seen_child_counter = child_stats.backup_counter;
          }

          // Materialize the active-seat-rotated baseline that BackupNet expects as input.
          typename NodeStats::Tensor S_baseline = stats.S_baseline;
          GameResultEncoding::left_rotate(S_baseline, seat);
          auto sw = backup_nn_evaluator_->apply(stats.backup_accumulator, stable_data.static_latent,
                                                S_baseline, stats.W_baseline(seat));
          // sw.S is active-seat-rotated; un-rotate to canonical frame for storage in stats.S.
          stats.S = sw.S;
          GameResultEncoding::right_rotate(stats.S, seat);
          stats.setW(sw.W);
        }
      }
    } else {  // !aggregation_needed
      // Set stats.S to the S of the child with the maximal Q for the active seat. This works
      // uniformly for all three sub-cases (provably_winning, provably_losing, and the W==0
      // fallback): the best-Q child's full WLD distribution is the most appropriate
      // representation of this node's value.
      DEBUG_ASSERT(best_q_idx >= 0);
      stats.S = child_stats_arr[best_q_idx].S;
      stats.setW(0.f);

      // S_baseline and W_baseline shouldn't matter from this point on, but set them anyway.
      stats.S_baseline = stats.S;
      stats.W_baseline.setZero();
    }
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::write_results(const Node* root) {
  // This should only be called in contexts where the search-threads are inactive, so we do not need
  // to worry about thread-safety

  const Params& params = params_;

  core::seat_index_t seat = root->stable_data().active_seat;
  DEBUG_ASSERT(seat >= 0 && seat < kNumPlayers);

  const auto& frame = results_.frame;
  auto& counts = results_.counts;
  auto& AV = results_.AV;
  auto& AU = results_.AU;
  auto& AQs = results_.AQs;

  counts.setZero();
  AV.setZero();
  AU.setZero();
  AQs.setZero();

  const auto& parent_stats = root->stats();  // thread-safe because single-threaded here

  bool provably_winning = parent_stats.provably_winning[seat];
  bool provably_losing = parent_stats.provably_losing[seat];

  for (int i = 0; i < root->stable_data().num_valid_moves; i++) {
    const Edge* edge = lookup_table_.get_edge(root, i);
    Move move = edge->move;
    auto index = PolicyEncoding::to_index(frame, move);

    int count = edge->E;
    int modified_count = count;

    const Node* child = lookup_table_.get_node(edge->child_index);
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
    }

    const auto& child_stable_data = child->stable_data();
    RELEASE_ASSERT(child_stable_data.R_valid);
    ValueArray V = child_stable_data.V();
    eigen_util::chip_assign(AV, eigen_util::reinterpret_as_tensor(V), index);

    // AU: use child's uncertainty estimate from the uncertainty head
    ValueArray U = child_stable_data.U();
    eigen_util::chip_assign(AU, eigen_util::reinterpret_as_tensor(U), index);
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::capture_backup_sample(const Node* root) {
  // Snapshot per-child N/Q/W at the current point during search for backup-NN training.
  // Called at most once per search, when total_count() reaches backup_sample_k_.
  core::seat_index_t seat = root->stable_data().active_seat;
  const auto root_stats = root->stats_safe();

  if (root_stats.W[seat] == 0.f) {
    // No uncertainty at the root, so backup-NN training would be meaningless. Skip the sample.
    return;
  }

  const auto& frame = root_info_.input_encoder.current_frame();

  auto& sample = results_.backup_sample;
  sample.valid = true;
  sample.N.setZero();
  sample.Qs.setZero();
  sample.Ws.setZero();
  sample.P.setZero();
  sample.AVs.setZero();
  sample.AUs.setZero();
  sample.S_baseline = root_stats.S_baseline;
  sample.W_baseline = root_stats.W_baseline;

  for (int i = 0; i < root->stable_data().num_valid_moves; i++) {
    const Edge* edge = lookup_table_.get_edge(root, i);
    if (!edge->E) continue;
    const Node* child = lookup_table_.get_node(edge->child_index);
    if (!child) continue;
    Move move = edge->move;
    auto index = PolicyEncoding::to_index(frame, move);
    const auto& child_stats = child->stats();
    sample.N.coeffRef(index) = edge->E;
    sample.Qs.coeffRef(index) = child_stats.Q(seat);
    sample.Ws.coeffRef(index) = child_stats.W(seat);

    // Note: we use the raw-P rather than the adjusted-P here, since the raw-P should work better
    // for accumulation. With our current implementation, there should be no difference, as the
    // P adjustment mechanisms (Dirichlet noise and softmax temperature) are only applied at the
    // root, and since root Q/W estimates are not use anywhere in the system. However, if we later
    // add softmax temperature to non-root nodes (like LeelaZero and KataGo do), then this choice
    // matters.
    sample.P.coeffRef(index) = edge->policy_prior_prob;

    sample.AVs.coeffRef(index) = edge->child_AV(seat);
    sample.AUs.coeffRef(index) = edge->child_AU(seat);
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::validate_state(Node* node) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  if (node->is_terminal()) return;

  mit::unique_lock lock(node->mutex());

  int N = 1;
  for (int i = 0; i < node->stable_data().num_valid_moves; ++i) {
    auto edge = lookup_table_.get_edge(node, i);
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

template <beta0::concepts::Spec Spec>
void Manager<Spec>::transform_policy(SearchContext& context, LocalPolicyArray& P) {
  core::node_pool_index_t index = context.initialization_index;
  const search::SearchParams& search_params = search_params_;
  const Params& manager_params = params_;

  if (index == root_info_.node_index) {
    if (search_params.full_search) {
      if (manager_params.dirichlet_mult) {
        add_dirichlet_noise(P);
      }
      float temp = root_softmax_temperature_.value();
      if (temp > 0.0f) {
        P = P.pow(1.0f / temp);
      }
      P /= P.sum();
    }
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::add_dirichlet_noise(LocalPolicyArray& P) {
  const Params& manager_params = params_;
  auto& dirichlet_gen = dirichlet_gen_;
  auto& rng = rng_;

  int n = P.rows();
  double alpha = manager_params.dirichlet_alpha_factor / sqrt(n);
  LocalPolicyArray noise = dirichlet_gen.template generate<LocalPolicyArray>(rng, alpha, n);
  P = (1.0 - manager_params.dirichlet_mult) * P + manager_params.dirichlet_mult * noise;
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::prune_policy_target() {
  const search::SearchParams& search_params = search_params_;
  const Params& manager_params = params_;

  if (manager_params.no_model) return;

  const auto& frame = results_.frame;
  const Node* root = lookup_table_.get_node(root_info_.node_index);
  PuctCalculator action_selector(lookup_table_, manager_params, search_params, root, true);

  const auto& P = action_selector.P;
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
    const Edge* edge = lookup_table_.get_edge(root, i);
    const Move& move = edge->move;
    auto index = PolicyEncoding::to_index(frame, move);
    if (mE(i) == 0) {
      results_.policy_target.coeffRef(index) = 0;
      continue;
    }
    if (mE(i) == mE_max) continue;
    if (denom(i) == 0) continue;
    if (mE_floor(i) >= mE(i)) continue;
    auto n = std::max(mE_floor(i), mE(i) - n_forced(i));
    if (n <= 1.0) {
      n = 0;
    }
    results_.policy_target.coeffRef(index) = n;
  }

  if (eigen_util::sum(results_.policy_target) <= 0) {
    results_.policy_target = results_.counts;
  }
}

template <beta0::concepts::Spec Spec>
void Manager<Spec>::print_action_selection_details(const SearchContext& context,
                                                   const PuctCalculator& selector,
                                                   int argmax_index) {
  Node* node = context.visit_node;
  if (search::kEnableSearchDebug) {
    std::ostringstream ss;

    core::seat_index_t seat = node->stable_data().active_seat;

    int n_moves = node->stable_data().num_valid_moves;

    ValueArray players;
    ValueArray nQ = node->stats().Q();
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

    ActionPrinter printer(lookup_table_.get_moves(node));
    for (int i = 0; i < n_moves; ++i) {
      const Edge* edge = lookup_table_.get_edge(node, i);
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

template <beta0::concepts::Spec Spec>
void Manager<Spec>::load_action_symmetries(const Node* root) {
  const auto& stable_data = root->stable_data();
  const State& root_state = root_info_.state;

  State state = root_state;  // copy
  for (int e = 0; e < stable_data.num_valid_moves; ++e) {
    Edge* edge = lookup_table_.get_edge(root, e);
    Game::Rules::apply(state, edge->move);
    InputFrame frame(state);
    action_symmetry_table_builder_.add(edge->move, frame);
    Game::Rules::backtrack_state(state, root_state);
  }

  results_.action_symmetry_table.load(action_symmetry_table_builder_);
  results_.trivial = (results_.action_symmetry_table.num_equivalence_classes() <= 1);
}

}  // namespace beta0
