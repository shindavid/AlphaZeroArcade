#include <common/Mcts.hpp>

#include <thread>
#include <utility>
#include <vector>

#include <util/EigenTorch.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::NNEvaluation::NNEvaluation(
    const ValueVector& value, const PolicyVector& policy, const ActionMask& valid_actions, float inv_temp)
{
  int num_valid_actions = valid_actions.count();
  local_policy_prob_distr_.resize(num_valid_actions);
  int i = 0;
  for (auto it : valid_actions) {
    local_policy_prob_distr_[i++] = policy(it);
  }
  value_prob_distr_ = eigen_util::softmax(value);
  local_policy_prob_distr_ = eigen_util::softmax(local_policy_prob_distr_ * inv_temp);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(
    const Tensorizor& tensorizor, const GameState& state, const GameResult& result, Node* parent,
    symmetry_index_t sym_index, action_index_t action)
: tensorizor_(tensorizor)
, state_(state)
, result_(result)
, valid_action_mask_(state.get_valid_actions())
, parent_(parent)
, sym_index_(sym_index)
, action_(action)
, current_player_(state.get_current_player()) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(const stable_data_t& data, bool prune_parent)
{
  *this = data;
  if (prune_parent) parent_ = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::stats_t::stats_t() {
  value_avg_.setZero();
  effective_value_avg_.setZero();
  V_floor_.setZero();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::Node(
    const Tensorizor& tensorizor, const GameState& state, const GameResult& result, symmetry_index_t sym_index,
    Node* parent, action_index_t action)
: stable_data_(tensorizor, state, result, parent, sym_index, action) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::Node(const Node& node, bool prune_parent)
: stable_data_(node.stable_data_, prune_parent)
, children_data_(node.children_data_)
, evaluation_(node.evaluation_)
, stats_(node.stats_) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::release(Node* protected_child) {
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    if (child != protected_child) child->release();
  }

  if (children_data_.first_child_) delete[] children_data_.first_child_;
  delete this;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline typename Mcts_<GameState, Tensorizor>::GlobalPolicyCountDistr
Mcts_<GameState, Tensorizor>::Node::get_effective_counts() const
{
  std::lock_guard<std::mutex> guard(mutex_);

  player_index_t cp = stable_data_.current_player;
  GlobalPolicyCountDistr counts;
  counts.setZero();
  if (stats_.eliminated_) {
    float max_V_floor = get_max_V_floor_among_children();
    for (int i = 0; i < children_data_.num_children_; ++i) {
      Node* child = children_data_.first_child_ + i;
      counts(child->action_) = (child->V_floor_(cp) == max_V_floor);
    }
    return counts;
  }
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    counts(child->action_) = child->effective_count();
  }
  return counts;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::expand_children() {
  std::lock_guard<std::mutex> guard(mutex_);

  if (has_children()) return;

  // TODO: use object pool
  children_data_.num_children_ = stable_data_.valid_action_mask_.count();
  void* raw_memory = operator new[](children_data_.num_children_ * sizeof(Node));
  Node* node = static_cast<Node*>(raw_memory);
  children_data_.first_child_ = node;
  for (auto it : stable_data_.valid_action_mask_) {
    action_index_t action = it;
    Tensorizor tensorizor_copy = stable_data_.tensorizor_;
    GameState state_copy = stable_data_.state_;

    symmetry_index_t sym_index = tensorizor_copy.get_random_symmetry_index(state_copy);
    GameResult result = state_copy.apply_move(action);
    new(node++) Node(tensorizor_copy, state_copy, result, sym_index, this, action);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::backprop(const ValueProbDistr& result, bool terminal) {
  {
    std::lock_guard<std::mutex> guard(mutex_);

    stats_.value_avg_ = (stats_.value_avg_ * stats_.count_ + result) / (stats_.count_ + 1);
    stats_.count_++;
    stats_.effective_value_avg_ = has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
  }

  if (parent()) parent()->backprop(result);
  if (terminal) terminal_backprop(result);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::terminal_backprop(const ValueProbDistr& result) {
  bool recurse = false;
  {
    std::lock_guard<std::mutex> guard(mutex_);

    if (is_terminal_result(result)) {
      stats_.V_floor_ = result;
    } else {
      player_index_t cp = stable_data_.current_player_;
      for (player_index_t p = 0; p < kNumPlayers; ++p) {
        if (p == cp) {
          stats_.V_floor_[p] = get_max_V_floor_among_children(p);
        } else {
          stats_.V_floor_[p] = get_min_V_floor_among_children(p);
        }
      }
    }

    stats_.effective_value_avg_ = has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
    if (can_be_eliminated()) {
      stats_.eliminated_ = true;
      recurse = parent();
    }
  }

  if (recurse) {
    parent()->terminal_backprop(result);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts_<GameState, Tensorizor>::Node*
Mcts_<GameState, Tensorizor>::Node::find_child(action_index_t action) const {
  // TODO: technically we can do a binary search here, as children should be in sorted order by action
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node *child = children_data_.first_child_ + i;
    if (child->stable_data_.action_ == action) return child;
  }
  return nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts_<GameState, Tensorizor>::Node::get_max_V_floor_among_children(player_index_t p) const {
  float max_V_floor = 0;
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    std::lock_guard<std::mutex> guard(child->mutex_);
    max_V_floor = std::max(max_V_floor, child->V_floor()(p));
  }
  return max_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts_<GameState, Tensorizor>::Node::get_min_V_floor_among_children(player_index_t p) const {
  float min_V_floor = 1;
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    std::lock_guard<std::mutex> guard(child->mutex_);
    min_V_floor = std::min(min_V_floor, child->V_floor()(p));
  }
  return min_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::SearchThread::SearchThread(Mcts_* mcts, int thread_id)
: mcts_(mcts)
, thread_id_(thread_id)
{
  throw std::exception();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::SearchThread::run() {
  throw std::exception();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::NNEvaluationThread::NNEvaluationThread(
    NeuralNet& net, int batch_size, int64_t timeout_ns, int cache_size)
: net_(net)
, policy_batch_(batch_size, kNumGlobalActions, util::to_std_array<int>(batch_size, kNumGlobalActions))
, value_batch_(batch_size, kNumPlayers, util::to_std_array<int>(batch_size, kNumPlayers))
, input_batch_(util::to_std_array<int>(batch_size, util::std_array_v<int, typename Tensorizor::Shape>))
, cache_(cache_size)
, timeout_ns_(timeout_ns)
, batch_size_limit_(batch_size)
{
  evaluation_data_arr_ = new evaluation_data_t[batch_size];
  evaluation_pool_.reserve(4096);
  torch_input_gpu_ = input_batch_.asTorch().clone().to(torch::kCUDA);
  input_vec_.push_back(torch_input_gpu_);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts_<GameState, Tensorizor>::NNEvaluationThread::evaluate(
    const Tensorizor& tensorizor, const GameState& state, symmetry_index_t sym_index, float inv_temp,
    NNEvaluation** eval_ptr)
{
  cache_key_t key{state, inv_temp, sym_index};
  auto cached = cache_.get(key);
  if (cached.has_value()) {
    *eval_ptr = cached.value();
    return;
  }

  auto& input = input_batch_.template eigenSlab<typename TensorizorTypes::Shape>(batch_write_index_);
  tensorizor.tensorize(input, state);
  auto transform = tensorizor.get_symmetry(sym_index);
  transform->transform_input(input);

  evaluation_data_t edata{eval_ptr, state.get_valid_actions(), inv_temp, sym_index};
  evaluation_data_arr_[batch_write_index_] = edata;
  ++batch_write_index_;

  // TODO: make this batched/asynchronous, respect timeout_ns_
  if (batch_write_index_ < batch_size_limit_) return;

  torch_input_gpu_.copy_(input_batch_.asTorch());
  net_.predict(input_vec_, policy_batch_.asTorch(), value_batch_.asTorch());

  for (int i = 0; i < batch_write_index_; ++i) {
    const evaluation_data_t& edata_i = evaluation_data_arr_[i];
    auto& policy_i = policy_batch_.eigenSlab(i);
    auto& value_i = value_batch_.eigenSlab(i);

    tensorizor.get_symmetry(edata_i.sym_index)->transform_policy(policy_i);
    evaluation_pool_.template emplace_back(
        eigen_util::to_vector(value_i), eigen_util::to_vector(policy_i), edata_i.valid_actions, edata_i.inv_temp);
    *eval_ptr = &evaluation_pool_.back();  // FIXME: this assumes batch_size_limit_ is 1
  }
  batch_write_index_ = 0;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Mcts_(NeuralNet& net, int batch_size, int64_t timeout_ns, int cache_size)
: nn_eval_thread_(net, batch_size, timeout_ns, cache_size) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::clear() {
  root_->release();
  root_ = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::receive_state_change(
    player_index_t player, const GameState& state, action_index_t action, const GameResult& result)
{
  if (!root_) return;

  Node* new_root = root_->find_child(action);
  if (!new_root) {
    root_->release();
    root_ = nullptr;
    return;
  }

  Node* new_root_copy = new Node(*new_root, true);
  root_->release(new_root);
  root_ = new_root_copy;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline const typename Mcts_<GameState, Tensorizor>::MctsResults* Mcts_<GameState, Tensorizor>::sim(
    const Tensorizor& tensorizor, const GameState& game_state, const Params& params)
{
  if (!params.can_reuse_subtree() || !root_) {
    auto result = make_non_terminal_result<kNumPlayers>();
    symmetry_index_t sym_index = tensorizor.get_random_symmetry_index(game_state);
    root_ = new Node(tensorizor, game_state, result, sym_index);  // TODO: use memory pool
  }

  if (params.num_threads == 1) {
    // run everything in main thread for simplicity
    while (root_->effective_count() < params.tree_size_limit && !root_->eliminated()) {
      GameState game_state_copy(game_state);
      Tensorizor tensorizor_copy(tensorizor);
      visit(root_, tensorizor_copy, game_state_copy, params, 1);
    }

    throw std::exception();
  } else {
    std::vector<std::thread> threads;
    for (int i = 0; i < params.num_threads; ++i) {
      SearchThread thread(this, i);
      threads.emplace_back(run_search, this, i);
    }

    for (auto& th : threads) th.join();

    throw std::exception();
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::visit(
    Node* tree, const Tensorizor& tensorizor, const GameState& state, const Params& params, int depth)
{
  const auto& result = tree->result();
  if (is_terminal_result(result)) {
    tree->backprop(result, params.allow_eliminations);
    return;
  }

  float inv_temp = (1.0 / params.root_softmax_temperature) ? tree->is_root() : 1.0;
  symmetry_index_t sym_index = tree->get_sym_index();

  // TODO: make this block on asynchronous call via condition_variable
  NNEvaluation* evaluation = nullptr;
  nn_eval_thread_.evaluate(tensorizor, state, sym_index, inv_temp, &evaluation);

//  bool leaf = tree->is_leaf();
  tree->expand_children();

//  auto cPUCT = params.cPUCT;
  LocalPolicyProbDistr P = evaluation->local_policy_prob_distr();

  LocalPolicyProbDistr noise;
  if (tree->is_root() && params.dirichlet_mult) {
//    noise = dirichlet();
    P = (1.0 - params.dirichlet_mult) * P + params.dirichlet_mult * noise;
  } else {  // TODO - only need to setZero() if generating debug file
    noise = P;
    noise.setZero();
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::run_search(Mcts_* mcts, int thread_id) {
  SearchThread thread(mcts, thread_id);
  thread.run();
}

}  // namespace common
