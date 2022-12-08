#include <common/Mcts.hpp>

#include <thread>
#include <vector>

#include <util/EigenTorch.hpp>

namespace common {

/*
template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluation::NNEvaluation(
    const NeuralNet& net, const Tensorizor& tensorizor, const GameState& state,
    common::NeuralNet::input_vec_t& input_vec, symmetry_index_t symmetry_index, float inv_temp)
{
  using PolicyVector = typename GameStateTypes<GameState>::PolicyVector;
  using ValueVector = typename GameStateTypes<GameState>::ValueVector;
  using InputTensor = typename Tensorizor::InputTensor;

  PolicyVector policy;
  ValueVector value;
  InputTensor input;

  tensorizor.tensorize(input.toEigen(), state);
  auto transform = tensorizor.get_symmetry(state, symmetry_index);
  transform->transform_input(input.asEigen());
  input_vec[0].toTensor().copy_(input.asTorch());
  net.predict(input_vec, policy.asTorch(), value.asTorch());
  transform->transform_policy(policy.asEigen());

  auto valid_action_mask = state.get_valid_actions();
  int num_valid_actions = valid_action_mask.count();
  local_policy_prob_distr.resize(num_valid_actions);
  int i = 0;
  for (auto it : valid_action_mask) {
    local_policy_prob_distr[i++] = policy.asEigen()[*it];
  }
  value_prob_distr = eigen_util::softmax(value.asEigen());
  local_policy_prob_distr = eigen_util::softmax(local_policy_prob_distr * inv_temp);
}
 */

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(
    const GameState& state, const Result& result, Node* parent, action_index_t action)
: state_(state)
, result_(result)
, valid_action_mask_(state.get_valid_actions())
, parent_(parent)
, action_(action)
, current_player_(state.get_current_player()) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(const stable_data_t& data, bool prune_parent)
{
  *this = data;
  if (prune_parent) parent_ = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::stats_t::stats_t() {
  value_avg_.setZero();
  effective_value_avg_.setZero();
  V_floor_.setZero();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::Node(
    const GameState& state, const Result& result, Node* parent, action_index_t action)
: stable_data_(state, result, parent, action) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::Node(const Node& node, bool prune_parent)
: stable_data_(node.stable_data_, prune_parent)
, children_data_(node.children_data_)
, evaluation_(node.evaluation_)
, stats_(node.stats_) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::release(Node* protected_child) {
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    if (child != protected_child) child->release();
  }

  if (children_data_.first_child_) delete[] children_data_.first_child_;
  delete this;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline typename Mcts<GameState, Tensorizor>::GlobalPolicyCountDistr
Mcts<GameState, Tensorizor>::Node::get_effective_counts() const
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
inline void Mcts<GameState, Tensorizor>::Node::expand_children() {
  std::lock_guard<std::mutex> guard(mutex_);

  if (has_children()) return;

  // TODO: use object pool
  children_data_.num_children_ = stable_data_.valid_action_mask_.count();
  void* raw_memory = operator new[](children_data_.num_children_ * sizeof(Node));
  Node* node = static_cast<Node*>(raw_memory);
  children_data_.first_child_ = node;
  for (auto it : stable_data_.valid_action_mask) {
    action_index_t action = *it;
    // TODO: consider making the state copy and applying move lazily?
    GameState state_copy = stable_data_.state_;
    Result result = state_copy.apply_move(action);
    new(node++) Node(state_copy, result, this, action);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::backprop(const ValueProbDistr& result, bool terminal) {
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
inline void Mcts<GameState, Tensorizor>::Node::terminal_backprop(const ValueProbDistr& result) {
  bool recurse = false;
  {
    std::lock_guard<std::mutex> guard(mutex_);

    if (is_terminal_result(result)) {
      stats_.V_floor_ = result;
    } else {
      player_index_t cp = stable_data_.current_player;
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
    parent()->terminal_backprop();
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
Mcts<GameState, Tensorizor>::Node* Mcts<GameState, Tensorizor>::Node::find_child(action_index_t action) const {
  // TODO: technically we can do a binary search here, as children should be in sorted order by action
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node *child = children_data_.first_child_ + i;
    if (child->stable_data_.action_ == action) return child;
  }
  return nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts<GameState, Tensorizor>::Node::get_max_V_floor_among_children(player_index_t p) const {
  float max_V_floor = 0;
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    max_V_floor = std::max(max_V_floor, child->V_floor_(p));
  }
  return max_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts<GameState, Tensorizor>::Node::get_min_V_floor_among_children(player_index_t p) const {
  float min_V_floor = 1;
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    min_V_floor = std::min(min_V_floor, child->V_floor_(p));
  }
  return min_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::SearchThread::SearchThread(Mcts* mcts, int thread_id)
: mcts_(mcts)
, thread_id_(thread_id)
{
  throw std::exception();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::run() {
  throw std::exception();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluationThread::NNEvaluationThread(
    NeuralNet& net, int batch_size, int64_t timeout_ns, int cache_size)
: net_(net)
, policy_(batch_size, kNumGlobalActions, util::to_std_array<int>(batch_size, kNumGlobalActions))
, value_(batch_size, kNumPlayers, util::to_std_array<int>(batch_size, kNumPlayers))
, input_(util::to_std_array<int>(batch_size, util::std_array_v<int, typename Tensorizor::Shape>))
, cache_(cache_size)
, timeout_ns_(timeout_ns)
, batch_size_(batch_size)
{
  torch_input_gpu_ = input_.asTorch().clone().to(torch::kCUDA);
  input_vec_.push_back(torch_input_gpu_);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationThread::evaluate(
    const Tensorizor& tensorizor, const GameState& state, symmetry_index_t index)
{
  cache_key_t key{state, index};
  auto cached = cache_.get(key);
  if (cached.has_value()) {
    return cached.value();
  }

  auto& input = input_.eigenSlab<TensorizorTypes::Shape>(batch_write_index_);
  ++batch_write_index_;
  if (batch_write_index_ == batch_size_) {
    // TODO: make net evaluation, update cache, write to memory where search threads can read, notify search
    // threads
  }

  // TODO: respect timeout_ns_
  throw std::exception();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Mcts(NeuralNet& net, int batch_size, int64_t timeout_ns, int cache_size)
: nn_eval_thread_(net, batch_size, timeout_ns, cache_size) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::clear() {
  root_->release();
  root_ = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::receive_state_change(
    player_index_t player, const GameState& state, action_index_t action, const Result& result)
{
  if (root_) {
    Node* new_root = root_->find_child(action);
    if (new_root) {
      Node* new_root_copy = new Node(*new_root, true);
      root_->release(new_root);
      root_ = new_root_copy;
    } else {
      root_ = new Node(state, result);
    }
  } else {
    root_ = new Node(state, result);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline const Mcts<GameState, Tensorizor>::Results* Mcts<GameState, Tensorizor>::sim(
    const Tensorizor& tensorizor, const GameState& game_state, const Params& params)
{
  if (!params.can_reuse_subtree() || !root_) {
    auto result = make_non_terminal_result<kNumPlayers>();
    root_ = new Node(game_state, result);  // TODO: use memory pool
  }

  if (params.num_threads == 1) {
    // run everything in main thread for simplicity
    while (root_->effective_count() < params.tree_size_limit && !root_->eliminated()) {
      visit(root_, tensorizor, game_state, params, 1);
    }
  } else {
    std::vector<std::thread> threads;
    for (int i = 0; i < params.num_threads; ++i) {
      SearchThread thread(this, i);
      threads.emplace_back(run_search, this, i);
    }

    for (auto& th : threads) th.join();
  }
  throw std::exception();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::visit(
    Node* tree, const Tensorizor& tensorizor, const GameState& state, const Params& params, int depth)
{

}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::run_search(Mcts* mcts, int thread_id) {
  SearchThread thread(mcts, thread_id);
  thread.run();
}

}  // namespace common
