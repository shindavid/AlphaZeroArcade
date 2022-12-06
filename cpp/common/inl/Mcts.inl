#include <common/Mcts.hpp>

#include <util/EigenTorch.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluation::NNEvaluation(
    const NeuralNet& net, const Tensorizor& tensorizor, const GameState& state,
    common::NeuralNet::input_vec_t& input_vec, symmetry_index_t symmetry_index, float inv_temp)
{
  using PolicyVector = typename GameStateTypes<GameState>::PolicyVector;
  using ValueVector = typename GameStateTypes<GameState>::ValueVector;
  using InputTensor = typename Tensorizor::InputTensor;

  using EigenTorchPolicy = eigentorch::to_eigentorch_t<PolicyVector>;
  using EigenTorchValue = eigentorch::to_eigentorch_t<ValueVector>;
  using EigenTorchInput = eigentorch::to_eigentorch_t<InputTensor>;

  EigenTorchPolicy policy;
  EigenTorchValue value;
  EigenTorchInput input;

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
inline Mcts<GameState, Tensorizor>::Node::stats_t::stats_t() {
//  value_sum_.setZero();
  value_avg_.setZero();
  effective_value_avg_.setZero();
  V_floor_.setZero();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::Node(
    const GameState& state, const Result& result, Node* parent, action_index_t action)
: stable_data_(state, result, parent, action) {}

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
      Node* child = *children_data_.first_child_ + i;
      counts(child->action_) = (child->V_floor_(cp) == max_V_floor);
    }
    return counts;
  }
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = *children_data_.children_ + i;
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
inline float Mcts<GameState, Tensorizor>::Node::get_max_V_floor_among_children(player_index_t p) const {
  float max_V_floor = 0;
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = *children_data_.first_child_ + i;
    max_V_floor = std::max(max_V_floor, child->V_floor_(p));
  }
  return max_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts<GameState, Tensorizor>::Node::get_min_V_floor_among_children(player_index_t p) const {
  float min_V_floor = 1;
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = *children_data_.first_child_ + i;
    min_V_floor = std::min(min_V_floor, child->V_floor_(p));
  }
  return min_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Mcts(NeuralNet& net) : net_(net) {
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::clear() {
  root_ = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::receive_state_change(
    player_index_t player, const GameState& state, action_index_t action, const Result& result)
{
  throw std::exception();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline const Mcts<GameState, Tensorizor>::Results* Mcts<GameState, Tensorizor>::sim(
    const Tensorizor& tensorizor, const GameState& game_state, const Params& params)
{
  if (!params.can_reuse_subtree() || !root_) {
    auto result = make_non_terminal_result<kNumPlayers>();
    root_ = new Node(game_state, result);  // TODO: use memory pool
  }
  throw std::exception();
}


}  // namespace common
