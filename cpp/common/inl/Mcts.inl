#include <common/Mcts.hpp>

#include <util/EigenTorch.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::StateEvaluation::init(
    const NeuralNet& net, const Tensorizor& tensorizor, const GameState& state, const Result& result,
    common::NeuralNet::input_vec_t& input_vec, float inv_temp)
{
  current_player = state.get_current_player();
  result = result;
  initialized = true;

  if (is_terminal()) return;  // game is over, don't bother computing other fields

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
  auto transform = tensorizor.get_random_symmetry(state);
  transform->transform_input(input.asEigen());
  input_vec[0].toTensor().copy_(input.asTorch());
  net.predict(input_vec, policy.asTorch(), value.asTorch());
  transform->transform_policy(policy.asEigen());

  valid_action_mask = state.get_valid_actions();
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
inline Mcts<GameState, Tensorizor>::Tree::Tree(action_index_t action, Tree* parent)
{
  parent_ = parent;
  action_ = action;
  value_sum_.setZero();
  value_avg_.setZero();
  effective_value_avg_.setZero();
  V_floor_.setZero();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline typename Mcts<GameState, Tensorizor>::GlobalPolicyCountDistr
Mcts<GameState, Tensorizor>::Tree::get_effective_counts() const
{
  player_index_t cp = evaluation_.current_player;
  GlobalPolicyCountDistr counts;
  counts.setZero();
  if (eliminated_) {
    float max_V_floor = get_max_V_floor_among_children();
    for (int i = 0; i < num_children_; ++i) {
      Tree* child = *children_ + i;
      counts(child->action_) = (child->V_floor_(cp) == max_V_floor);
    }
    return counts;
  }
  for (int i = 0; i < num_children_; ++i) {
    Tree* child = *children_ + i;
    counts(child->action_) = child->effective_count();
  }
  return counts;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Tree::expand_children() {
  if (has_children()) return;

  num_children_ = evaluation_.valid_action_mask.count();
  children_ = new Tree*[num_children_];  // TODO: use memory pool
  int i = 0;
  for (auto it : evaluation_.valid_action_mask) {
    children_[i++] = new Tree(*it, this);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Tree::backprop(const ValueProbDistr& result, bool terminal) {
  count_++;
  value_sum_ += result;
  value_avg_ = value_sum_ / count_;
  effective_value_avg_ = has_certain_outcome() ? V_floor_ : value_avg_;

  if (parent_) parent_->backprop(result);
  if (terminal) terminal_backprop(result);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Tree::terminal_backprop(const ValueProbDistr& result) {
  if (is_terminal_result(result)) {
    V_floor_ = result;
  } else {
    player_index_t cp = evaluation_.current_player;
    for (player_index_t p = 0; p < kNumPlayers; ++p) {
      if (p == cp) {
        V_floor_[p] = get_max_V_floor_among_children(p);
      } else {
        V_floor_[p] = get_min_V_floor_among_children(p);
      }
    }
  }

  effective_value_avg_ = has_certain_outcome() ? V_floor_ : value_avg_;
  if (can_be_eliminated()) {
    eliminated_ = true;
    if (parent_) parent_->terminal_backprop();
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts<GameState, Tensorizor>::Tree::get_max_V_floor_among_children(player_index_t p) const {
  float max_V_floor = 0;
  for (int i = 0; i < num_children_; ++i) {
    Tree* child = *children_ + i;
    max_V_floor = std::max(max_V_floor, child->V_floor_(p));
  }
  return max_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts<GameState, Tensorizor>::Tree::get_min_V_floor_among_children(player_index_t p) const {
  float min_V_floor = 1;
  for (int i = 0; i < num_children_; ++i) {
    Tree* child = *children_ + i;
    min_V_floor = std::min(min_V_floor, child->V_floor_(p));
  }
  return min_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Mcts() {
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::clear() {
  throw std::exception();
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
    root_ = new Tree();  // TODO: use memory pool
  }
  throw std::exception();
}


}  // namespace common
