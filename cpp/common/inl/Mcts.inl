#include <common/Mcts.hpp>

#include <util/EigenTorch.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::StateEvaluation::init(
    const NeuralNet& net, const Tensorizor& tensorizor, const GameState& state, const Result& result,
    common::NeuralNet::input_vec_t& input_vec)
{
  current_player_ = state.get_current_player();
  result_ = result;
  initialized_ = true;

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

  valid_action_mask_ = state.get_valid_actions();
  int num_valid_actions = valid_action_mask_.count();
  local_policy_prob_distr_.resize(num_valid_actions);
  int i = 0;
  for (auto it : valid_action_mask_) {
    local_policy_prob_distr_[i++] = policy.asEigen()[*it];
  }
  local_policy_prob_distr_ = eigen_util::softmax(local_policy_prob_distr_);
  value_prob_distr_ = eigen_util::softmax(value.asEigen());
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
inline Mcts<GameState, Tensorizor>::Mcts()
{
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::clear()
{
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
