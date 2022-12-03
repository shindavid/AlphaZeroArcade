#include <common/Mcts.hpp>

#include <util/EigenTorch.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::StateEvaluation::StateEvaluation(
    const NeuralNet& net, const Tensorizor& tensorizor, const GameState& state, const Result& result,
    common::NeuralNet::input_vec_t& input_vec)
: current_player_(state.get_current_player())
, result_(result)
{
  if (is_terminal_result(result)) return;  // game is over, don't bother computing other fields

  using InputTensor = typename Tensorizor::InputTensor;
  InputTensor input;
  tensorizor.tensorize(input, state);
  auto transform = tensorizor.get_random_symmetry(state);
  transform->transform_input(input);

  auto torch_input = eigentorch::eigen2torch(input);
  input_vec[0].toTensor().copy_(torch_input);
//  net.predict(input_vec, torch_policy_, torch_value_);

  throw std::exception();
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
//  if (!params.can_reuse_subtree() || )
  throw std::exception();
}


}  // namespace common
