#include <common/Mcts.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::StateEvaluation::StateEvaluation(
    const NeuralNet& net, const Tensorizor& tensorizor, const GameState& state, const Result& result)
{

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

}  // namespace common
