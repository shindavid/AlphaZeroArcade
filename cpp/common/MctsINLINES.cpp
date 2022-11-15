#include <common/Mcts.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept Tensorizor>
inline Mcts<GameState, Tensorizor>::StateEvaluation::StateEvaluation(
    const NeuralNet& net, const Tensorizor& tensorizor, const GameState& state, const Result& result)
{

}

}  // namespace common
