#include <core/GameStateHistory.hpp>

#include <cstring>

namespace core {

template <GameStateConcept GameState, int BufferSize>
void GameStateHistory<GameState, BufferSize>::clear() {
  start_ = 0;
  size_ = 0;
}

template <GameStateConcept GameState, int BufferSize>
void GameStateHistory<GameState, BufferSize>::push_back(const GameStateData& data) {
  circ_buffer_[(start_ + size_) % BufferSize] = data;
  if (size_ < BufferSize) {
    size_++;
  } else {
    start_ = (start_ + 1) % BufferSize;
  }
}

template <GameStateConcept GameState, int BufferSize>
template<typename Transform>
void GameStateHistory<GameState, BufferSize>::apply(Transform* transform) {
  for (int i = 0; i < size_; ++i) {
    transform->apply((*this)[i]);
  }
}

template <GameStateConcept GameState, int BufferSize>
template <typename Transform>
void GameStateHistory<GameState, BufferSize>::undo(Transform* transform) {
  for (int i = 0; i < size_; ++i) {
    transform->undo((*this)[i]);
  }
}

}  // namespace core
