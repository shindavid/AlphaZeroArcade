#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>

namespace core {

template <concepts::Game Game, int BufferSize>
class GameStateHistory {
 public:
  using GameStateData = typename GameState::Data;

  void clear();
  void push_back(const GameStateData&);

  template<typename Transform> void apply(Transform*);
  template<typename Transform> void undo(Transform*);
  const GameStateData& operator[](int i) const { return circ_buffer_[(start_ + i) % BufferSize]; }

 protected:
  GameStateData circ_buffer_[BufferSize];
  int start_ = 0;
  int size_ = 0;
};

// trivial specialization for BufferSize = 0
template <concepts::Game Game>
class GameStateHistory<GameState, 0> {
 public:
  using GameStateData = typename GameState::Data;

  void clear() {}
  void push_back(const GameStateData&) {}

  template<typename Transform> void apply(Transform*) {}
  template<typename Transform> void undo(Transform*) {}
};

namespace concepts {

template<typename T>
struct IsGameStateHistory {
  static constexpr bool value = false;
};

template<concepts::Game Game, int BufferSize>
struct IsGameStateHistory<core::GameStateHistory<GameState, BufferSize>> {
  static constexpr bool value = true;
};

template <typename T>
concept GameStateHistory = IsGameStateHistory<T>::value;

}  // namespace concepts

}  // namespace core

#include <inline/core/GameStateHistory.inl>
