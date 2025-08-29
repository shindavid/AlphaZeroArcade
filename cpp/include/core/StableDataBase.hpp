#pragma once

#include "core/concepts/Game.hpp"
#include "util/CppUtil.hpp"

namespace core {

/*
 * StableDataBase<Game, false> is an empty class that does nothing.
 *
 * StableDataBase<Game, true> is a class that stores a game state.
 *
 * StableData<Game> inherits from StableDataBase<Game, B>, with B set to true only if the macro
 * STORE_STATES is enabled OR if the Game's MctsConfiguration has kStoreStates set to true (this is
 * used in unit tests).
 *
 * This allows for us to store the game state in the node object, which can be useful for debugging
 * and analysis.
 *
 * Note that StableDataBase<Game, false> is an empty base-class, allowing us to get the empty
 * base-class optimization in StableData<Game>.
 */
template <core::concepts::Game Game, bool EnableStorage>
struct StableDataBase {
  using State = Game::State;

  StableDataBase(const State&) {}
  const State* get_state() const { return nullptr; }
};

template <core::concepts::Game Game>
struct StableDataBase<Game, true> {
  using State = Game::State;

  StableDataBase(const State& s) : state(s) {}
  const State* get_state() const { return &state; }

  State state;
};

template <core::concepts::Game Game>
constexpr bool kStoreStates = IS_DEFINED(STORE_STATES) || Game::MctsConfiguration::kStoreStates;

template <core::concepts::Game Game>
using StableDataBaseImpl = StableDataBase<Game, kStoreStates<Game>>;

}  // namespace core
