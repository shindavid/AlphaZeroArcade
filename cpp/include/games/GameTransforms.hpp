#pragma once

#include <core/concepts/Game.hpp>

/*
 * This file contains metafunctions that create Game types from other Game types.
 */

namespace game_transform {

/*
 * AddStateStorage is a game transformation that adds state storage to a game by setting
 * MctsConfiguration::kStoreStates to true.
 */

template <core::concepts::Game Game>
struct AddStateStorage : public Game {
  struct MctsConfiguration : public Game::MctsConfiguration {
    static constexpr bool kStoreStates = true;
  };
};

}  // namespace game_transform
