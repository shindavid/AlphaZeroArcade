#pragma once

#include <core/concepts/Game.hpp>

namespace game_transform {

template <core::concepts::Game Game>
struct AddStateStorage : public Game {
  struct MctsConfiguration : public Game::MctsConfiguration {
    static constexpr bool kStoreStates = true;
  };
};

}  // namespace game_wrapper

