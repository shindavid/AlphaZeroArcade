#pragma once

#include <core/concepts/Game.hpp>

template <core::concepts::Game Game>
struct StateStoring : public Game {
  struct MctsConfiguration : public Game::MctsConfiguration {
    static constexpr bool kStoreStates = true;
  };
};
