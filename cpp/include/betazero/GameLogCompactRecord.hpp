#pragma once

#include "alphazero/GameLogCompactRecord.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct GameLogCompactRecord : public alpha0::GameLogCompactRecord<Game> {
  using ValueTensor = Game::Types::ValueTensor;
  ValueTensor Q_posterior;
};

}  // namespace beta0
