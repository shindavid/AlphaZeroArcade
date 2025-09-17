#pragma once

#include "alphazero/GameLogCompactRecord.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct GameLogCompactRecord : public alpha0::GameLogCompactRecord<Game> {
  float Q_prior;
  float Q_posterior;
};

}  // namespace beta0
