#pragma once

#include <string>
#include <vector>

#include <common/AbstractPlayerGenerator.hpp>
#include <common/BasicTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/players/RandomPlayer.hpp>

namespace common {

template<GameStateConcept GameState>
class RandomPlayerGenerator : public AbstractPlayerGenerator<GameState> {
public:
  std::vector<std::string> get_types() const override { return {"Random"}; }
  std::string get_description() const override { return "Random player"; }
  AbstractPlayer<GameState>* generate(game_thread_id_t) override { return new RandomPlayer<GameState>(); }
};

}  // namespace common
