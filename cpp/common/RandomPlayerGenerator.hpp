#pragma once

#include <string>
#include <vector>

#include <common/RandomPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameStateConcept.hpp>

namespace core {

template<GameStateConcept GameState>
class RandomPlayerGenerator : public AbstractPlayerGenerator<GameState> {
public:
  std::vector<std::string> get_types() const override { return {"Random"}; }
  std::string get_description() const override { return "Random player"; }
  AbstractPlayer<GameState>* generate(game_thread_id_t) override { return new RandomPlayer<GameState>(); }
};

}  // namespace core
