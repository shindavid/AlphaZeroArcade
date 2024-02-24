#pragma once

#include <string>
#include <vector>

#include <games/generic/players/RandomPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameStateConcept.hpp>

namespace generic {

template <core::GameStateConcept GameState>
class RandomPlayerGenerator : public core::AbstractPlayerGenerator<GameState> {
 public:
  std::string get_default_name() const override { return "Random"; }
  std::vector<std::string> get_types() const override { return {"Random"}; }
  std::string get_description() const override { return "Random player"; }
  core::AbstractPlayer<GameState>* generate(core::game_thread_id_t) override {
    return new RandomPlayer<GameState>();
  }
};

}  // namespace generic
