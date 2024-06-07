#pragma once

#include <string>
#include <vector>

#include <generic_players/RandomPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameStateConcept.hpp>

namespace generic {

template <core::concepts::Game Game>
class RandomPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  std::string get_default_name() const override { return "Random"; }
  std::vector<std::string> get_types() const override { return {"Random"}; }
  std::string get_description() const override { return "Random player"; }
  core::AbstractPlayer<Game>* generate(core::game_thread_id_t) override {
    return new RandomPlayer<Game>();
  }
};

}  // namespace generic
