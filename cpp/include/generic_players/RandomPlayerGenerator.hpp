#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/concepts/Game.hpp"
#include "generic_players/RandomPlayer.hpp"

#include <string>
#include <vector>

namespace generic {

template <core::concepts::Game Game>
class RandomPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  RandomPlayerGenerator(core::GameServerBase*) {}

  std::string get_default_name() const override { return "Random"; }
  std::vector<std::string> get_types() const override { return {"Random"}; }
  std::string get_description() const override { return "Random player"; }
  core::AbstractPlayer<Game>* generate(core::game_slot_index_t) override {
    return new RandomPlayer<Game>();
  }
};

}  // namespace generic
