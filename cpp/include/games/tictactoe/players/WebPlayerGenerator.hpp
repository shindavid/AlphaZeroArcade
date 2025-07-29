#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/players/WebPlayer.hpp"

namespace tictactoe {

class WebPlayerGenerator : public core::AbstractPlayerGenerator<tictactoe::Game> {
 public:
  WebPlayerGenerator(core::GameServerBase*) {}

  std::string get_default_name() const override { return "Human"; }
  std::vector<std::string> get_types() const override { return {"web"}; }
  std::string get_description() const override { return "Web player"; }
  int max_simultaneous_games() const override { return 1; }

  core::AbstractPlayer<tictactoe::Game>* generate(core::game_slot_index_t) override {
    return new tictactoe::WebPlayer();
  }
};

}  // namespace tictactoe
