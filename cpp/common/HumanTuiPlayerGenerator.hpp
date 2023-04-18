#pragma once

#include <string>
#include <vector>

#include <common/AbstractPlayerGenerator.hpp>
#include <common/GameStateConcept.hpp>
#include <common/HumanTuiPlayer.hpp>

namespace common {

template<GameStateConcept GameState>
class HumanTuiPlayerGenerator : public AbstractPlayerGenerator<GameState> {
public:
  std::vector<std::string> get_types() const override { return {"TUI"}; }
  std::string get_description() const override { return "Human player"; }
  AbstractPlayer<GameState>* generate(game_thread_id_t) override { return new HumanTuiPlayer<GameState>(); }
  int max_simultaneous_games() const override { return 1; }
};

}  // namespace common
