#pragma once

#include <string>
#include <vector>

#include <core/AbstractPlayerGenerator.hpp>
#include <core/GameStateConcept.hpp>
#include <core/players/HumanTuiPlayer.hpp>

namespace common {

template<GameStateConcept GameState>
class HumanTuiPlayerGenerator : public AbstractPlayerGenerator<GameState> {
public:
  virtual ~HumanTuiPlayerGenerator() = default;
  std::vector<std::string> get_types() const override { return {"TUI"}; }
  std::string get_description() const override { return "Human player"; }
  virtual AbstractPlayer<GameState>* generate(game_thread_id_t) = 0;
  int max_simultaneous_games() const override { return 1; }
};

}  // namespace common
