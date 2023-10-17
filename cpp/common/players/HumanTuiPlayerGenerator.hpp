#pragma once

#include <string>
#include <vector>

#include <common/players/HumanTuiPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/GameStateConcept.hpp>

namespace common {

template<core::GameStateConcept GameState>
class HumanTuiPlayerGenerator : public core::AbstractPlayerGenerator<GameState> {
public:
  virtual ~HumanTuiPlayerGenerator() = default;
  std::string get_default_name() const override { return "Human"; }
  std::vector<std::string> get_types() const override { return {"TUI"}; }
  std::string get_description() const override { return "Human player"; }
  virtual core::AbstractPlayer<GameState>* generate(core::game_thread_id_t) = 0;
  int max_simultaneous_games() const override { return 1; }
};

}  // namespace common
