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
  AbstractPlayer<GameState>* generate(void* play_address) override { return new HumanTuiPlayer<GameState>(); }
};

}  // namespace common
