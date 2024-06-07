#pragma once

#include <string>
#include <vector>

#include <generic_players/HumanTuiPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/GameStateConcept.hpp>

namespace generic {

template<core::concepts::Game Game>
class HumanTuiPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
public:
  virtual ~HumanTuiPlayerGenerator() = default;
  std::string get_default_name() const override { return "Human"; }
  std::vector<std::string> get_types() const override { return {"TUI"}; }
  std::string get_description() const override { return "Human player"; }
  virtual core::AbstractPlayer<Game>* generate(core::game_thread_id_t) = 0;
  int max_simultaneous_games() const override { return 1; }
};

}  // namespace generic
