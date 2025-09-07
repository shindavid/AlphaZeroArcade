#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/GameServerBase.hpp"
#include "core/concepts/GameConcept.hpp"

#include <string>
#include <vector>

namespace generic {

template <core::concepts::Game Game>
class HumanTuiPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  HumanTuiPlayerGenerator(core::GameServerBase*) {}
  virtual ~HumanTuiPlayerGenerator() = default;

  std::string get_default_name() const override { return "Human"; }
  std::vector<std::string> get_types() const override { return {"TUI"}; }
  std::string get_description() const override { return "Human player"; }
  virtual core::AbstractPlayer<Game>* generate(core::game_slot_index_t) override = 0;
  int max_simultaneous_games() const override { return 1; }
};

}  // namespace generic
