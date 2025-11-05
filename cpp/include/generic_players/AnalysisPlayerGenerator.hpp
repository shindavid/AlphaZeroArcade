#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/concepts/GameConcept.hpp"
#include "generic_players/AnalysisPlayer.hpp"
#include "search/VerboseManager.hpp"

namespace generic {

template <core::concepts::Game Game>
class AnalysisPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  AnalysisPlayerGenerator(core::AbstractPlayerGenerator<Game>* wrapped_generator)
      : wrapped_generator_(wrapped_generator) {}
  ~AnalysisPlayerGenerator() { delete wrapped_generator_; }

  const std::string& get_name() const override { return wrapped_generator_->get_name(); }
  std::string get_default_name() const override { return wrapped_generator_->get_default_name(); }
  std::vector<std::string> get_types() const override { return {""}; }
  std::string get_description() const override { return "Analysis Player"; }
  int max_simultaneous_games() const override { return 1; }

  virtual core::AbstractPlayer<Game>* generate(core::game_slot_index_t id) override {
    auto* wrapped_player = wrapped_generator_->generate_with_name(id);
    return new AnalysisPlayer<Game>(wrapped_player);
  }
  void start_session() override {
    VerboseManager::get_instance()->disable_auto_terminal_printing();
  }

 private:
  core::AbstractPlayerGenerator<Game>* const wrapped_generator_;
};

}  // namespace generic
