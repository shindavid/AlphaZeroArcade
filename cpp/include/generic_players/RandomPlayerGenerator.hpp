#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/concepts/GameConcept.hpp"
#include "generic_players/RandomPlayer.hpp"

#include <string>
#include <vector>

namespace generic {

template <core::concepts::Game Game>
class RandomPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  RandomPlayerGenerator(core::GameServerBase*) {}

  void set_base_seed(int seed) { base_seed_ = seed; }

  std::string get_default_name() const override { return "Random"; }
  std::vector<std::string> get_types() const override { return {"Random"}; }
  std::string get_description() const override { return "Random player"; }
  core::AbstractPlayer<Game>* generate(core::game_slot_index_t) override {
    return new RandomPlayer<Game>(base_seed_);
  }

 protected:
  int base_seed() const { return base_seed_; }

 private:
  int base_seed_ = -1;
};

}  // namespace generic
