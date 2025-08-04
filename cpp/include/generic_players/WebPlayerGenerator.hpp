#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"

namespace generic {

template <class WebPlayer>
class WebPlayerGenerator : public core::AbstractPlayerGenerator<typename WebPlayer::GameClass> {
 public:
  using Game = WebPlayer::GameClass;
  WebPlayerGenerator(core::GameServerBase*) {}

  std::string get_default_name() const override { return "Human"; }
  std::vector<std::string> get_types() const override { return {"web"}; }
  std::string get_description() const override { return "Web player"; }
  int max_simultaneous_games() const override { return 1; }

  virtual core::AbstractPlayer<Game>* generate(core::game_slot_index_t) override {
    return new WebPlayer();
  }
};

}  // namespace generic
