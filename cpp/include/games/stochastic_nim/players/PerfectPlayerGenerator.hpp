#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "games/stochastic_nim/Game.hpp"
#include "games/stochastic_nim/players/PerfectPlayer.hpp"
#include "util/BoostUtil.hpp"

#include <string>
#include <vector>

namespace stochastic_nim {

class PerfectPlayerGenerator : public core::AbstractPlayerGenerator<stochastic_nim::Game> {
 public:
  using Player = core::AbstractPlayer<stochastic_nim::Game>;

  PerfectPlayerGenerator(core::GameServerBase*) {}

  std::string get_default_name() const override { return "Perfect"; }
  std::vector<std::string> get_types() const override { return {"Perfect"}; }
  std::string get_description() const override { return "Perfect player"; }
  Player* generate(core::game_slot_index_t) override {
    return new PerfectPlayer(params_, &strategy_);
  }
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override {
    namespace po2 = boost_util::program_options;
    po2::parse_args(params_.make_options_description(), args);
  }

 private:
  PerfectPlayer::Params params_;
  PerfectStrategy strategy_;
};

}  // namespace stochastic_nim
