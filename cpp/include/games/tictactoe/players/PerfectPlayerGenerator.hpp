#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "games/tictactoe/Game.hpp"
#include "games/tictactoe/players/PerfectPlayer.hpp"
#include "util/BoostUtil.hpp"

#include <string>
#include <vector>

namespace tictactoe {

class PerfectPlayerGenerator : public core::AbstractPlayerGenerator<tictactoe::Game> {
 public:
  using Player = core::AbstractPlayer<tictactoe::Game>;

  PerfectPlayerGenerator(core::GameServerBase*) {}

  std::string get_default_name() const override { return "Perfect"; }
  std::vector<std::string> get_types() const override { return {"Perfect"}; }
  std::string get_description() const override { return "Perfect player"; }
  Player* generate(core::game_slot_index_t) override {
    return new PerfectPlayer(params_);
  }
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override {
    namespace po2 = boost_util::program_options;
    po2::parse_args(params_.make_options_description(), args);
  }

 private:
  PerfectPlayer::Params params_;
};

}  // namespace tictactoe
