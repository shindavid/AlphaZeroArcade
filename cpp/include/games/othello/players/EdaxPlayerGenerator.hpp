#pragma once

#include <string>
#include <vector>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/othello/Game.hpp>
#include <games/othello/players/EdaxPlayer.hpp>
#include <util/BoostUtil.hpp>

namespace othello {

class EdaxPlayerGenerator : public core::AbstractPlayerGenerator<othello::Game> {
 public:
  using Player = core::AbstractPlayer<othello::Game>;
  std::string get_default_name() const override {
    return util::create_string("Edax-%d", params_.depth);
  }
  std::vector<std::string> get_types() const override { return {"edax"}; }
  std::string get_description() const override { return "edax player"; }
  Player* generate(core::game_thread_id_t) override { return new EdaxPlayer(params_); }
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) {
    namespace po2 = boost_util::program_options;
    po2::parse_args(params_.make_options_description(), args);
  }

 private:
  EdaxPlayer::Params params_;
};

}  // namespace othello
