#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/tictactoe/GameState.hpp>
#include <games/tictactoe/players/PerfectPlayer.hpp>
#include <util/BoostUtil.hpp>

namespace tictactoe {

class PerfectPlayerGenerator : public core::AbstractPlayerGenerator<tictactoe::GameState> {
 public:
  using Player = core::AbstractPlayer<tictactoe::GameState>;
  std::string get_default_name() const override { return "Perfect"; }
  std::vector<std::string> get_types() const override { return {"Perfect"}; }
  std::string get_description() const override { return "Perfect player"; }
  Player* generate(core::game_thread_id_t) override { return new PerfectPlayer(params_); }
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) {
    namespace po = boost::program_options;
    po::variables_map vm;
    po::store(po::command_line_parser(args).options(params_.make_options_description()).run(), vm);
    po::notify(vm);
  }

 private:
  PerfectPlayer::Params params_;
};

}  // namespace tictactoe
