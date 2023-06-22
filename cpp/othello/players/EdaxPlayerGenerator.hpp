#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <othello/GameState.hpp>
#include <othello/players/EdaxPlayer.hpp>
#include <util/BoostUtil.hpp>

namespace othello {

class EdaxPlayerGenerator : public common::AbstractPlayerGenerator<othello::GameState> {
public:
  using Player = common::AbstractPlayer<othello::GameState>;
  std::vector<std::string> get_types() const override { return {"edax"}; }
  std::string get_description() const override { return "edax player"; }
  Player* generate(common::game_thread_id_t) override { return new EdaxPlayer(params_); }
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) {
    namespace po = boost::program_options;
    po::variables_map vm;
    po::store(po::command_line_parser(args).options(params_.make_options_description()).run(), vm);
    po::notify(vm);
  }

private:
  EdaxPlayer::Params params_;
};

}  // namespace othello
