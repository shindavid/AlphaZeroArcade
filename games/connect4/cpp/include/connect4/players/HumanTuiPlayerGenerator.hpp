#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <generic_players/HumanTuiPlayerGenerator.hpp>
#include <util/BoostUtil.hpp>

#include <connect4/GameState.hpp>
#include <connect4/players/HumanTuiPlayer.hpp>

namespace c4 {

class HumanTuiPlayerGenerator : public generic::HumanTuiPlayerGenerator<c4::GameState> {
 public:
  struct Params {
    bool cheat_mode;

    auto make_options_description() {
      namespace po = boost::program_options;
      namespace po2 = boost_util::program_options;

      po2::options_description desc("c4::HumanTUIPlayer options");
      return desc.template add_option<"cheat-mode", 'C'>(
          po::bool_switch(&cheat_mode)->default_value(false), "show winning moves");
    }
  };

  core::AbstractPlayer<c4::GameState>* generate(core::game_thread_id_t) override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

 private:
  Params params_;
};

}  // namespace c4

#include <inline/connect4/players/HumanTuiPlayerGenerator.inl>
