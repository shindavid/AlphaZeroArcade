#pragma once

#include <string>
#include <vector>

#include <common/AbstractPlayer.hpp>
#include <common/AbstractPlayerGenerator.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4CheatingHumanTuiPlayer.hpp>
#include <connect4/C4PerfectPlayer.hpp>

namespace c4 {

class HumanTuiPlayerGenerator : public common::AbstractPlayerGenerator<c4::GameState> {
public:
  struct Params {
    bool cheat_mode;

    auto make_options_description() {
      namespace po = boost::program_options;
      namespace po2 = boost_util::program_options;

      po2::options_description desc("c4::HumanTUIPlayer options");
      return desc
          .template add_option<"cheat-mode", 'C'>(po::bool_switch(&cheat_mode)->default_value(false),
                                                  "show winning moves")
          ;
    }
  };

  std::vector<std::string> get_types() const override { return {"TUI"}; }
  std::string get_description() const override { return "Human player"; }
  common::AbstractPlayer<c4::GameState>* generate(void* play_address) override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

private:
  Params params_;
};

}  // namespace c4

#include <connect4/inl/C4HumanTuiPlayerGenerator.inl>
