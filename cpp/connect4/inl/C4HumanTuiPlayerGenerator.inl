#include <connect4/C4HumanTuiPlayerGenerator.hpp>

#include <boost/program_options.hpp>

namespace c4 {

inline common::AbstractPlayer<c4::GameState>* HumanTuiPlayerGenerator::generate(void* play_address) {
  if (params_.cheat_mode) {
    return new CheatingHumanTuiPlayer();
  } else {
    return new common::HumanTuiPlayer<GameState>();
  }
}

inline void HumanTuiPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  this->parse_args_helper(params_.make_options_description(), args);
}

}  // namespace c4
