#include "core/BasicTypes.hpp"
#include "games/connect4/players/HumanTuiPlayerGenerator.hpp"

#include <boost/program_options.hpp>

namespace c4 {

inline core::AbstractPlayer<c4::Game>* HumanTuiPlayerGenerator::generate(core::game_slot_index_t) {
  return new c4::HumanTuiPlayer(params_.cheat_mode);
}

inline void HumanTuiPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  this->parse_args_helper(params_.make_options_description(), args);
}

}  // namespace c4
