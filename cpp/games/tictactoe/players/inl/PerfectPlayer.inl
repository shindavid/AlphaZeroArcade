#include <games/tictactoe/players/PerfectPlayer.hpp>

#include <util/BitSet.hpp>

namespace tictactoe {

inline auto PerfectPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("tictactoe::PerfectPlayer options");
  return desc
      .template add_option<"strength", 's'>
          (po::value<int>(&strength)->default_value(strength),
          "strength (0-1). 0 is random, 1 is perfect.")
      ;
}

inline PerfectPlayer::PerfectPlayer(const Params& params) : params_(params) {
  util::clean_assert(params_.strength >= 0 && params_.strength <= 1, "strength must be in [0, 1]");
}

inline core::action_t PerfectPlayer::get_action(const GameState& state, const ActionMask& valid_actions) {
  if (params_.strength == 0) {
    return bitset_util::choose_random_on_index(valid_actions);
  }

  throw std::exception();  // TODO
}

}  // namespace tictactoe
