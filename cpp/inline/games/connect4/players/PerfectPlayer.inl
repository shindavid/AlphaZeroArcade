#include <games/connect4/players/PerfectPlayer.hpp>

#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>

namespace c4 {

inline auto PerfectPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("c4::PerfectPlayer options");
  return desc
    .template add_option<"strength", 's'>(
      po::value<int>(&strength)->default_value(strength),
      "strength (0-21). The last s moves are played perfectly, the others randomly. 0 is "
      "random, 21 is perfect.")
    .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
                                         "verbose mode")
    .template add_option<"num-oracle-procs", 'n'>(
      po::value<int>(&num_oracle_procs)->default_value(num_oracle_procs),
      "number of oracle processes to use (defaults to number of game threads)");
}

inline PerfectPlayer::PerfectPlayer(OraclePool* oracle_pool, const Params& params)
    : oracle_pool_(oracle_pool), params_(params) {
  CLEAN_ASSERT(params_.strength >= 0 && params_.strength <= 21,
                     "strength must be in [0, 21]");
}

inline void PerfectPlayer::start_game() { move_history_.reset(); }

inline void PerfectPlayer::receive_state_change(core::seat_index_t, const State&,
                                                core::action_t action) {
  move_history_.append(action);
}

}  // namespace c4
